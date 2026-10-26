"""A native process-bigraph process that integrates an SBML model with COPASI.

:class:`CopasiProcess` is the COPASI/basico analogue of viva-tellurium's
``TelluriumProcess``: it holds one COPASI model and advances it over a shared
``species`` store, one coupling interval per ``update``. It supports two modes
so it can play both roles arm 10 needs:

* ``coupling_mode="overwrite"`` (the default) — the **monolithic second engine**.
  There is a single writer, so the process integrates its own model
  *continuously* (each interval continues from COPASI's own current state) and
  writes the absolute 17-species state back with ``overwrite`` semantics, plus
  the current time. This is COPASI-as-a-second-engine on the whole Teusink model
  and is validated against the libroadrunner reference.

* ``coupling_mode="delta"`` — the **cross-engine coupling** role. The process is
  built from a *reaction subset* of the model (all species/parameters/rules
  retained; the complementary reactions removed, exactly as arm 3's
  ``SubmodelProcess`` does for Tellurium). Each ``update`` injects the shared
  store's settable-species values as COPASI initial concentrations, integrates
  the subset over the interval, and returns the **delta** it produced on the
  settable species. Because the shared ``map[float]`` store applies numeric
  updates additively, a COPASI delta and a Tellurium delta written to the same
  store reconcile by summation — two different SBML engines composing through
  the pbg typed store.

**basico API used — build vs. hot loop.** The model is loaded once (see the
.cps caveat below); ``basico.get_species(model=dm)`` / ``basico.get_reactions(
model=dm)`` are called ONCE at build to read the canonical ``sbml_id`` / ``type``
columns, and ``basico.set_task_settings(basico.T.TIME_COURSE, {...}, model=dm)``
sets the LSODA tolerances to arm 1's. The per-interval HOT LOOP never calls
``basico.set_species``/``get_species`` — those are per-NAME dataframe-rebuilding
calls (~290 µs each → ~10 ms/update for 17 species, dwarfing the ~0.3 ms solve
and making COPASI look ~30x slower than it is). Instead it writes/reads the
cached ``COPASI.CMetab`` handles directly (``setInitialConcentration`` +
``model.updateInitialValues(Framework_Concentration)``; ``getConcentration``),
which is a genuine bulk write/read (~0.5 ms set / ~0 ms get per update) and is
verified bit-identical to basico's per-name ``set_species``/``get_species`` path
(max |Δ| ~1.6e-13 over a step) — so the fast path is pinned, not a silent no-op.
Each injected state is integrated with ``run_time_course(..., intervals=1,
use_initial_values=True)``, so the step starts from exactly the injected initial
values (no reliance on retained transient state).

**Two caveats, both handled structurally:**

1. *Global current model.* basico/COPASI keep a single global current model and a
   global datamodel container. Every operation here passes ``model=dm`` and calls
   ``set_current_model(dm)`` before touching the model, and the datamodels are
   cached for the process lifetime (never GC'd mid-process), so instances never
   collide and the GC-race abort cannot fire. The matching constraint: two
   CopasiProcess nodes sharing the SAME model source in ONE composite would share
   one datamodel; arm 10 never does this (one COPASI node per composite).

2. *Flaky SBML import.* ``COPASI.importSBMLFromString`` (what
   ``basico.load_model_from_string`` calls) aborts the whole process
   *nondeterministically* on this python-copasi build (an uncatchable C++
   ``abort()`` in the SBML parser, ~1-in-5). Loading COPASI's own ``.cps`` native
   format is rock-solid (100/100 loads here). So the SBML import is quarantined to
   a one-shot SUBPROCESS (with retries) that writes a disk-cached ``.cps`` keyed
   by the SBML's content hash; the in-process code only ever loads the stable
   ``.cps``. The risky parse runs at most a few times per distinct model per
   machine (then the cache hits), and a crash there kills only the throwaway
   subprocess, never the composite / test run.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
from typing import ClassVar, Dict

import libsbml
from process_bigraph import Process, allocate_core

# arm-1-matched integrator tolerances (COPASI default LSODA is 1e-6/1e-12).
ABSOLUTE_TOLERANCE = 1e-10
RELATIVE_TOLERANCE = 1e-8

# Process-lifetime cache of loaded COPASI models, keyed by the exact SBML string.
#
# WHY A CACHE (the global-model caveat, made concrete). basico/COPASI keep a
# single *global* root container of datamodels, and the current model is global
# state. Loading a fresh datamodel per CopasiProcess instance means every
# composite build adds another model to that container, and when a dropped
# ``Composite`` lets its CopasiProcess (hence the SWIG ``CDataModel`` wrapper) be
# garbage-collected, COPASI removes that datamodel from the container. A GC that
# fires while another COPASI model is live — exactly what happens when one
# composite is dropped and the next is built in the same process (across tests,
# across a timing loop) — corrupts the global container and COPASI hard-
# ``abort()``s *nondeterministically* (an uncatchable C++ abort, reproduced here
# as intermittent SIGABRT).
#
# The cache removes the race at its source: a given SBML string is loaded into
# COPASI exactly ONCE and reused for the whole process lifetime, so the container
# holds a tiny fixed set (here 2: the full model and the lower-glycolysis subset)
# and nothing is ever removed mid-process. Reuse is numerically clean because
# every ``update`` injects the incoming state as COPASI initial concentrations and
# integrates one interval with ``use_initial_values=True`` — the model never
# relies on retained transient state, so a reused model behaves identically to a
# freshly loaded one. The matching constraint: two CopasiProcess nodes that share
# the SAME SBML in ONE composite would share one datamodel and clobber each
# other's per-step state; arm 10's composites never do this (one COPASI node
# each), and a future multi-COPASI-node composite would need distinct models.
_MODEL_CACHE: dict = {}

# Disk cache of COPASI ``.cps`` files, keyed by the SBML content hash. Generated
# once per distinct model per machine (via the quarantined subprocess below), then
# loaded natively — the stable path around the flaky SBML importer.
_CPS_CACHE_DIR = os.path.join(tempfile.gettempdir(), "viva_sbml_decomposition_copasi_cps")

# Subprocess body: load an SBML file with basico and save COPASI's .cps. Runs the
# flaky ``importSBMLFromString`` in isolation, so its (nondeterministic) abort
# kills only this throwaway process; the parent retries.
_CPS_GEN_SRC = (
    "import sys, basico\n"
    "dm = basico.load_model(sys.argv[1])\n"
    "assert dm is not None\n"
    "basico.save_model(sys.argv[2], model=dm)\n"
)


def _ensure_cps(sbml):
    """Return a path to a COPASI ``.cps`` for ``sbml``, generating it if needed.

    The SBML → .cps conversion (which runs COPASI's flaky SBML importer) happens
    in a retrying throwaway subprocess, so an abort there never touches the caller.
    """
    os.makedirs(_CPS_CACHE_DIR, exist_ok=True)
    digest = hashlib.sha256(sbml.encode("utf-8")).hexdigest()[:32]
    cps_path = os.path.join(_CPS_CACHE_DIR, digest + ".cps")
    if os.path.exists(cps_path):
        return cps_path

    xml_path = cps_path + ".sbml"
    with open(xml_path, "w", encoding="utf-8") as fh:
        fh.write(sbml)

    last = None
    for _ in range(6):  # importSBMLFromString aborts ~1-in-5; P(all 6 fail) ~ 6e-5
        tmp = f"{cps_path}.{os.getpid()}.tmp"
        proc = subprocess.run(
            [sys.executable, "-c", _CPS_GEN_SRC, xml_path, tmp],
            capture_output=True, text=True,
        )
        if proc.returncode == 0 and os.path.exists(tmp):
            os.replace(tmp, cps_path)   # atomic publish
            return cps_path
        last = proc
        if os.path.exists(tmp):
            os.remove(tmp)
    raise RuntimeError(
        "COPASI could not import the SBML model after 6 subprocess attempts "
        f"(returncode={getattr(last, 'returncode', '?')}); "
        f"stderr tail: {(getattr(last, 'stderr', '') or '')[-500:]}"
    )


class _CopasiModel:
    """A loaded COPASI model + cached handles, shared across CopasiProcess runs."""

    __slots__ = ("dm", "model", "metabs", "conc_framework",
                 "species_ids", "settable", "initial")

    def __init__(self, dm, model, metabs, conc_framework, species_ids, settable, initial):
        self.dm = dm
        self.model = model
        self.metabs = metabs
        self.conc_framework = conc_framework
        self.species_ids = species_ids
        self.settable = settable
        self.initial = initial  # sbml_id -> initial concentration (for seeding)


def _get_or_build_model(sbml, reactions, rtol, atol):
    """Return the cached :class:`_CopasiModel` for ``sbml`` (loading once)."""
    import COPASI
    import basico

    cached = _MODEL_CACHE.get(sbml)
    if cached is not None:
        return cached

    # Load COPASI's native .cps (generated once, out-of-process) — NOT
    # load_model_from_string, whose SBML parser aborts nondeterministically.
    dm = basico.load_model(_ensure_cps(sbml))
    if dm is None:
        raise ValueError("COPASI failed to load the model source.")
    basico.set_current_model(dm)

    sp = basico.get_species(model=dm)
    species_ids = [str(row.sbml_id) for _, row in sp.iterrows()]
    settable = [str(row.sbml_id) for _, row in sp.iterrows() if row.type != "assignment"]

    if reactions:
        present = set(basico.get_reactions(model=dm).sbml_id)
        unknown = [r for r in reactions if r not in present]
        if unknown:
            raise ValueError(
                f"reactions {unknown} are not reactions of the model subset "
                f"(have {sorted(present)})"
            )

    model = dm.getModel()
    metabs = {model.getMetabolite(i).getSBMLId(): model.getMetabolite(i)
              for i in range(model.getNumMetabs())}
    initial = {sid: float(metabs[sid].getInitialConcentration()) for sid in species_ids}

    # Match LSODA tolerances to arm 1's CVODE tolerances. Best-effort: a basico
    # that does not expose the setter leaves COPASI's defaults (1e-6/1e-12).
    try:
        basico.set_task_settings(
            basico.T.TIME_COURSE,
            {"method": {"Relative Tolerance": float(rtol),
                        "Absolute Tolerance": float(atol)}},
            model=dm,
        )
    except Exception:
        pass

    cm = _CopasiModel(dm, model, metabs, COPASI.CCore.Framework_Concentration,
                      species_ids, settable, initial)
    _MODEL_CACHE[sbml] = cm
    return cm


def _antimony_to_sbml(model_source: str, model_format: str) -> str:
    """Return SBML for ``model_source`` (converting Antimony via tellurium)."""
    if model_format == "sbml":
        return model_source
    import tellurium as te

    return te.antimonyToSBML(model_source)


def _subset_sbml(base_sbml: str, keep_reactions) -> str:
    """SBML keeping only ``keep_reactions`` (all species/params/rules retained).

    Identical partitioning to arm 3's ``_submodel_sbml`` so the COPASI subset and
    the Tellurium subset are carved from the same base along the same seam.
    """
    doc = libsbml.readSBMLFromString(base_sbml)
    model = doc.getModel()
    keep = set(keep_reactions)
    for rid in [model.getReaction(i).getId() for i in range(model.getNumReactions())]:
        if rid not in keep:
            model.removeReaction(rid)
    return libsbml.writeSBMLToString(doc)


class CopasiProcess(Process):
    """Integrate an SBML model (or a reaction subset of it) with COPASI/basico.

    See the module docstring for the two coupling modes and the global-model
    caveat. The ``species`` ports exchange concentrations keyed by SBML id
    (the model's canonical species ids, matching every other arm).
    """

    config_schema: ClassVar[Dict] = {
        "model": {"_type": "string", "_default": ""},
        "model_format": {"_type": "string", "_default": "antimony"},
        "reactions": {"_type": "list[string]", "_default": []},
        "coupling_mode": {"_type": "string", "_default": "overwrite"},
        "absolute_tolerance": {"_type": "float", "_default": ABSOLUTE_TOLERANCE},
        "relative_tolerance": {"_type": "float", "_default": RELATIVE_TOLERANCE},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self._cm = None            # the shared _CopasiModel (cached per SBML)
        self._t = 0.0              # overwrite mode: accumulated model time

    # -- ports ------------------------------------------------------------- #
    def inputs(self):
        return {"species": "map[float]"}

    def outputs(self):
        if self.config["coupling_mode"] == "delta":
            # Plain map[float]: the update is a DELTA that applies additively and
            # reconciles by summation with the other engine's delta.
            return {"species": "map[float]"}
        # Monolithic second engine: a single writer overwrites the absolute state.
        return {"species": "overwrite[map[float]]", "time": "overwrite[float]"}

    # -- model build ------------------------------------------------------- #
    def _build(self):
        if self._cm is not None:
            return
        cfg = self.config
        if not cfg["model"]:
            raise ValueError("CopasiProcess requires a 'model' source.")
        base_sbml = _antimony_to_sbml(cfg["model"], cfg["model_format"])
        reactions = list(cfg["reactions"])
        sbml = _subset_sbml(base_sbml, reactions) if reactions else base_sbml
        self._cm = _get_or_build_model(
            sbml, reactions, cfg["relative_tolerance"], cfg["absolute_tolerance"])

    # -- helpers ----------------------------------------------------------- #
    def _read(self, ids):
        """Current (transient) concentrations for ``ids`` via cached metab handles."""
        metabs = self._cm.metabs
        return {sid: float(metabs[sid].getConcentration()) for sid in ids}

    def _inject_and_step(self, base, interval):
        """Set ``base`` as settable initial conc, integrate one interval.

        Direct COPASI handle writes (``setInitialConcentration`` +
        ``updateInitialValues``) are ~30x faster than basico's dataframe-building
        ``set_species`` (verified bit-identical). ``use_initial_values=True`` makes
        the integration start from the injected state, so neither mode relies on
        COPASI retaining transient state between calls — which is what lets the
        datamodel be safely reused across runs (see ``_MODEL_CACHE``).
        """
        import basico

        cm = self._cm
        for sid in cm.settable:
            cm.metabs[sid].setInitialConcentration(base[sid])
        cm.model.updateInitialValues(cm.conc_framework)
        basico.run_time_course(
            duration=float(interval), intervals=1,
            use_initial_values=True, model=cm.dm,
        )

    def get_species_ids(self):
        self._build()
        return list(self._cm.species_ids)

    # -- lifecycle --------------------------------------------------------- #
    def initial_state(self):
        self._build()
        if self.config["coupling_mode"] == "delta":
            # The shared store is seeded by the composite document (arm 3 style),
            # so the subset process contributes no initial state of its own.
            return {}
        # Seed the shared store with the model's INITIAL concentrations (read from
        # the model definition, not a possibly-reused transient state).
        return {"species": dict(self._cm.initial), "time": 0.0}

    def update(self, state, interval):
        import basico

        self._build()
        # Pin the global current model to THIS instance's model before any op, so a
        # COPASI node composed beside other COPASI models never acts on the wrong
        # one (the basico global-current-model caveat).
        basico.set_current_model(self._cm.dm)
        if self.config["coupling_mode"] == "delta":
            return self._update_delta(state, interval)
        return self._update_overwrite(state, interval)

    def _update_overwrite(self, state, interval):
        """Inject the incoming absolute state, integrate, overwrite all 17 + time.

        Injecting the previous step's settable state and integrating one interval
        is equivalent to a continuous integration (the state is Markovian; the
        assignment-rule species ATP/ADP/AMP are recomputed by COPASI from the
        settable phosphate pool), and keeps every step self-contained so the
        cached model can be reused run-to-run.
        """
        cm = self._cm
        incoming = (state.get("species") if state else None) or {}
        base = {
            sid: float(incoming[sid]) if sid in incoming else cm.initial[sid]
            for sid in cm.settable
        }
        self._inject_and_step(base, interval)
        self._t += float(interval)
        return {"species": self._read(cm.species_ids), "time": self._t}

    def _update_delta(self, state, interval):
        """Inject shared settable state, integrate the subset, return the delta."""
        cm = self._cm
        incoming = (state.get("species") if state else None) or {}
        base = {
            sid: float(incoming[sid]) if sid in incoming
            else float(cm.metabs[sid].getConcentration())
            for sid in cm.settable
        }
        try:
            self._inject_and_step(base, interval)
        except Exception:
            # Left the sqrt domain / integrator failed: write a non-finite delta
            # so the arm surfaces as diverged instead of crashing the composite.
            return {"species": {sid: float("nan") for sid in cm.settable}}
        after = self._read(cm.settable)
        return {"species": {sid: after[sid] - base[sid] for sid in cm.settable}}


def register_copasi(core=None):
    """Register :class:`CopasiProcess` + the RAM emitter on ``core``."""
    from process_bigraph.emitter import RAMEmitter

    if core is None:
        core = allocate_core()
    core.register_link("CopasiProcess", CopasiProcess)
    core.register_link("ram-emitter", RAMEmitter)
    return core
