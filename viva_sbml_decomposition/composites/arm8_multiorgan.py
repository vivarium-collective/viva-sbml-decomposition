"""Arm 8: the MULTI-ORGAN SCALING arm — stretch Tellurium to its strain point.

Arms 1-7 all run ONE fixed model (17-species Teusink glycolysis). Arm 8 adds a
**synthetic scaling knob**: a multi-organ "body" of ``N`` small metabolic
compartments coupled through a shared blood pool. Growing ``N`` grows the species
and reaction count without bound (species ``= 3N + 1``, reactions ``= 4N``), so we
can MEASURE where a monolithic libroadrunner/CVODE instance strains and where the
vivarium decomposition wins at scale.

**The per-organ motif (small, so N is the knob).** Each organ ``i`` is a 3-species
enzymatic chain — ``o{i}_G -> o{i}_M -> o{i}_P`` — fed from and returning to a
single shared metabolite ``Glc_blood`` (the plasma pool):

    o{i}_uptake:  Glc_blood -> o{i}_G ;  k_up  * Glc_blood     (blood -> organ)
    o{i}_r1:      o{i}_G    -> o{i}_M ;  k1_i  * o{i}_G
    o{i}_r2:      o{i}_M    -> o{i}_P ;  k2    * o{i}_M
    o{i}_rel:     o{i}_P    -> Glc_blood ; k_rel * o{i}_P       (organ -> blood)

Every organ draws from and feeds back to the SAME ``Glc_blood``, so the ``N`` organs
are genuinely coupled (contrast arm 7, which is ``N`` *independent* monoliths). The
loop is closed, so total mass is conserved and the body relaxes to a finite steady
state at any ``N`` — CVODE stays well-posed as the knob grows. Per-organ rate
``k1_i = 1 + 0.05 i`` is mildly heterogeneous so the organs are non-degenerate.

**Two implementations (the comparison).**

1. **Monolithic Tellurium** (:func:`glycolysis_multiorgan_mono`) — the whole
   ``N``-organ body as ONE Antimony model (:func:`build_multiorgan_antimony`),
   integrated by a single libroadrunner/CVODE instance. This is what strains: the
   libroadrunner model BUILD (compile) is superlinear in the reaction count, and
   the dense CVODE step grows superlinearly in species.

2. **Decomposed (vivarium)** (:func:`glycolysis_multiorgan_decomposed`) — each
   organ its own :class:`OrganProcess` (a libroadrunner instance built from that
   organ's 4 reactions only), all sharing one ``blood`` store. Exactly arm 3's
   additive-delta pattern: ``Glc_blood`` lives in the shared store and every organ
   writes the per-interval **delta** it produced on it, so the deltas reconcile
   **additively** through the bigraph apply path (a Jacobi operator split of the
   blood ODE). Organ-private species (``o{i}_G/M/P``) also live in that store but
   are written by their one organ only (no double counting). Building the body this
   way compiles ``N`` *small* models (one libroadrunner setup per organ) instead of
   one giant one.

**The strain curve** (:func:`multiorgan_scaling`) captures, per ``N``, the honest
END-TO-END picture: ``mono_build_s`` (the one big libroadrunner compile — where
Tellurium strains) and ``mono_run_s`` (raw CVODE integration) and their sum
``mono_total_s``; ``decomp_build_s`` (the ``N`` small compiles) and ``decomp_run_s``
(the vivarium composite run — orchestration + ``N`` small integrations) and their sum
``decomp_total_s``; and ``accuracy`` (decomposed vs the monolithic reference).

**Measured regime (recorded 2026-10-25, M-series laptop; see the figure/study).**
The one genuinely strong, well-measured result is that the **monolithic BUILD is
superlinear in the reaction count** — ``0.20 s`` at 97 species rising to ``18.5 s``
at 1537 species (~94x for ~16x the reactions, ~O(N^1.64)) — with the dense-CVODE run
blowing up alongside it (``0.0023 s -> 2.56 s``). That is the real strain point.

But decomposition is **NOT a free win**, and the arm is framed honestly end-to-end:

- The decomposed BUILD is ``N`` separate libroadrunner setups, so it is *slower* than
  the single monolithic compile up to ~100 species (``1.7x`` slower at 97 species),
  reaches rough **parity** around 385 species (``1.40`` vs ``1.76 s``) and only becomes
  a clear win at very large ``N`` (``2.4x`` at 1537 species). It does not "build several-
  fold faster by hundreds of species".
- The decomposed per-tick RUN overhead dwarfs the monolith's CVODE integration
  (``~16x`` at 385 species: ``0.95 s`` vs ``0.058 s``), so the body is **SLOWER
  END-TO-END at every ``N`` measured** (``decomp_total 2.35 s`` vs ``mono_total 1.81 s``
  at 385 species). The end-to-end gap narrows as ``N`` grows, but the decomposed run is
  left UNMEASURED past ``N=128`` (too slow for the harness), so **no end-to-end crossover
  is claimed** — decomposition trades the monolith's superlinear build for much higher
  run overhead and would only pay off end-to-end at models larger than measured.
- Accuracy (additive-split error) is first-order in the coupling interval (halves as the
  interval halves) and degrades with coupling density: ``max_rel_err`` ``~2e-3`` at ``N=2``,
  ``~1.2e-2`` at ``N=8``, ``~5.2e-2`` at ``N=32``, ``~0.31`` at ``N=128``. So the speed gap
  narrows exactly where accuracy becomes unusable — the speed and acceptable-accuracy
  regimes do NOT co-occur at a fixed interval.

Numbers are REAL — re-run :func:`multiorgan_scaling` to refresh; do not hand-edit.
"""
from __future__ import annotations

import copy
import statistics
import time
from functools import lru_cache
from typing import List, Sequence

import numpy as np
from process_bigraph import Composite, Process, allocate_core
from process_bigraph.composite_generator import composite_generator
from process_bigraph.emitter import RAMEmitter
from viva_tellurium.composites import make_tellurium_document, register_tellurium

from viva_sbml_decomposition.harness.accuracy import compare_to_reference

from ._emit import strip_emitter
from ._run import extract_run

# --- the synthetic body ------------------------------------------------------

BLOOD = "Glc_blood"          # the single shared (plasma) metabolite
BLOOD_INITIAL = 100.0        # all mass starts in the blood pool
PER_ORGAN_SPECIES = ("G", "M", "P")   # 3-species enzymatic chain per organ
T_END = 10.0                 # the body relaxes to steady state well before this

# Per-organ kinetics (shared across organs except k1, which is mildly
# heterogeneous so the organs are non-degenerate). k_up small enough that the N
# organs do not collectively drain the pool in a single coupling interval.
K_UP, K2, K_REL = 0.1, 0.5, 0.3

ABSOLUTE_TOLERANCE = 1e-10
RELATIVE_TOLERANCE = 1e-8

# Defaults for the workbench generators.
DEFAULT_N_ORGANS = 8
DEFAULT_N_STEPS = 200        # monolithic emitter resolution (interval = T_END/N_STEPS)

# Decomposed coupling interval = the scheduler step at which the shared blood
# store reconciles every organ's delta. The additive-split error is first-order
# in this interval (halves as it halves); 0.05 keeps the default body well within
# tolerance at the N a unit test affords.
DEFAULT_COUPLING_INTERVAL = 0.05


def _k1(i: int) -> float:
    return 1.0 + 0.05 * i


def organ_species(i: int) -> List[str]:
    """The private floating-species ids of organ ``i`` (``o{i}_G/M/P``)."""
    return [f"o{i}_{s}" for s in PER_ORGAN_SPECIES]


def all_species(n: int) -> List[str]:
    """Every floating species of the ``n``-organ body: the blood pool + all organs."""
    out = [BLOOD]
    for i in range(n):
        out.extend(organ_species(i))
    return out


def _organ_reactions(i: int) -> str:
    g, m, p = organ_species(i)
    return (
        f"  {g} = 0; {m} = 0; {p} = 0;\n"
        f"  o{i}_uptake: {BLOOD} -> {g}; k_up * {BLOOD};\n"
        f"  o{i}_r1: {g} -> {m}; {_k1(i):.6f} * {g};\n"
        f"  o{i}_r2: {m} -> {p}; k2 * {m};\n"
        f"  o{i}_rel: {p} -> {BLOOD}; k_rel * {p};\n"
    )


def _header() -> str:
    return (f"  {BLOOD} = {BLOOD_INITIAL};\n"
            f"  k_up = {K_UP}; k2 = {K2}; k_rel = {K_REL};\n")


def build_multiorgan_antimony(n: int) -> str:
    """The whole ``n``-organ body as ONE Antimony model (the monolithic arm).

    Species ``= 3n + 1`` (the blood pool + 3 per organ), reactions ``= 4n``. This
    is the string handed to the single libroadrunner instance; its compile time is
    what strains as ``n`` grows. Also surfaced on the monolithic generator's
    ``config.model`` so the workbench Model page can show it.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    body = "".join(_organ_reactions(i) for i in range(n))
    return f"model multiorgan_{n}()\n{_header()}{body}end"


def single_organ_antimony(i: int) -> str:
    """One organ (``blood`` pool + organ ``i``'s 3 species and 4 reactions).

    Each decomposed :class:`OrganProcess` builds its own libroadrunner instance
    from this — ``N`` small compiles (linear in ``N``) rather than one giant one.
    """
    return f"model organ_{i}()\n{_header()}{_organ_reactions(i)}end"


# --- decomposed arm: one process per organ, shared blood store ---------------


def _load(antimony):
    import tellurium as te

    rr = te.loada(antimony)
    rr.integrator.absolute_tolerance = ABSOLUTE_TOLERANCE
    rr.integrator.relative_tolerance = RELATIVE_TOLERANCE
    return rr


class OrganProcess(Process):
    """One organ as a standalone bigraph process node, coupled via the blood store.

    Holds a single libroadrunner instance built from :func:`single_organ_antimony`
    (this organ's 4 reactions). Each ``update(interval)`` reads the shared
    ``Glc_blood`` and this organ's own species from the store, integrates its
    reactions over the interval, and returns the **delta** it produced. Because the
    store's ``map[float]`` applies numeric updates additively, every organ's delta
    on ``Glc_blood`` reconciles by summation through the bigraph apply path — the
    genuine multi-organ coupling. Organ-private species are written by this organ
    only, so they carry no competing writer.
    """

    config_schema = {
        "organ_index": {"_type": "integer", "_default": 0},
        "absolute_tolerance": {"_type": "float", "_default": ABSOLUTE_TOLERANCE},
        "relative_tolerance": {"_type": "float", "_default": RELATIVE_TOLERANCE},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self._rr = None
        self._own = organ_species(int(self.config["organ_index"]))

    def inputs(self):
        return {"species": "map[float]"}

    def outputs(self):
        # Plain map[float] (NOT overwrite): the update is a DELTA that applies
        # additively and reconciles by summation across organ nodes (on Glc_blood).
        return {"species": "map[float]"}

    def _build(self):
        if self._rr is not None:
            return
        self._rr = _load(single_organ_antimony(int(self.config["organ_index"])))

    def update(self, state, interval):
        self._build()
        rr = self._rr
        incoming = (state.get("species") if state else None) or {}
        blood0 = float(incoming.get(BLOOD, BLOOD_INITIAL))
        own0 = {s: float(incoming.get(s, 0.0)) for s in self._own}
        rr.reset()  # clock -> 0, species -> model defaults
        rr[BLOOD] = blood0
        for s in self._own:
            rr[s] = own0[s]
        try:
            rr.oneStep(0.0, interval)
        except RuntimeError:
            # CVODE failed on this organ: write non-finite deltas so the arm
            # surfaces as diverged (accuracy harness) instead of crashing.
            out = {BLOOD: float("nan")}
            out.update({s: float("nan") for s in self._own})
            return {"species": out}
        out = {BLOOD: float(rr[BLOOD]) - blood0}
        out.update({s: float(rr[s]) - own0[s] for s in self._own})
        return {"species": out}


def register_multiorgan(core=None):
    """Register :class:`OrganProcess` + the RAM emitter on ``core``."""
    if core is None:
        core = allocate_core()
    core.register_link("OrganProcess", OrganProcess)
    core.register_link("ram-emitter", RAMEmitter)
    return core


def _initial_blood_store(n: int) -> dict:
    """t=0 state of the shared store: all mass in the blood pool, organs empty."""
    store = {BLOOD: BLOOD_INITIAL}
    for i in range(n):
        for s in organ_species(i):
            store[s] = 0.0
    return store


def _build_decomposed_document(n: int, coupling_interval: float) -> dict:
    """The decomposed composite: one :class:`OrganProcess` per organ + a shared
    ``blood`` store + a RAM emitter.

    The ``blood`` store is the single shared vivarium store all organ nodes
    reconcile their ``Glc_blood`` deltas through each ``coupling_interval``; it also
    carries each organ's private species (single-writer). The emitter aggregates it
    so :func:`extract_run` can read the full-body trajectory back.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    species_schema = {s: "float" for s in all_species(n)}
    doc = {
        "blood": _initial_blood_store(n),
        "emitter": {
            "_type": "step",
            "address": "local:ram-emitter",
            "config": {"emit": {"species": "map[float]", "time": "float",
                                "global_time": "float"}},
            "inputs": {"species": ["blood"], "time": ["global_time"],
                       "global_time": ["global_time"]},
        },
    }
    for i in range(n):
        doc[f"organ_{i}"] = {
            "_type": "process",
            "address": "local:OrganProcess",
            "config": {"organ_index": i,
                       "absolute_tolerance": ABSOLUTE_TOLERANCE,
                       "relative_tolerance": RELATIVE_TOLERANCE},
            "interval": coupling_interval,
            "inputs": {"species": ["blood"]},
            "outputs": {"species": ["blood"]},
        }
    return doc


# --- composite generators ----------------------------------------------------


@composite_generator(
    name="glycolysis_multiorgan_mono",
    description=(
        "Arm 8 (monolithic): the whole N-organ synthetic body as ONE Antimony/"
        "Tellurium model integrated by a single libroadrunner/CVODE instance. The "
        "scaling knob (N organs, species=3N+1, reactions=4N) is what strains — the "
        "libroadrunner model build is superlinear in reactions. config.model carries "
        "the N-organ Antimony so the Model page can show it."
    ),
    parameters={"n_organs": {"type": "integer", "default": DEFAULT_N_ORGANS}},
    core_extensions=[register_tellurium],
    fidelity="real",
    methods=["single libroadrunner/CVODE instance over the whole N-organ body"],
)
def glycolysis_multiorgan_mono(core=None, n_organs: int = DEFAULT_N_ORGANS):
    """Monolithic N-organ body in one TelluriumProcess (emitter stripped)."""
    doc = make_tellurium_document(
        model=build_multiorgan_antimony(n_organs),
        model_format="antimony",
        absolute_tolerance=ABSOLUTE_TOLERANCE,
        relative_tolerance=RELATIVE_TOLERANCE,
        interval=T_END / DEFAULT_N_STEPS,
    )
    return strip_emitter(doc)


@composite_generator(
    name="glycolysis_multiorgan_decomposed",
    description=(
        "Arm 8 (decomposed): the N-organ body as N per-organ OrganProcess nodes, "
        "each its own small libroadrunner instance, coupled through ONE shared "
        "blood store — every organ writes its per-interval Glc_blood delta, which "
        "reconcile additively through the bigraph apply path (arm-3 pattern). "
        "Builds N small models (one libroadrunner setup per organ) instead of one "
        "giant one — a structural build-vs-run trade, not a guaranteed speed win."
    ),
    parameters={
        "n_organs": {"type": "integer", "default": DEFAULT_N_ORGANS},
        "coupling_interval": {"type": "float", "default": DEFAULT_COUPLING_INTERVAL},
    },
    core_extensions=[register_multiorgan],
    fidelity="real",
    methods=[
        "N per-organ libroadrunner/CVODE instances (one reaction-subset each)",
        "shared blood store, additive Glc_blood delta coupling (bigraph apply path)",
    ],
)
def glycolysis_multiorgan_decomposed(core=None, n_organs: int = DEFAULT_N_ORGANS,
                                     coupling_interval: float = DEFAULT_COUPLING_INTERVAL):
    """Decomposed N-organ composite: per-organ processes + shared blood (stripped)."""
    return strip_emitter(_build_decomposed_document(n_organs, coupling_interval))


# --- run helpers (unstripped documents, for accuracy + trajectories) ---------


def run_mono(n_organs: int, *, n_steps: int = DEFAULT_N_STEPS):
    """Run the monolithic body directly through libroadrunner; return ``(traj, t)``.

    Uses ``te.loada`` + ``simulate`` (raw CVODE, no framework) so the monolithic
    integration is measured and compared without vivarium orchestration — the
    accuracy reference for the decomposed arm.
    """
    rr = _load(build_multiorgan_antimony(n_organs))
    cols = ["time"] + all_species(n_organs)
    result = rr.simulate(0, T_END, n_steps + 1, cols)
    arr = np.asarray(result)
    t = arr[:, 0].astype(float)
    traj = {s: arr[:, j + 1].astype(float) for j, s in enumerate(all_species(n_organs))}
    return traj, t


def run_decomposed(core=None, n_organs: int = DEFAULT_N_ORGANS,
                   coupling_interval: float = DEFAULT_COUPLING_INTERVAL):
    """Run the decomposed composite through process-bigraph; return ``(traj, t)``.

    Uses the *unstripped* document (library RAM emitter) so the full-body history
    can be read back over :func:`all_species`.
    """
    if core is None:
        core = allocate_core()
    register_multiorgan(core)
    doc = _build_decomposed_document(n_organs, coupling_interval)
    return extract_run(doc, core, T_END, all_species(n_organs))


# --- the strain curve --------------------------------------------------------


def _median_time(fn, repeats: int) -> float:
    fn()  # warmup
    xs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        xs.append(time.perf_counter() - t0)
    return statistics.median(xs)


def multiorgan_scaling(ns: Sequence[int] = (2, 8, 32), *,
                       coupling_interval: float = DEFAULT_COUPLING_INTERVAL,
                       repeats: int = 3, run_decomposed_max_n: int | None = None) -> List[dict]:
    """Measure the honest end-to-end strain curve, monolithic vs decomposed, vs ``N``.

    For each ``n`` in ``ns`` returns ``{n, species, mono_build_s, mono_run_s,
    mono_total_s, decomp_build_s, decomp_run_s, decomp_total_s, accuracy}``:

    - ``species`` = ``3n + 1`` (the scaling knob's species count).
    - ``mono_build_s`` — median wall-clock of ``te.loada(build_multiorgan_antimony(n))``,
      i.e. the ONE big libroadrunner model COMPILE. **This is where Tellurium
      strains** — superlinear in the reaction count (~O(N^1.65)).
    - ``mono_run_s`` — median raw-CVODE integration of the monolith over ``T_END``.
    - ``mono_total_s`` = ``mono_build_s + mono_run_s`` (compile once, then integrate).
    - ``decomp_build_s`` — median wall-clock to compile the ``N`` SMALL per-organ
      models (``te.loada(single_organ_antimony(i))`` for every organ). This is the
      decomposed build cost; it is roughly LINEAR in ``N`` but carries ``N`` separate
      libroadrunner setups, so at small/moderate ``N`` it is SLOWER than the single
      monolithic compile, and only undercuts it once the monolith's superlinearity
      dominates (very large ``N``).
    - ``decomp_run_s`` — median run of the decomposed vivarium composite over
      ``T_END`` (construction excluded; orchestration + ``N`` small integrations).
      This carries per-tick framework overhead the raw ``mono_run_s`` does not, so
      it is MUCH larger than the monolith's CVODE run. ``None`` when ``n`` exceeds
      ``run_decomposed_max_n`` (then ``decomp_total_s``/``accuracy`` are ``None`` too:
      an honestly UNMEASURED point, not a guess).
    - ``decomp_total_s`` = ``decomp_build_s + decomp_run_s`` (or ``None``).
    - ``accuracy`` — decomposed-vs-monolithic ``max_rel_err`` over all species
      (:func:`compare_to_reference`), the additive-split error at ``coupling_interval``
      (``None`` when the decomposed run was skipped).

    All numbers are whatever was measured. The headline is honest end-to-end:
    the monolithic BUILD is superlinear, but decomposition does NOT build faster
    until very large ``N`` and its per-tick RUN overhead makes it SLOWER end-to-end
    at the moderate ``N`` a test/affordable measurement can reach. Re-run to refresh.
    """
    import tellurium as te

    out: List[dict] = []
    for n in ns:
        antimony = build_multiorgan_antimony(n)
        mono_build_s = _median_time(lambda a=antimony: te.loada(a), repeats)

        rr = _load(antimony)
        cols = ["time"] + all_species(n)
        mono_run_s = _median_time(
            lambda rr=rr, cols=cols: (rr.reset(), rr.simulate(0, T_END, 51, cols)),
            repeats)

        # Decomposed BUILD = compiling the N SMALL per-organ models (the work
        # Composite construction defers to each OrganProcess._build). Timed the same
        # way as mono_build_s (te.loada), so the two builds are apples-to-apples.
        organ_srcs = [single_organ_antimony(i) for i in range(n)]
        decomp_build_s = _median_time(
            lambda srcs=organ_srcs: [te.loada(s) for s in srcs], repeats)

        decomp_run_s = decomp_total_s = accuracy = None
        if run_decomposed_max_n is None or n <= run_decomposed_max_n:
            # Decomposed run, construction excluded: build a fresh composite per
            # repeat OUTSIDE the timer, time only comp.run(T_END).
            doc = _build_decomposed_document(n, coupling_interval)

            def _decomp_once(doc=doc):
                core = allocate_core()
                register_multiorgan(core)
                comp = Composite({"state": copy.deepcopy(doc)}, core=core)
                t0 = time.perf_counter()
                comp.run(T_END)
                return time.perf_counter() - t0

            _decomp_once()  # warmup
            decomp_run_s = statistics.median(_decomp_once() for _ in range(repeats))
            decomp_total_s = decomp_build_s + decomp_run_s

            ref_traj, ref_t = run_mono(n)
            arm_traj, arm_t = run_decomposed(n_organs=n, coupling_interval=coupling_interval)
            accuracy = float(compare_to_reference(ref_traj, ref_t, arm_traj, arm_t).max_rel_err)

        out.append({
            "n": int(n),
            "species": 3 * n + 1,
            "mono_build_s": float(mono_build_s),
            "mono_run_s": float(mono_run_s),
            "mono_total_s": float(mono_build_s + mono_run_s),
            "decomp_build_s": float(decomp_build_s),
            "decomp_run_s": None if decomp_run_s is None else float(decomp_run_s),
            "decomp_total_s": None if decomp_total_s is None else float(decomp_total_s),
            "accuracy": accuracy,
        })
    return out
