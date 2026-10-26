"""Per-organ DECOMPOSITION of the multi-organ body + its strain-curve measurement.

Not a registered composite generator: the generic composite layer covers the
monolithic body (``engine_monolithic(model='multiorgan_scaled')``). This private
module keeps the N-organ decomposed run (one :class:`OrganProcess` per organ, all
coupled through ONE shared ``blood`` store by additive ``Glc_blood`` deltas) and
:func:`multiorgan_scaling`, the monolith-vs-decomposition strain curve.

Numbers from :func:`multiorgan_scaling` are REAL - re-run to refresh; do not hand-edit.
"""
from __future__ import annotations

import copy
import statistics
import time
from typing import List, Sequence

import numpy as np
from process_bigraph import Composite, Process, allocate_core
from process_bigraph.emitter import RAMEmitter

from viva_sbml_decomposition.harness.accuracy import compare_to_reference
from viva_sbml_decomposition.models.multiorgan import (
    ABSOLUTE_TOLERANCE, BLOOD, BLOOD_INITIAL, DEFAULT_N_ORGANS, DEFAULT_N_STEPS,
    RELATIVE_TOLERANCE, T_END, all_species, build_multiorgan_antimony,
    organ_species, single_organ_antimony)

from ._run import extract_run

# Coupling interval at which the shared blood store reconciles every organ's delta
# (additive-split error is first-order in it).
DEFAULT_COUPLING_INTERVAL = 0.05


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
