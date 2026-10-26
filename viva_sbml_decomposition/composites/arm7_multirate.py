"""Arm 7: the MULTI-RATE arm — a scheduler stress test, not a biology result.

Arms 1-6 all use **equal** coupling intervals (the investigation's decision table
says to prefer them, because a mismatched-interval decomposition carried an
``O(P^2)`` per-tick scheduler cost in process-bigraph ``<=1.10.x``). Arm 7 is the
one arm that deliberately does the thing the investigation otherwise avoids: it
builds ``N`` **independent** monolithic glycolysis instances (each the arm-1
whole-model ``TelluriumProcess`` with a bulk-bound species port), each advanced on
a **different, geometrically-spaced** coupling interval — a genuinely multi-rate
composite.

The instances are INDEPENDENT: no shared store, no coupling, empty ``inputs``.
Each is just the arm-1 monolith on its own clock, writing to its own ``stores.s<i>``
leaf. So there is **no operator-splitting divergence** — the trajectory each
instance computes is identical whichever scheduler drives the loop. The point is
to exercise and MEASURE the temporal scheduler, not the biology.

**Why it matters.** With many distinct intervals the stock ("scan") tick loop
calls ``run_process`` on every process every tick (``O(P)`` per tick) and the tick
count itself grows with ``P`` as finer intervals are added — ``O(P^2)`` total.
process-bigraph **#270** adds an event-min-heap scheduler (``scheduler="heap"``):
each tick pops only the due processes in ``O(log P)`` and drops a process whose
next step overruns the run horizon, giving ``~O(P log P)``. Because the two
schedulers run the *same* math in a *different* order, their results are
bit-identical; only the wall-clock differs.

**Honesty.** The separation at the ``N`` a unit test can afford (``<=64`` real
libroadrunner/CVODE instances) is modest — the per-process CVODE update dominates
the per-tick Python scan at small ``P`` (the measured crossover where heap
overtakes scan is ``~P=50-64``; see :func:`multirate_scaling` and the study
findings for the measured curve). The speedup grows monotonically with ``P`` and is reported as measured —
never fabricated. #270's own micro-benchmark on cheap processes (where the
scheduler, not the integrator, is the whole cost) is ``4.8x`` at ``P=100`` rising
to ``46x`` at ``P=1000``; arm 7 reproduces the *trend* with real integrator
instances, not those multipliers.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Sequence

from process_bigraph import Composite, allocate_core
from process_bigraph.composite_generator import composite_generator
from viva_tellurium import TelluriumProcess
from viva_tellurium.composites import register_tellurium

from viva_sbml_decomposition.harness.timing import time_run
from viva_sbml_decomposition.models.glycolysis import GLYCOLYSIS_ANTIMONY, GLYCOLYSIS_META

from ._emit import strip_emitter

# Default multi-rate composite size for the workbench generator.
DEFAULT_N_INSTANCES = 16

# Geometric-interval regime (shared by the generator, the measurement helper, and
# the figure so they tell one consistent story). ``i_fast`` scales FINER as the
# instance count grows (so the tick count grows with P — the O(P^2) driver the
# heap scheduler cancels); ``i_slow`` overruns the run horizon so the slow tail is
# exactly the "next step past the horizon" case #270 drops from the heap.
HORIZON = 20.0           # run horizon for the scaling measurement (< t_end=30)
_I_FAST_FACTOR = 3.0     # i_fast = HORIZON / (_I_FAST_FACTOR * n)
_I_SLOW_FACTOR = 3.0     # i_slow = HORIZON * _I_SLOW_FACTOR (> HORIZON: dead tail)
DEFAULT_SCHEDULER = "heap"


def geometric_intervals(n: int, horizon: float = HORIZON) -> List[float]:
    """``n`` geometrically-spaced coupling intervals, fastest -> slowest.

    ``i_fast = horizon / (3 n)`` (finer as ``n`` grows, so the tick count grows
    with ``P``); ``i_slow = 3 horizon`` (overruns the horizon). All distinct, so
    the composite is genuinely multi-rate.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    i_fast = horizon / (_I_FAST_FACTOR * n)
    i_slow = horizon * _I_SLOW_FACTOR
    if n == 1:
        return [i_fast]
    r = (i_slow / i_fast) ** (1.0 / (n - 1))
    return [i_fast * r**k for k in range(n)]


@lru_cache(maxsize=1)
def _species_order() -> tuple[str, ...]:
    """Floating-species id order of each instance's bulk ``species`` array."""
    core = allocate_core()
    register_tellurium(core)
    proc = TelluriumProcess(
        config={"model": GLYCOLYSIS_ANTIMONY, "model_format": "antimony",
                "bulk_species": True},
        core=core)
    return tuple(proc.get_species_ids())


def _instance_node(interval: float) -> dict:
    """One arm-1 monolithic ``TelluriumProcess`` node (bulk species, no inputs)."""
    return {
        "_type": "process",
        "address": "local:TelluriumProcess",
        "config": {
            "model": GLYCOLYSIS_ANTIMONY,
            "model_format": "antimony",
            "integrator": "cvode",
            "absolute_tolerance": 1e-10,
            "relative_tolerance": 1e-8,
            "seed": -1,
            "species_overrides": {},
            "parameter_overrides": {},
            "bulk_species": True,
        },
        "interval": interval,
        # Independent: no upstream writer, so empty inputs (as arm 1). The
        # instance only advances its own copy and emits into its own store.
        "inputs": {},
        "outputs": {"species": ["stores", "species"]},  # per-instance path set below
    }


def build_document(n_instances: int, horizon: float = HORIZON) -> dict:
    """Build the multi-rate document: ``n_instances`` independent monolith nodes.

    Each node ``inst_<i>`` is an arm-1 ``TelluriumProcess`` on a distinct
    geometric interval, wired to its OWN ``stores.s<i>`` leaf (no shared store, no
    coupling). No emitter: the scaling helper reads final state straight from the
    stores and the generator is stripped for the workbench anyway.
    """
    intervals = geometric_intervals(n_instances, horizon)
    doc: dict = {"stores": {}}
    for i, interval in enumerate(intervals):
        node = _instance_node(interval)
        node["outputs"] = {"species": ["stores", f"s{i}"]}
        doc[f"inst_{i}"] = node
    return doc


@composite_generator(
    name="glycolysis_multirate",
    description=(
        "Arm 7 (multi-rate scheduler stress test): N INDEPENDENT monolithic "
        "glycolysis instances (each the arm-1 whole-model TelluriumProcess, "
        "bulk-bound species), each advanced on a DIFFERENT geometrically-spaced "
        "coupling interval. No shared store and no coupling, so there is no "
        "operator-splitting error — the point is to exercise process-bigraph's "
        "temporal scheduler (#270's event-heap: O(P log P) vs the stock scan's "
        "O(P^2) on many-rate composites), whose scan/heap results are bit-identical."
    ),
    parameters={
        "n_instances": {"type": "integer", "default": DEFAULT_N_INSTANCES},
        "scheduler": {"type": "string", "default": DEFAULT_SCHEDULER},
    },
    core_extensions=[register_tellurium],
    fidelity="real",
    methods=[
        "N independent libroadrunner/CVODE monolith instances (arm 1 per node)",
        "geometrically-spaced per-instance coupling intervals (genuinely multi-rate)",
        "process-bigraph #270 event-heap temporal scheduler (O(P log P))",
    ],
)
def glycolysis_multirate(core=None, n_instances: int = DEFAULT_N_INSTANCES,
                         scheduler: str = DEFAULT_SCHEDULER):
    """N independent multi-rate glycolysis monoliths (emitter-free; stripped).

    ``scheduler`` is a NO-OP here, carried only for discoverability (it labels the
    #270 strategy this arm is about). The generator returns a plain document; the
    scheduler is NOT selectable at this layer — the scan-vs-heap comparison is
    driven by :func:`multirate_scaling`, which sets the ``"scheduler"`` **config
    field** when it constructs the ``Composite`` (#270's API).
    """
    return strip_emitter(build_document(n_instances))


def _make_composite(n_instances: int, scheduler: str, horizon: float) -> Composite:
    """Construct the multi-rate ``Composite`` under the given scheduler.

    #270's API: the scheduler strategy is the ``"scheduler"`` **config field** on
    the ``Composite`` (``"scan"`` | ``"heap"`` | ``"auto"``), passed in the same
    config dict as ``"state"``.
    """
    core = allocate_core()
    register_tellurium(core)
    return Composite({"state": build_document(n_instances, horizon),
                      "scheduler": scheduler}, core=core)


def _reset(comp: Composite) -> None:
    """Rewind a constructed composite to t=0 so ``run`` can be re-timed.

    Resets each instance's roadrunner clock and clears the scheduler state (front,
    global_time, event heap). Lets :func:`time_run` measure ``run()`` alone — the
    one-off ``Composite`` construction (N roadrunner compilations) is excluded, so
    the scan-vs-heap difference is not diluted by construction cost shared by both.
    """
    for path in comp.process_paths:
        node = comp.state
        for key in path:
            node = node[key]
        rr = getattr(node.get("instance"), "_rr", None)
        if rr is not None:
            rr.reset()
    comp.front = {}
    comp.state["global_time"] = 0.0
    comp.global_time = 0.0
    comp._event_heap = None
    comp._run_ticks = 0


def _final_state(comp: Composite) -> dict:
    """Final per-instance species arrays, keyed by store id ``s<i>``."""
    import numpy as np

    stores = comp.state["stores"]
    return {k: np.asarray(v, dtype=float) for k, v in stores.items()}


def multirate_scaling(ns: Sequence[int] = (4, 16, 64), *,
                      horizon: float = HORIZON, repeats: int = 3) -> List[dict]:
    """Time the multi-rate composite under ``scan`` (O(P^2)) vs ``heap`` (#270).

    For each ``n`` in ``ns``, constructs the composite once per scheduler and uses
    the Task-4 :func:`time_run` harness (median of ``repeats``, one warmup) to time
    ``run()`` ALONE (construction excluded via :func:`_reset`). Returns
    ``[{n, scan_s, heap_s, speedup, bit_identical}]`` with
    ``speedup = scan_s / heap_s``.

    ``bit_identical`` is computed per ``n`` by comparing the two schedulers' final
    per-instance state (``_final_state``, ``np.array_equal`` across every instance
    store): the schedulers run the SAME math in a different tick order, so this must
    be ``True`` at every ``n`` — it backs the scan==heap claim across the measured
    range, not just the small-N unit test.

    Honest: ``speedup`` is whatever was measured. With real CVODE instances the
    per-process integration dominates the per-tick scan at small ``P`` (measured
    crossover ``~P=50-64``), so ``speedup`` is modest (even slightly below 1 at
    tiny ``n``, where the heap bookkeeping has nothing to gain) and climbs
    monotonically with ``n``; the regime where it grows large (``P`` in the
    hundreds-to-thousands, cheap processes) is documented, not faked.
    """
    import numpy as np

    out: List[dict] = []
    for n in ns:
        comp_scan = _make_composite(n, "scan", horizon)
        scan = time_run(lambda c=comp_scan: (_reset(c), c.run(horizon)),
                        repeats=repeats, warmup=True)
        comp_heap = _make_composite(n, "heap", horizon)
        heap = time_run(lambda c=comp_heap: (_reset(c), c.run(horizon)),
                        repeats=repeats, warmup=True)
        scan_s, heap_s = scan.median_s, heap.median_s
        # The last timed run leaves each composite's final state in place; the
        # schedulers must agree to the bit (same math, different tick order).
        fs_scan, fs_heap = _final_state(comp_scan), _final_state(comp_heap)
        bit_identical = (set(fs_scan) == set(fs_heap)
                         and all(np.array_equal(fs_scan[k], fs_heap[k]) for k in fs_scan))
        out.append({
            "n": int(n),
            "scan_s": float(scan_s),
            "heap_s": float(heap_s),
            "speedup": float(scan_s / heap_s) if heap_s > 0 else float("nan"),
            "bit_identical": bool(bit_identical),
        })
    return out
