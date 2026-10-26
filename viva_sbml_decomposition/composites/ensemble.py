"""Generic ensemble composite: N instances of ANY model/engine, same-rate or multi-rate.

``ensemble(model, engine, n_instances, intervals, param_draws)`` replaces arm 5
(Ray-parallel same-rate ensemble) and arm 7 (multi-rate scheduler stress test):

* ``intervals=None``  -> SAME-RATE: ``n_instances`` independent whole-model runs at
  the model's own interval (``t_end / n_steps``), optionally with per-instance
  ``param_draws``. The runner (:func:`run_ensemble`) executes them on Ray OS
  processes (arm 5's ``_try_ray`` path; libroadrunner holds the GIL so threads do
  not help), falling back to serial only if Ray is unavailable.
* ``intervals=[...]`` or ``"geometric"`` -> MULTI-RATE: one node per instance, each
  on its own interval, under process-bigraph #270's event-heap scheduler
  (``Composite({"state": doc, "scheduler": "heap"})`` -- the scheduler is a
  Composite *config field*, as in arm 7; see :func:`run_multirate`).

Each instance is a monolithic run of the whole model (:func:`engine_node`), so
``config.model`` carries the Antimony for every engine. Instances are independent
(no shared store, no coupling), written to their own ``s<i>`` leaf.
"""
from __future__ import annotations

import os
import time
from typing import List, NamedTuple, Optional, Sequence, Union

import numpy as np
from process_bigraph import Composite, allocate_core
from process_bigraph.composite_generator import composite_generator

from viva_sbml_decomposition.harness.timing import time_run

from ._emit import strip_emitter
from ._engine import ENGINES, EngineNode, engine_node, register_engines
from .engine_monolithic import (DEFAULT_N_STEPS, DEFAULT_T_END, T_END_BY_MODEL,
                                resolve_model, run_engine_monolithic)

DEFAULT_N_INSTANCES = 8

ParamDraw = Optional[dict]
Intervals = Union[None, str, Sequence[float]]

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


# ---------------------------------------------------------------- parallel helpers

def _draws(n_instances: int, param_draws: Optional[Sequence[ParamDraw]]) -> List[ParamDraw]:
    if n_instances < 1:
        raise ValueError("n_instances must be >= 1")
    if param_draws is None:
        return [None] * n_instances
    draws = list(param_draws)
    if len(draws) != n_instances:
        raise ValueError(
            f"param_draws has {len(draws)} entries but n_instances={n_instances}"
        )
    return draws


def _try_ray():
    """Return the ``ray`` module initialized with a modest cpu count, or None."""
    try:
        import ray
    except ImportError:
        return None
    try:
        if not ray.is_initialized():
            n_cpus = min(DEFAULT_N_INSTANCES, os.cpu_count() or 1)
            ray.init(
                num_cpus=n_cpus,
                ignore_reinit_error=True,
                include_dashboard=False,
                logging_level="ERROR",
                configure_logging=False,
            )
        return ray
    except (ConnectionError, OSError, RuntimeError):
        # Ray init/connection failure = availability problem -> serial fallback.
        return None


def _dispatch_errors():
    """Ray errors that mean the backend (not the model) failed."""
    from ray.exceptions import RaySystemError

    return (RaySystemError, ConnectionError)


def _resolve_intervals(intervals: Intervals, n: int, horizon: float) -> Optional[List[float]]:
    if intervals is None or (isinstance(intervals, str) and intervals in ("", "same_rate")):
        return None
    if isinstance(intervals, str):
        if intervals != "geometric":
            raise ValueError(f"intervals must be None, 'geometric' or a list, got {intervals!r}")
        return geometric_intervals(n, horizon)
    ivs = [float(x) for x in intervals]
    if len(ivs) != n:
        raise ValueError(f"intervals has {len(ivs)} entries but n_instances={n}")
    return ivs


def _namespaced(en: EngineNode, i: int):
    """Re-wire one engine node to its own ``s<i>`` leaf (independent instance)."""
    node = dict(en.node)
    if en.engine == "tellurium":     # arm 7 layout: species array at stores.s<i>
        node["outputs"] = {"species": ["stores", f"s{i}"]}
        species_path = ["stores", f"s{i}"]
    else:
        pre = [f"s{i}"]
        node["inputs"] = {k: pre + list(v) for k, v in en.node["inputs"].items()}
        node["outputs"] = {k: pre + list(v) for k, v in en.node["outputs"].items()}
        species_path = node["outputs"]["species"]
    return node, species_path


def build_document(model="glycolysis", engine="tellurium", n_instances=DEFAULT_N_INSTANCES,
                   intervals: Intervals = None, param_draws=None,
                   n_steps=DEFAULT_N_STEPS, t_end=None, horizon=HORIZON,
                   emit=False) -> dict:
    """Unstripped ensemble document: ``inst_<i>`` nodes (+ ``emitter_<i>`` if ``emit``).

    ``emit=True`` adds one RAM emitter per instance (species + scheduler time) so
    full trajectories can be read back; the workbench generator leaves it off.
    """
    model_id, antimony = resolve_model(model)
    draws = _draws(n_instances, param_draws)
    if engine not in ENGINES:
        raise ValueError(f"unknown engine {engine!r}; expected one of {ENGINES}")
    if engine != "tellurium" and any(draws):
        raise ValueError("param_draws are supported for engine='tellurium' only")
    ivs = _resolve_intervals(intervals, n_instances, horizon)
    if t_end is None:
        t_end = T_END_BY_MODEL.get(model_id, DEFAULT_T_END)
    doc: dict = {"stores": {}}
    for i in range(n_instances):
        interval = ivs[i] if ivs is not None else t_end / n_steps
        en = engine_node(engine, antimony, interval=interval, parameter_overrides=draws[i])
        node, species_path = _namespaced(en, i)
        doc[f"inst_{i}"] = node
        if engine != "tellurium":
            doc[f"s{i}"] = {k: (dict(v) if isinstance(v, dict) else v)
                            for k, v in en.stores.items()}
        if emit:
            doc[f"emitter_{i}"] = {
                "_type": "step", "address": "local:ram-emitter",
                "config": {"emit": {"species": en.emit["species"], "time": "float"}},
                "inputs": {"species": species_path, "time": ["global_time"]},
            }
    return doc


@composite_generator(
    name="ensemble",
    description=(
        "Generic ensemble: N independent whole-model instances of any registry "
        "model/engine. intervals=None is the same-rate ensemble (Ray-parallel "
        "runner, optional per-instance parameter draws); intervals='geometric' (or "
        "a list) is the multi-rate case, one node per instance on its own interval, "
        "driven by process-bigraph #270's event-heap scheduler. config.model carries "
        "the Antimony."
    ),
    parameters={
        "model": {"type": "string", "default": "glycolysis"},
        "engine": {"type": "string", "default": "tellurium"},
        "n_instances": {"type": "integer", "default": DEFAULT_N_INSTANCES},
        "intervals": {"type": "string", "default": "same_rate",
                      "description": "'same_rate' | 'geometric' (a list is accepted from Python)"},
        "n_steps": {"type": "integer", "default": DEFAULT_N_STEPS},
        "scheduler": {"type": "string", "default": DEFAULT_SCHEDULER},
    },
    default_n_steps=DEFAULT_N_STEPS,
    core_extensions=[register_engines],
    fidelity="real",
    methods=["N independent whole-model engine processes",
             "same-rate: Ray OS-process parallel runner",
             "multi-rate: geometric per-instance intervals, #270 heap scheduler"],
)
def ensemble(core=None, model="glycolysis", engine="tellurium",
             n_instances=DEFAULT_N_INSTANCES, intervals: Intervals = None,
             param_draws=None, n_steps=DEFAULT_N_STEPS, scheduler=DEFAULT_SCHEDULER):
    """Ensemble document (emitter stripped). ``scheduler`` labels the multi-rate
    strategy; it is a Composite config field applied by :func:`run_multirate`, not
    a document field (as in arm 7)."""
    return strip_emitter(build_document(model, engine, n_instances, intervals,
                                        param_draws, n_steps))


# ---------------------------------------------------------------- same-rate runner

def _run_instance(model, engine, n_steps, param_overrides=None) -> dict:
    """One monolithic instance -> final state ``{species: float}`` (Ray-shippable)."""
    traj, _t = run_engine_monolithic(model, engine, n_steps=n_steps,
                                     parameter_overrides=param_overrides)
    return {s: float(v[-1]) for s, v in traj.items()}


def _run_instance_traj(model, engine, n_steps, param_overrides=None):
    return run_engine_monolithic(model, engine, n_steps=n_steps,
                                 parameter_overrides=param_overrides)


class EnsembleResults(list):
    """Per-instance results tagged with the backend that ran (``ray`` | ``serial``)."""
    backend: str = "serial"


def run_ensemble(model="glycolysis", engine="tellurium", n_instances=DEFAULT_N_INSTANCES,
                 param_draws=None, n_steps=DEFAULT_N_STEPS, full=False) -> EnsembleResults:
    """Run the same-rate ensemble on Ray (serial fallback if Ray is unavailable).

    Returns final-state dicts, or ``(traj, t)`` per instance when ``full=True``.
    Model errors propagate; only a Ray backend failure falls back to serial.
    """
    draws = _draws(n_instances, param_draws)
    fn = _run_instance_traj if full else _run_instance
    ray = _try_ray()
    if ray is not None:
        try:
            remote = ray.remote(fn)
            out = EnsembleResults(ray.get([remote.remote(model, engine, n_steps, d)
                                           for d in draws]))
            out.backend = "ray"
            return out
        except _dispatch_errors():
            pass
    out = EnsembleResults(fn(model, engine, n_steps, d) for d in draws)
    out.backend = "serial"
    return out


class EnsembleSpeedup(NamedTuple):
    parallel_s: Optional[float]
    serial_s: Optional[float]
    speedup: Optional[float]
    backend: str
    speedup_claimed: bool
    n_instances: int
    n_cpus: int
    setup_s: Optional[float] = None
    setup_excluded: bool = False


def _serial_only(serial_s, n, setup_s=None):
    return EnsembleSpeedup(None, serial_s, None, "serial", False, n,
                           os.cpu_count() or 1, setup_s, False)


def ensemble_speedup(model="glycolysis", engine="tellurium", n_instances=DEFAULT_N_INSTANCES,
                     *, n_steps=DEFAULT_N_STEPS, param_draws=None, repeats=3) -> EnsembleSpeedup:
    """Serial vs Ray wall-clock (arm 5 semantics: speedup claimed only if Ray truly wins;
    Ray init/warmup reported as ``setup_s`` and excluded from ``parallel_s``)."""
    draws = _draws(n_instances, param_draws)
    serial = time_run(lambda: [_run_instance(model, engine, n_steps, d) for d in draws],
                      repeats=repeats, warmup=True)
    serial_s = serial.median_s
    state = {}

    def _setup():
        ray = _try_ray()
        if ray is None:
            return
        remote = ray.remote(_run_instance)
        ray.get([remote.remote(model, engine, n_steps, d) for d in draws])
        state["ray"], state["remote"] = ray, remote

    def _batch():
        state["ray"].get([state["remote"].remote(model, engine, n_steps, d) for d in draws])

    try:
        t0 = time.perf_counter()
        _setup()
        setup_s = time.perf_counter() - t0
        if "ray" not in state:
            return _serial_only(serial_s, n_instances)
        par = time_run(_batch, repeats=repeats, warmup=False)
    except _dispatch_errors():
        return _serial_only(serial_s, n_instances)
    parallel_s = par.median_s
    speedup = serial_s / parallel_s if parallel_s > 0 else 0.0
    if parallel_s < serial_s and speedup > 1.0:
        return EnsembleSpeedup(parallel_s, serial_s, speedup, "ray", True, n_instances,
                               os.cpu_count() or 1, setup_s, True)
    return _serial_only(serial_s, n_instances, setup_s=setup_s)


# ---------------------------------------------------------------- multi-rate runner

def heap_scheduler_available() -> bool:
    """True if this process-bigraph exposes #270's ``scheduler`` config field."""
    return "scheduler" in (Composite.config_schema if hasattr(Composite, "config_schema") else {})


def run_multirate(model="glycolysis", engine="tellurium", n_instances=DEFAULT_N_INSTANCES,
                  intervals: Intervals = "geometric", param_draws=None, horizon=HORIZON,
                  scheduler=DEFAULT_SCHEDULER, n_steps=DEFAULT_N_STEPS):
    """Run the multi-rate ensemble under ``scheduler``; return ``[(traj, t), ...]`` per instance.

    Species keys are the model's floating-species ids (tellurium array order).
    """
    from ._engine import tellurium_species_order

    _, antimony = resolve_model(model)
    doc = build_document(model, engine, n_instances, intervals, param_draws,
                         n_steps, horizon=horizon, emit=True)
    core = allocate_core()
    register_engines(core)
    comp = Composite({"state": doc, "scheduler": scheduler}, core=core)
    comp.run(horizon)
    order = (list(tellurium_species_order(antimony)) if engine == "tellurium" else None)
    out = []
    for i in range(n_instances):
        rows = comp.state[f"emitter_{i}"]["instance"].history
        t = np.array([r["time"] for r in rows], dtype=float)
        if order is not None:
            traj = {s: np.array([np.asarray(r["species"])[k] for r in rows], dtype=float)
                    for k, s in enumerate(order)}
        else:
            keys = list(rows[0]["species"])
            traj = {s: np.array([r["species"][s] for r in rows], dtype=float) for s in keys}
        out.append((traj, t))
    return out


# ---------------------------------------------------------------- scheduler scaling

def _make_composite(n_instances: int, scheduler: str, horizon: float) -> Composite:
    """The multi-rate composite under ``scheduler`` (``scan`` | ``heap`` | ``auto``).

    #270's API: the strategy is the ``"scheduler"`` config field on the
    ``Composite``, passed next to ``"state"``.
    """
    core = allocate_core()
    register_engines(core)
    doc = build_document("glycolysis", "tellurium", n_instances, "geometric", horizon=horizon)
    return Composite({"state": doc, "scheduler": scheduler}, core=core)


def _reset(comp: Composite) -> None:
    """Rewind a constructed composite to t=0 so ``run`` can be re-timed.

    Resets each instance's roadrunner clock and clears the scheduler state, so
    :func:`time_run` measures ``run()`` alone (construction excluded).
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
    return {k: np.asarray(v, dtype=float) for k, v in comp.state["stores"].items()}


def multirate_scaling(ns: Sequence[int] = (4, 16, 64), *,
                      horizon: float = HORIZON, repeats: int = 3) -> List[dict]:
    """Time the multi-rate ensemble under ``scan`` (O(P^2)) vs ``heap`` (#270).

    For each ``n``, builds the composite once per scheduler and times ``run()``
    alone (median of ``repeats``, one warmup). Returns
    ``[{n, scan_s, heap_s, speedup, bit_identical}]`` with ``speedup = scan_s /
    heap_s``. ``bit_identical`` compares the two schedulers' final per-instance
    state (same math, different tick order), so it must be ``True`` at every ``n``.
    ``speedup`` is whatever was measured: with real CVODE instances integration
    dominates the per-tick scan at small P (crossover ~P=50-64).
    """
    out: List[dict] = []
    for n in ns:
        comp_scan = _make_composite(n, "scan", horizon)
        scan = time_run(lambda c=comp_scan: (_reset(c), c.run(horizon)),
                        repeats=repeats, warmup=True)
        comp_heap = _make_composite(n, "heap", horizon)
        heap = time_run(lambda c=comp_heap: (_reset(c), c.run(horizon)),
                        repeats=repeats, warmup=True)
        scan_s, heap_s = scan.median_s, heap.median_s
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
