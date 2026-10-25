"""Arm 5: an ensemble of monolithic glycolysis instances on a REAL parallel backend.

This is the one arm where vivarium can win on the *speed* axis: N independent
copies of the whole 17-species model (arm 1), optionally with perturbed rate
constants, run concurrently so throughput actually scales with cores.

Parallelism uses **Ray's OS processes** (``ray.remote`` tasks), NOT
process-bigraph's in-process ``parallel_processes=True`` ThreadPool. libroadrunner
does NOT release the GIL during ``.simulate()`` (measured: a ThreadPool of N
replicates is ~1.0x serial — no win), so in-process threads cannot parallelize
these pure-Python/CVODE-wrapping runs. Separate processes are the only real win.

Honesty: ``ensemble_speedup`` only claims a speedup when a Ray batch actually
beat the serial baseline on the wall clock. If Ray is unavailable/unusable, or
ran without beating serial, it declares ``backend == "serial"`` and
``speedup_claimed is False`` and reports no speedup — never a faked number.
"""
from __future__ import annotations

import os
import time
from typing import List, NamedTuple, Optional, Sequence

from viva_sbml_decomposition.composites.arm1_monolithic import (
    DEFAULT_N_STEPS,
    run_arm as run_monolithic,
)
from viva_sbml_decomposition.models.glycolysis import GLYCOLYSIS_META

DEFAULT_N_INSTANCES = 8

ParamDraw = Optional[dict]


def _run_instance(param_overrides: ParamDraw = None, n_steps: int = DEFAULT_N_STEPS) -> dict:
    """Run one monolithic instance; return its final-state ``{species: float}``.

    Module-level + picklable so Ray can ship it to a worker process. Returning
    the compact final state (not full trajectories) keeps the Ray object small.
    """
    traj, _t = run_monolithic(n_steps=n_steps, parameter_overrides=param_overrides)
    return {s: float(traj[s][-1]) for s in GLYCOLYSIS_META["species"]}


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
    except Exception:
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
    except Exception:
        return None


def glycolysis_ensemble(
    core=None,
    n_instances: int = DEFAULT_N_INSTANCES,
    param_draws: Optional[Sequence[ParamDraw]] = None,
    n_steps: int = DEFAULT_N_STEPS,
) -> List[dict]:
    """Run ``n_instances`` monolithic glycolysis instances concurrently via Ray.

    Each instance is one arm-1 run, optionally perturbed by the matching entry
    of ``param_draws`` (a sequence of ``{param_id: value}`` or ``None``). Returns
    a list of ``n_instances`` final-state dicts (17 species each). Falls back to a
    serial loop (same results) when Ray is unavailable — correctness never
    depends on the backend.
    """
    draws = _draws(n_instances, param_draws)
    ray = _try_ray()
    if ray is not None:
        try:
            remote = ray.remote(_run_instance)
            futures = [remote.remote(d, n_steps) for d in draws]
            return ray.get(futures)
        except Exception:
            pass  # fall through to serial — results are identical
    return [_run_instance(d, n_steps) for d in draws]


class EnsembleSpeedup(NamedTuple):
    """Timing outcome for an ensemble run.

    ``parallel_s``/``serial_s``/``speedup`` are the documented first three fields.
    ``backend`` and ``speedup_claimed`` carry the honesty verdict: a speedup is
    claimed only when a real Ray batch beat the serial baseline on the clock.
    """

    parallel_s: Optional[float]
    serial_s: Optional[float]
    speedup: Optional[float]
    backend: str  # "ray" | "serial"
    speedup_claimed: bool
    n_instances: int
    n_cpus: int


def _serial_only(serial_s: float, n_instances: int) -> EnsembleSpeedup:
    return EnsembleSpeedup(
        parallel_s=None,
        serial_s=serial_s,
        speedup=None,
        backend="serial",
        speedup_claimed=False,
        n_instances=n_instances,
        n_cpus=os.cpu_count() or 1,
    )


def ensemble_speedup(
    n_instances: int = DEFAULT_N_INSTANCES,
    *,
    n_steps: int = DEFAULT_N_STEPS,
    param_draws: Optional[Sequence[ParamDraw]] = None,
) -> EnsembleSpeedup:
    """Measure serial vs Ray-parallel wall-clock for ``n_instances`` instances.

    Returns an :class:`EnsembleSpeedup`. The speedup is claimed (``backend=="ray"``,
    ``speedup_claimed=True``, ``speedup>1``) ONLY when the warmed Ray batch
    genuinely finished faster than the serial loop. Otherwise the result declares
    a serial fallback and reports no speedup.
    """
    draws = _draws(n_instances, param_draws)

    # Serial baseline (in-process warm): one untimed warmup, then timed loop.
    _run_instance(draws[0], n_steps)
    t0 = time.perf_counter()
    for d in draws:
        _run_instance(d, n_steps)
    serial_s = time.perf_counter() - t0

    ray = _try_ray()
    if ray is None:
        return _serial_only(serial_s, n_instances)

    try:
        remote = ray.remote(_run_instance)
        # Warm the worker processes (pay the per-worker import cost) UNTIMED,
        # so the timed batch measures steady-state parallel throughput.
        ray.get([remote.remote(d, n_steps) for d in draws])
        t0 = time.perf_counter()
        ray.get([remote.remote(d, n_steps) for d in draws])
        parallel_s = time.perf_counter() - t0
    except Exception:
        return _serial_only(serial_s, n_instances)

    speedup = serial_s / parallel_s if parallel_s > 0 else 0.0
    if parallel_s < serial_s and speedup > 1.0:
        return EnsembleSpeedup(
            parallel_s=parallel_s,
            serial_s=serial_s,
            speedup=speedup,
            backend="ray",
            speedup_claimed=True,
            n_instances=n_instances,
            n_cpus=os.cpu_count() or 1,
        )
    # Ray ran but did not beat serial (e.g. single-core / contended host).
    # Honest: declare serial fallback, report no speedup.
    return _serial_only(serial_s, n_instances)
