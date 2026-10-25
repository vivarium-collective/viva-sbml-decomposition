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
from viva_sbml_decomposition.harness.timing import time_run
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


class EnsembleResults(list):
    """List of per-instance final-state dicts, tagged with the backend that ran.

    ``backend`` is ``"ray"`` (OS-process parallel) or ``"serial"`` (fallback).
    """

    backend: str = "serial"


def glycolysis_ensemble(
    core=None,
    n_instances: int = DEFAULT_N_INSTANCES,
    param_draws: Optional[Sequence[ParamDraw]] = None,
    n_steps: int = DEFAULT_N_STEPS,
) -> EnsembleResults:
    """Run ``n_instances`` monolithic glycolysis instances concurrently via Ray.

    Each instance is one arm-1 run, optionally perturbed by the matching entry
    of ``param_draws`` (a sequence of ``{param_id: value}`` or ``None``). Returns
    an :class:`EnsembleResults` list of ``n_instances`` final-state dicts (17
    species each) whose ``.backend`` records what actually ran. Falls back to a
    serial loop only when Ray is unavailable or its dispatch fails; a genuine
    model/integrator error in an instance PROPAGATES (never masked).

    ``core`` is accepted for interface symmetry but ignored: a core is not
    picklable/shareable, so every instance (and Ray worker) builds its own.
    """
    draws = _draws(n_instances, param_draws)
    ray = _try_ray()
    if ray is not None:
        try:
            remote = ray.remote(_run_instance)
            out = EnsembleResults(ray.get([remote.remote(d, n_steps) for d in draws]))
            out.backend = "ray"
            return out
        except _dispatch_errors():
            pass  # Ray backend failure -> serial; task (model) errors propagate
    out = EnsembleResults(_run_instance(d, n_steps) for d in draws)
    out.backend = "serial"
    return out


class EnsembleSpeedup(NamedTuple):
    """Timing outcome for an ensemble run.

    ``parallel_s``/``serial_s``/``speedup`` are the documented first three fields
    (medians from the shared ``time_run`` harness). ``backend`` and
    ``speedup_claimed`` carry the honesty verdict: a speedup is claimed only when
    a real Ray batch beat the serial baseline on the clock.

    ``setup_s`` is the Ray init + worker warmup time (typically ~15-20 s). It is
    EXCLUDED from ``parallel_s`` (``setup_excluded=True``): ``speedup`` is a
    steady-state throughput figure and must not be read as including setup.
    """

    parallel_s: Optional[float]
    serial_s: Optional[float]
    speedup: Optional[float]
    backend: str  # "ray" | "serial"
    speedup_claimed: bool
    n_instances: int
    n_cpus: int
    setup_s: Optional[float] = None
    setup_excluded: bool = False


def _serial_only(serial_s: float, n_instances: int, setup_s: Optional[float] = None) -> EnsembleSpeedup:
    return EnsembleSpeedup(
        parallel_s=None,
        serial_s=serial_s,
        speedup=None,
        backend="serial",
        speedup_claimed=False,
        n_instances=n_instances,
        n_cpus=os.cpu_count() or 1,
        setup_s=setup_s,
        setup_excluded=False,
    )


def ensemble_speedup(
    n_instances: int = DEFAULT_N_INSTANCES,
    *,
    n_steps: int = DEFAULT_N_STEPS,
    param_draws: Optional[Sequence[ParamDraw]] = None,
    repeats: int = 3,
) -> EnsembleSpeedup:
    """Measure serial vs Ray-parallel wall-clock for ``n_instances`` instances.

    Both sides use the shared :func:`time_run` harness (median of ``repeats``,
    one warmup). Ray init + worker warmup runs as the harness ``setup`` and is
    reported as ``setup_s``, EXCLUDED from ``parallel_s``. The speedup is claimed
    (``backend=="ray"``, ``speedup_claimed=True``, ``speedup>1``) ONLY when the
    Ray batch genuinely finished faster than the serial loop. Otherwise the
    result declares a serial fallback and reports no speedup. Model errors
    propagate.
    """
    draws = _draws(n_instances, param_draws)

    serial = time_run(
        lambda: [_run_instance(d, n_steps) for d in draws], repeats=repeats, warmup=True
    )
    serial_s = serial.median_s

    state = {}

    def _ray_setup():
        ray = _try_ray()
        if ray is None:
            return
        remote = ray.remote(_run_instance)
        ray.get([remote.remote(d, n_steps) for d in draws])  # warm workers
        state["ray"], state["remote"] = ray, remote

    def _ray_batch():
        state["ray"].get([state["remote"].remote(d, n_steps) for d in draws])

    try:
        t0 = time.perf_counter()
        _ray_setup()
        setup_s = time.perf_counter() - t0
        if "ray" not in state:
            return _serial_only(serial_s, n_instances)
        par = time_run(_ray_batch, repeats=repeats, warmup=False)
    except _dispatch_errors():
        return _serial_only(serial_s, n_instances)
    parallel_s = par.median_s

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
            setup_s=setup_s,
            setup_excluded=True,
        )
    # Ray ran but did not beat serial: honest serial declaration, no speedup.
    return _serial_only(serial_s, n_instances, setup_s=setup_s)
