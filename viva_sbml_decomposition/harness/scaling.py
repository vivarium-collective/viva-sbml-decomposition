"""Scaling-benchmarks harness: how does decomposition scale, and what is the
Phase-6 performance baseline?

Two axes, both reusing existing measured machinery:

- **Model size** — `multiorgan_scaling(ns)` gives the monolithic vs decomposed
  build/run/total wall-clock and accuracy vs species count. The headline is the
  monolithic BUILD strain; we fit its exponent (slope of log build-time vs log
  species) so "superlinear" is a number, not a vibe.
- **#instances** — `ensemble_speedup(n_instances)` times the same-rate ensemble
  serial vs parallel (Ray, serial fallback) and reports the backend used.

Wall-clock is NOT reproducible across machines/BLAS/load, so callers must grade
only robust properties (exponent > 1; speedup > 1 when a parallel backend ran).
The returned numbers are this machine's; `machine` tags them.
"""
from __future__ import annotations

import platform

import numpy as np

from viva_sbml_decomposition.composites._multiorgan import multiorgan_scaling
from viva_sbml_decomposition.composites.ensemble import ensemble_speedup


def scaling_benchmarks(model_sizes=(1, 2, 4), instance_counts=(1, 2, 4), *,
                       repeats: int = 3, seed: int = 0) -> dict:
    """Model-size + ensemble scaling sweep. See module docstring.

    `seed` is accepted for a stable call signature; the timings are deterministic
    in structure, not magnitude. Never raises on Ray-unavailable (serial fallback).
    """
    curve = multiorgan_scaling(tuple(model_sizes), repeats=repeats)
    species = np.array([p["species"] for p in curve], dtype=float)
    build = np.array([max(p["mono_build_s"], 1e-9) for p in curve], dtype=float)
    build_exponent = (
        float(np.polyfit(np.log(species), np.log(build), 1)[0]) if len(curve) >= 2 else float("nan")
    )

    instances, speedups = [], []
    backend = "serial"
    for p in instance_counts:
        res = ensemble_speedup(n_instances=int(p), repeats=repeats)
        instances.append(int(p))
        speedups.append(float(res.speedup) if res.speedup is not None else 1.0)
        backend = res.backend

    return {
        "model_size": [
            {"n": p["n"], "species": p["species"], "mono_total_s": p["mono_total_s"],
             "decomp_total_s": p["decomp_total_s"], "accuracy": p["accuracy"]}
            for p in curve
        ],
        "build_exponent": build_exponent,
        "ensemble": {"instances": instances, "speedup": speedups, "backend": backend},
        "machine": platform.platform(),
    }
