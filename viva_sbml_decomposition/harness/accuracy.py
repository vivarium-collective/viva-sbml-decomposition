"""Accuracy harness: divergence of an arm's trajectory from the reference.

Pure numpy. Trajectories are ``{species_name: np.ndarray}`` plus a time array.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass
class AccuracyResult:
    per_species_rmse: dict = field(default_factory=dict)
    max_rel_err: float = 0.0
    diverged: bool = False
    status: Literal["ok", "diverged"] = "ok"


def compare_to_reference(ref_traj, ref_t, arm_traj, arm_t) -> AccuracyResult:
    """Resample the arm onto the reference grid and compare, pairing by name.

    Raises ValueError on species-set mismatch or disjoint time spans.
    Non-finite arm values give diverged=True and max_rel_err=inf.
    """
    if set(ref_traj) != set(arm_traj):
        missing = sorted(set(ref_traj) ^ set(arm_traj))
        raise ValueError(f"species sets differ: {missing}")
    ref_t = np.asarray(ref_t, dtype=float)
    arm_t = np.asarray(arm_t, dtype=float)
    if ref_t.size == 0 or arm_t.size == 0:
        raise ValueError("empty time array")
    if arm_t.max() < ref_t.min() or arm_t.min() > ref_t.max():
        raise ValueError("reference and arm time spans do not overlap")

    nonfinite = any(not np.all(np.isfinite(np.asarray(v, dtype=float)))
                    for v in arm_traj.values())
    if nonfinite:
        return AccuracyResult(
            per_species_rmse={k: float("inf") for k in ref_traj},
            max_rel_err=float("inf"), diverged=True, status="diverged")

    rmse, max_rel = {}, 0.0
    for name, ref in ref_traj.items():
        ref = np.asarray(ref, dtype=float)
        arm = np.interp(ref_t, arm_t, np.asarray(arm_traj[name], dtype=float))
        err = arm - ref
        rmse[name] = float(np.sqrt(np.mean(err ** 2)))
        scale = max(float(np.max(np.abs(ref))), 1e-12)
        max_rel = max(max_rel, float(np.max(np.abs(err))) / scale)
    return AccuracyResult(per_species_rmse=rmse, max_rel_err=max_rel)
