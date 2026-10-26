"""Collect the real per-arm metric table that the visualization suite renders.

The comparative figures (runtime, process-vs-framework overhead, accuracy,
convergence, ensemble scaling, composability, the 3-way tradeoff map) are
cross-arm aggregates — they cannot come from a single run's xarray. This module
gathers them into one dict: the fast arms are measured LIVE (so the numbers are
real, not transcribed), while the slow arm-3 convergence and the Ray ensemble
carry the values measured in their own studies (biological-modular-seam / ensemble-and-multirate), with provenance.

`collect_metrics()` returns the dict; `write_metrics(path)` caches it as JSON so
the Visualizations render without re-running the arms. Re-run to refresh.
"""
from __future__ import annotations

import json
import os

import numpy as np

from ..composites.engine_monolithic import (run_engine_monolithic,
                                            time_engine_monolithic)
from ..composites.reaction_split import run_reaction_split, time_reaction_split
from ..harness.accuracy import compare_to_reference
from ..harness.composability import score_arm
from ..models.glycolysis import GLYCOLYSIS_META

SP = GLYCOLYSIS_META["species"]
T_END = GLYCOLYSIS_META["t_end"]

# Metric-column ids (stable internal keys) + READER-facing study labels (the spectrum).
# Column -> study: arm1=sbml-in-tellurium, arm2=mechanical-split-cost (additive/Jacobi
# coupling), arm3=biological-modular-seam, arm4=fully-native-pbg,
# arm5=ensemble-and-multirate, arm6=cross-engine-coupling. The ids never reach a reader.
ARMS = [
    ("arm1", "SBML in\nTellurium"),
    ("arm2", "Mechanical split\n(additive)"),
    ("arm3", "Biological\nmodular seam"),
    ("arm4", "Fully native\nPBG"),
    ("arm5", "Ensemble\n(Ray)"),
    ("arm6", "Cross-engine\ncoupling"),
]


def collect_metrics() -> dict:
    # Every arm column is measured LIVE through the GENERIC composites:
    #   arm1 = engine_monolithic(tellurium)     arm4 = engine_monolithic(native)
    #   arm2 = reaction_split(seam=mechanical)  arm6 = reaction_split(aldolase, tellurium+native)
    # --- arm 1: the reference (bulk-bound tellurium monolith) ---
    ref_traj, ref_t, rt1 = time_engine_monolithic("glycolysis", "tellurium", 300)

    # --- arm 2: mechanical split, fine interval ---
    _, _, rt2 = time_reaction_split("glycolysis", "mechanical", "tellurium", "tellurium", 0.0125)
    a2_traj, a2_t = run_reaction_split("glycolysis", "mechanical", "tellurium", "tellurium",
                                       coupling_interval=0.0125)
    acc2 = compare_to_reference(ref_traj, ref_t, a2_traj, a2_t)
    a2c_traj, a2c_t = run_reaction_split("glycolysis", "mechanical", "tellurium", "tellurium",
                                         coupling_interval=0.1)
    acc2_coarse = compare_to_reference(ref_traj, ref_t, a2c_traj, a2c_t)

    # --- arm 4: native, both leaf variants ---
    _, _, rt4s = time_engine_monolithic("glycolysis", "native", 200, native_leaf="scalar")
    _, _, rt4a = time_engine_monolithic("glycolysis", "native", 200, native_leaf="array")
    # Accuracy must use the SAME output grid as the reference (n_steps=300) - a
    # coarser grid would smear the sharp early transient under interpolation.
    a4_traj, a4_t = run_engine_monolithic("glycolysis", "native", n_steps=300)
    acc4 = compare_to_reference(ref_traj, ref_t, a4_traj, a4_t)

    # --- arm 6: cross-engine hybrid (tellurium upper + native lower); accuracy-exempt ---
    a6_traj, a6_t, rt6 = time_reaction_split("glycolysis", "aldolase", "tellurium", "native",
                                             0.005)
    arm6_finite = bool(all(np.all(np.isfinite(v)) for v in a6_traj.values()))

    def ov(rt):
        return {"total_s": rt.total_s, "process_s": rt.process_s,
                "framework_s": rt.framework_s, "overhead_fraction": rt.overhead_fraction}

    overhead = {
        "arm1": ov(rt1),
        "arm2": ov(rt2),
        "arm3": {"overhead_fraction": 0.05, "total_s": None, "process_s": None,
                 "framework_s": None, "_provenance": "arm3 study (coarse interval), slow to re-run live"},
        "arm4_scalar": ov(rt4s),
        "arm4_array": ov(rt4a),
        "arm6": ov(rt6),
    }
    # arm4 headline = scalar variant
    overhead["arm4"] = overhead["arm4_scalar"]

    accuracy = {
        "arm1": 0.0,
        "arm2": float(acc2.max_rel_err),
        "arm3": 0.100,   # measured in arm3 study at the fine coupling step (T=0.15)
        "arm4": float(acc4.max_rel_err),
        "arm5": None,    # ensemble — not a single-trajectory accuracy
        "arm6": None,    # accuracy-exempt (no monolithic reference)
    }

    convergence = {
        "arm2": [[0.1, float(acc2_coarse.max_rel_err)], [0.0125, float(acc2.max_rel_err)]],
        "arm3": [[0.002, 0.468], [0.0005, 0.100]],
    }

    ensemble = {"n_instances": 8, "n_cpus": 16, "serial_s": 1.189,
                "parallel_s": 0.158, "speedup": 7.52, "backend": "ray",
                "gil_note": "libroadrunner does not release the GIL; ThreadPool(8)=0.99x",
                "_provenance": "arm5 study"}

    composability = {}
    for arm in ["arm1", "arm2", "arm3", "arm4", "arm5", "arm6"]:
        try:
            s = score_arm(arm)
            composability[arm] = {
                "substitutability": int(getattr(s.substitutability, "value", s.substitutability)),
                "coupling_readiness": int(getattr(s.coupling_readiness, "value", s.coupling_readiness)),
                "multiscale_fit": int(getattr(s.multiscale_fit, "value", s.multiscale_fit)),
                "accuracy_exempt": bool(getattr(s, "accuracy_exempt", arm == "arm6")),
            }
        except Exception as e:  # pragma: no cover - score_arm shape guard
            composability[arm] = {"_error": str(e)}

    return {
        "arms": [a for a, _ in ARMS],
        "labels": {a: lbl for a, lbl in ARMS},
        "reference_trajectory": {"time": [float(x) for x in ref_t],
                                 "species": {s: [float(x) for x in ref_traj[s]] for s in SP}},
        "overhead": overhead,
        "accuracy": accuracy,
        "convergence": convergence,
        "ensemble": ensemble,
        "composability": composability,
        "notes": {
            "overhead_fraction": "framework SHARE of wall-clock (Composite.timing_summary); highest for the CHEAP monolith, NOT the decomposition cost.",
            "scalar_vs_array": "indistinguishable at 17 species; scalar-dict tax is linear in element count, bites only at 100s-1000s.",
            "mismatched_interval": "O(P^2) multi-rate scheduler cost in pb<=1.10.x; PR #270 (unmerged 2026-10-25) -> O(P log P). Prefer equal coupling intervals.",
        },
    }


def write_metrics(path: str) -> dict:
    data = collect_metrics()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)
    return data


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "metrics.json"
    write_metrics(out)
    print("wrote", out)
