#!/usr/bin/env python
"""Run + grade the bayesian-calibration study (code-authoritative).

Creates one canonical run, grades its behavior test through this workspace's
`calibration_coverage` evaluator (which runs the grid posterior), and commits the
machine-tagged posterior baseline the figure reads. Grid posterior on the
deterministic ODE is reproducible given the seed; the graded properties are
coverage + informativeness.

Usage:  python scripts/grade_calibration.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import ruamel.yaml
from ruamel.yaml.comments import CommentedMap

try:  # engine ≥ nested study.* layout
    from discovery_engine.study.evaluator import clear_workspace_evaluator_cache, compute_outcomes
    from discovery_engine.study.verdict import roll_up_verdict, write_gate_evaluator
except ImportError:  # older engine: flat module layout
    from discovery_engine.study_evaluator import clear_workspace_evaluator_cache, compute_outcomes
    from discovery_engine.study_verdict import roll_up_verdict, write_gate_evaluator

SLUG = "bayesian-calibration"
INV = "discovery-sbml-recovery"


def _ws_root() -> Path:
    node = Path(__file__).resolve().parent
    for _ in range(8):
        if (node / "workspace.yaml").is_file():
            return node
        node = node.parent
    return Path.cwd()


def grade() -> dict:
    ws = _ws_root()
    sd = ws / "workspace" / "investigations" / INV / "studies" / SLUG
    if not (sd / "study.yaml").is_file():
        raise SystemExit(f"no study.yaml at {sd}")
    y = ruamel.yaml.YAML()
    y.preserve_quotes = True
    p = sd / "study.yaml"
    d = y.load(p.read_text())

    run = CommentedMap()
    run["name"] = f"calibration:{int(time.time())}"
    run["status"] = "completed"
    run["canonical"] = True
    runs = d.setdefault("runs", [])
    runs[:] = [r for r in runs if not str(r.get("name", "")).startswith("calibration:")]
    for r in runs:
        if r.get("canonical"):
            r["canonical"] = False
    runs.insert(0, run)
    d["status"] = "ran"
    with p.open("w") as f:
        y.dump(d, f)

    clear_workspace_evaluator_cache()
    compute_outcomes(sd, ws_root=ws)
    write_gate_evaluator(sd)
    verdict = roll_up_verdict(ruamel.yaml.YAML().load(p.read_text()))
    print(f"{SLUG}: gate -> {verdict.get('result')}")

    _write_baseline(ws)
    return verdict


def _write_baseline(ws: Path) -> None:
    """Commit the machine-tagged grid posterior the figure reads."""
    from viva_sbml_decomposition.harness.calibration import grid_calibration

    r = grid_calibration(grid_points=11, span=0.5, sigma=0.05, n_points=40, seed=0)
    out = ws / "viva_sbml_decomposition" / "visualizations" / "calibration_posterior.json"
    out.write_text(json.dumps(r, indent=2) + "\n")
    cover = all(r["ci95"][p][0] <= r["true"][p] <= r["ci95"][p][1] for p in r["params"])
    print(f"{SLUG}: wrote baseline -> {out.name} (covers_truth={cover})")


if __name__ == "__main__":
    grade()
