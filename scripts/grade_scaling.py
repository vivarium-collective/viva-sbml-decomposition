#!/usr/bin/env python
"""Run + grade the scaling-benchmarks study (code-authoritative).

Creates one canonical run, lets the discovery-engine grade its two behavior tests
through this workspace's `scaling_ensemble_speedup` / `scaling_model_size_cost`
evaluators (which run the scaling sweep, cached), and commits the machine-tagged
baseline JSON the figure and Phase 6 read. Wall-clock is machine-specific; the
graded properties are robust (speedup > 1; decomposition slower end-to-end).

Usage:  python scripts/grade_scaling.py
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

SLUG = "scaling-benchmarks"
INV = "sbml-decomposition"


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
    run["name"] = f"scaling:{int(time.time())}"
    run["status"] = "completed"
    run["canonical"] = True
    runs = d.setdefault("runs", [])
    runs[:] = [r for r in runs if not str(r.get("name", "")).startswith("scaling:")]
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
    """Commit the machine-tagged scaling baseline the figure + Phase 6 read."""
    from viva_sbml_decomposition.harness.scaling import scaling_benchmarks

    r = scaling_benchmarks(model_sizes=(2, 8, 32), instance_counts=(1, 4, 8), repeats=2)
    out = ws / "viva_sbml_decomposition" / "visualizations" / "scaling_benchmarks.json"
    out.write_text(json.dumps(r, indent=2) + "\n")
    print(f"{SLUG}: wrote baseline -> {out.name} "
          f"(exponent {r['build_exponent']:.2f}, ensemble {r['ensemble']['backend']} "
          f"max {max(r['ensemble']['speedup']):.2f}x)")


if __name__ == "__main__":
    grade()
