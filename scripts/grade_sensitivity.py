#!/usr/bin/env python
"""Run + grade the sensitivity-uq study (code-authoritative).

Creates one canonical run, grades its behavior test through this workspace's
`sensitivity_ranking` evaluator (which runs the seeded Morris screen), and commits
the machine-tagged indices baseline the figure reads. Morris on the deterministic
ODE is reproducible given the seed; the graded property is the ranking's
discrimination, not raw magnitudes.

Usage:  python scripts/grade_sensitivity.py
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

SLUG = "sensitivity-uq"
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
    run["name"] = f"sensitivity:{int(time.time())}"
    run["status"] = "completed"
    run["canonical"] = True
    runs = d.setdefault("runs", [])
    runs[:] = [r for r in runs if not str(r.get("name", "")).startswith("sensitivity:")]
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
    """Commit the machine-tagged Morris indices the figure reads."""
    from viva_sbml_decomposition.harness.sensitivity import morris_sensitivity

    r = morris_sensitivity(r=6, levels=4, span=0.2, seed=0, qoi="ATP")
    out = ws / "viva_sbml_decomposition" / "visualizations" / "sensitivity_indices.json"
    out.write_text(json.dumps(r, indent=2) + "\n")
    print(f"{SLUG}: wrote baseline -> {out.name} (top {r['ranking'][0]}, backend {r['backend']})")


if __name__ == "__main__":
    grade()
