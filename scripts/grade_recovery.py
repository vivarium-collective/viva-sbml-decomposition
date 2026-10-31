#!/usr/bin/env python
"""Grade a recovery study end-to-end (reproducible closed-loop step).

Runs the discovery-engine evaluation (which invokes this workspace's
``viva_sbml_decomposition.evaluators`` hook — TrajectoryComparator + StructuralScorer
against the hidden oracle), then promotes the deterministic code grades to authored
``outcomes`` and recomputes the pipeline gate. Recovery grading is a deterministic
metric vs a known oracle, so the code IS the authority — promotion is the reconcile.

Usage:  python scripts/grade_recovery.py <study-slug> [investigation-slug]
"""
from __future__ import annotations

import sys
from pathlib import Path

import ruamel.yaml
from ruamel.yaml.comments import CommentedMap

from discovery_engine.study_evaluator import clear_workspace_evaluator_cache, compute_outcomes
from discovery_engine.study_outcomes import canonical_run
from discovery_engine.study_verdict import roll_up_verdict, write_gate_evaluator


def _ws_root() -> Path:
    node = Path(__file__).resolve().parent
    for _ in range(8):
        if (node / "workspace.yaml").is_file():
            return node
        node = node.parent
    return Path.cwd()


def grade(study_slug: str, inv_slug: str = "discovery-sbml-recovery") -> dict:
    ws = _ws_root()
    sd = ws / "workspace" / "investigations" / inv_slug / "studies" / study_slug
    if not (sd / "study.yaml").is_file():
        raise SystemExit(f"no study.yaml at {sd}")
    clear_workspace_evaluator_cache()
    compute_outcomes(sd, ws_root=ws)  # -> computed_outcomes via the evaluators hook

    y = ruamel.yaml.YAML()
    y.preserve_quotes = True
    p = sd / "study.yaml"
    d = y.load(p.read_text())
    run = canonical_run(d.get("runs")) or {}
    co = run.get("computed_outcomes") or {}
    outs = run.get("outcomes")
    if not isinstance(outs, (dict, CommentedMap)):
        outs = CommentedMap()
        run["outcomes"] = outs
    promoted = 0
    for name, c in co.items():
        if isinstance(c, dict) and str(c.get("result", "")).upper() in ("PASS", "FAIL"):
            outs[name] = {
                "result": "ok" if c["result"] == "PASS" else "fail",
                "detail": str(c.get("detail", "")) + " (code-graded)",
            }
            promoted += 1
    if run:
        d["status"] = "ran"
    with p.open("w") as f:
        y.dump(d, f)
    write_gate_evaluator(sd)
    verdict = roll_up_verdict(ruamel.yaml.YAML().load(p.read_text()))
    print(f"{study_slug}: promoted {promoted} outcome(s) -> gate {verdict.get('result')}")
    return verdict


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    grade(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "discovery-sbml-recovery")
