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

try:  # engine ≥ nested study.* layout (viva-discovery-engine)
    from discovery_engine.study.evaluator import clear_workspace_evaluator_cache, compute_outcomes
    from discovery_engine.study.outcomes import canonical_run
    from discovery_engine.study.verdict import roll_up_verdict, write_gate_evaluator
except ImportError:  # older engine: flat module layout
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
    # Recovery is code-authoritative: compute_outcomes (above) wrote the computed
    # grades into the canonical run, and the verdict reads them computed-first
    # (discovery_engine.study_outcomes.canonical_graded_outcomes). We deliberately
    # do NOT promote them into an authored `outcomes` block: a promoted copy can
    # never truly diverge, and an authored ok/fail copy compared against computed
    # PASS/FAIL only produced a spurious `reconcile: divergent` in the rendered
    # report. With no authored copy the honest reconcile flag is `no_authored`.
    if run:
        d["status"] = "ran"
    with p.open("w") as f:
        y.dump(d, f)
    write_gate_evaluator(sd)
    verdict = roll_up_verdict(ruamel.yaml.YAML().load(p.read_text()))
    print(f"{study_slug}: code-graded -> gate {verdict.get('result')}")
    return verdict


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    grade(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "discovery-sbml-recovery")
