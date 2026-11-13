#!/usr/bin/env python
"""Run + grade the cross-engine-validation study (code-authoritative).

Creates one canonical run on the study and lets the discovery-engine grade its
behavior test through this workspace's ``cross_engine_agreement`` evaluator,
which re-runs the agreement harness (tellurium vs COPASI vs native on glycolysis)
and writes ``computed_outcomes``. Mirrors the recovery grade path. Deterministic
apart from COPASI's documented non-bit-reproducibility (tolerated, not gated).

Usage:  python scripts/grade_cross_engine.py
"""
from __future__ import annotations

import time
from pathlib import Path

import ruamel.yaml
from ruamel.yaml.comments import CommentedMap

try:  # engine ≥ nested study.* layout (viva-discovery-engine)
    from discovery_engine.study.evaluator import clear_workspace_evaluator_cache, compute_outcomes
    from discovery_engine.study.verdict import roll_up_verdict, write_gate_evaluator
except ImportError:  # older engine: flat module layout
    from discovery_engine.study_evaluator import clear_workspace_evaluator_cache, compute_outcomes
    from discovery_engine.study_verdict import roll_up_verdict, write_gate_evaluator

SLUG = "cross-engine-validation"
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

    # One canonical run to carry the computed grades (there is no emitter; the
    # evaluator re-derives the agreement). Replace any prior cross-engine run.
    run = CommentedMap()
    run["name"] = f"cross-engine:{int(time.time())}"
    run["status"] = "completed"
    run["canonical"] = True
    runs = d.setdefault("runs", [])
    runs[:] = [r for r in runs if not str(r.get("name", "")).startswith("cross-engine:")]
    for r in runs:
        if r.get("canonical"):
            r["canonical"] = False
    runs.insert(0, run)
    d["status"] = "ran"
    with p.open("w") as f:
        y.dump(d, f)

    clear_workspace_evaluator_cache()
    compute_outcomes(sd, ws_root=ws)  # -> computed_outcomes via the evaluators hook
    write_gate_evaluator(sd)
    verdict = roll_up_verdict(ruamel.yaml.YAML().load(p.read_text()))
    print(f"{SLUG}: gate -> {verdict.get('result')}")

    _write_figure_data(ws)
    return verdict


def _write_figure_data(ws: Path) -> None:
    """Commit the per-engine agreement as a small JSON for the figure to read, so
    the report renders deterministically (no live COPASI at render time)."""
    import json

    from viva_sbml_decomposition.harness.cross_engine import cross_engine_agreement
    from viva_sbml_decomposition.models.glycolysis import GLYCOLYSIS_ANTIMONY, GLYCOLYSIS_META

    res = cross_engine_agreement(
        GLYCOLYSIS_ANTIMONY, t_end=GLYCOLYSIS_META["t_end"], n_points=301)
    out = ws / "viva_sbml_decomposition" / "visualizations" / "cross_engine_data.json"
    out.write_text(json.dumps({
        "model": "glycolysis", "reference": res["reference"],
        "agreement": res["agreement"], "aborted": res["aborted"],
        "note": "max relative error of each engine's monolithic run vs tellurium; COPASI not bit-reproducible",
    }, indent=2) + "\n")
    print(f"{SLUG}: wrote figure data -> {out.name} ({res['agreement']})")


if __name__ == "__main__":
    grade()
