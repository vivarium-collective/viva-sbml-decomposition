#!/usr/bin/env python
"""Non-identity recovery — the build loop in action.

Starts from a WRONG candidate (the oracle's structure with every rate constant
doubled), fits it back to the target with RateLawFitter, grades the RECOVERED
(non-identity) model with the recovery evaluators, and records it as a run. Unlike
the identity baseline, this proves the fit step actually recovers a model that was
not the answer key.

Usage:  python scripts/recover.py [study-slug]   (default: recover-mass-action)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import ruamel.yaml
import tellurium as te
from ruamel.yaml.comments import CommentedMap

try:  # engine ≥ nested study.* layout (viva-discovery-engine)
    from discovery_engine.study.verdict import roll_up_verdict, write_gate_evaluator
except ImportError:  # older engine: flat module layout
    from discovery_engine.study_verdict import roll_up_verdict, write_gate_evaluator
from viva_sbml_decomposition.core import build_core
from viva_sbml_decomposition.models.recovery_targets import RECOVERY_TARGETS
from viva_sbml_decomposition.processes.recovery import RateLawFitter


def _ws_root() -> Path:
    node = Path(__file__).resolve().parent
    for _ in range(8):
        if (node / "workspace.yaml").is_file():
            return node
        node = node.parent
    return Path.cwd()


def recover(study_slug: str = "recover-mass-action") -> dict:
    ws = _ws_root()
    sd = ws / "workspace" / "investigations" / "discovery-sbml-recovery" / "studies" / study_slug
    core = build_core()
    truth = RECOVERY_TARGETS[study_slug]["antimony"]

    # Wrong candidate: the oracle structure with every rate constant doubled.
    rr = te.loada(truth)
    for p in rr.getGlobalParameterIds():
        rr[p] = float(rr[p]) * 2.0
    wrong = te.sbmlToAntimony(rr.getCurrentSBML())

    fit = RateLawFitter(config={"candidate_antimony": wrong, "target": study_slug, "max_iter": 800}, core=core).update({})
    print(f"{study_slug}: fit nrmse {fit['nrmse']:.4g}; fitted params {fit['fitted_params']}")

    # Record the fit as the canonical run. Grading is NOT authored here: recovery
    # is code-authoritative. grade_recovery.py's compute_outcomes independently
    # re-derives the recovered candidate (the @recovered evaluator) and writes the
    # computed grades, which the verdict reads computed-first. Writing an authored
    # ok/fail copy here only produced a spurious `reconcile: divergent` in the
    # rendered report (the engine compares authored ok/fail to computed PASS/FAIL
    # by ==), so we omit it — the honest flag is `no_authored`.
    y = ruamel.yaml.YAML()
    y.preserve_quotes = True
    p = sd / "study.yaml"
    d = y.load(p.read_text())

    run = CommentedMap()
    run["name"] = f"recovery-fit:{study_slug}:{int(time.time())}"
    run["status"] = "completed"
    run["canonical"] = True
    runs = d.setdefault("runs", [])
    # Replace any prior recovery-fit run for this study and clear other canonical
    # flags, so there is exactly one canonical run (this one) — never a stale pick.
    runs[:] = [r for r in runs if not str(r.get("name", "")).startswith(f"recovery-fit:{study_slug}")]
    for r in runs:
        if r.get("canonical"):
            r["canonical"] = False
    runs.insert(0, run)
    d["status"] = "ran"
    with p.open("w") as f:
        y.dump(d, f)
    write_gate_evaluator(sd)
    verdict = roll_up_verdict(ruamel.yaml.YAML().load(p.read_text()))
    print(f"{study_slug}: recovered-candidate gate -> {verdict.get('result')}")
    return verdict


if __name__ == "__main__":
    recover(sys.argv[1] if len(sys.argv) > 1 else "recover-mass-action")
