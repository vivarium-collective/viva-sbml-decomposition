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

from discovery_engine.study_verdict import roll_up_verdict, write_gate_evaluator
from viva_sbml_decomposition.core import build_core
from viva_sbml_decomposition.evaluators import register_evaluators
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
    fitted = fit["fitted_antimony"]
    print(f"{study_slug}: fit nrmse {fit['nrmse']:.4g}; fitted params {fit['fitted_params']}")

    # Grade the FITTED (non-identity) candidate.
    reg: dict = {}
    register_evaluators(reg)
    y = ruamel.yaml.YAML()
    y.preserve_quotes = True
    p = sd / "study.yaml"
    d = y.load(p.read_text())
    outcomes = CommentedMap()
    for t in d.get("behavior_tests") or []:
        kind = (t.get("measure") or {}).get("kind")
        fn = reg.get(kind)
        if not fn:
            continue
        test = {"measure": {**(t.get("measure") or {}), "candidate_model": fitted}}
        o = fn(test, None, str(ws))
        outcomes[t["name"]] = {"result": "ok" if o["result"] == "PASS" else "fail",
                               "detail": o["detail"] + " (recovered candidate, 2x-param start)"}

    run = CommentedMap()
    run["name"] = f"recovery-fit:{study_slug}:{int(time.time())}"
    run["status"] = "completed"
    run["outcomes"] = outcomes
    run["canonical"] = True
    d.setdefault("runs", []).insert(0, run)  # make it the canonical run
    d["status"] = "ran"
    with p.open("w") as f:
        y.dump(d, f)
    write_gate_evaluator(sd)
    verdict = roll_up_verdict(ruamel.yaml.YAML().load(p.read_text()))
    print(f"{study_slug}: recovered-candidate gate -> {verdict.get('result')}")
    return verdict


if __name__ == "__main__":
    recover(sys.argv[1] if len(sys.argv) > 1 else "recover-mass-action")
