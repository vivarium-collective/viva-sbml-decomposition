"""Tests for the recovery evaluator hook + the capability-gap build hook."""
from __future__ import annotations

import pytest

from viva_sbml_decomposition.build_hook import build_capability, register_build_hook
from viva_sbml_decomposition.evaluators import register_evaluators


@pytest.fixture
def evaluators():
    reg: dict = {}
    register_evaluators(reg)
    return reg


def test_all_five_kinds_registered(evaluators):
    assert set(evaluators) == {
        "recovery_trajectory", "recovery_structure", "recovery_parsimony",
        "recovery_bistability", "recovery_limit_cycle", "recovery_non_recoverable",
    }


@pytest.mark.parametrize("slug,kinds", [
    ("recover-mass-action", ["recovery_trajectory", "recovery_structure", "recovery_parsimony"]),
    ("recover-enzyme-kinetics", ["recovery_trajectory", "recovery_structure", "recovery_parsimony"]),
    ("recover-gene-switch", ["recovery_trajectory", "recovery_structure", "recovery_bistability"]),
    ("recover-oscillator", ["recovery_trajectory", "recovery_structure", "recovery_limit_cycle"]),
])
def test_identity_grades_pass(evaluators, slug, kinds):
    for kind in kinds:
        measure = {"target": slug, "tol": 0.08}
        out = evaluators[kind]({"measure": measure}, None, None)
        assert out["result"] == "PASS", (slug, kind, out)
        assert out["evaluated_by"] == "code"


def test_structure_fails_for_wrong_param(evaluators):
    bad = """
model c() species A,B,C; A=10;B=0;C=0; k1=0.9;k2=0.3;k2r=0.1;
r1:A->B;k1*A; r2:A+B->C;k2*A*B; r2r:C->A+B;k2r*C; end
""".strip()
    out = evaluators["recovery_structure"](
        {"measure": {"target": "recover-mass-action", "candidate_model": bad, "param_tol": 0.1}}, None, None)
    assert out["result"] == "FAIL"


def test_build_hook_reuse_matches_built_capability():
    out = build_capability({"capability": "rate-law-fitter", "id": "cg:abc123"})
    assert out["status"] == "reuse"
    assert out["process"].endswith("RateLawFitter")
    assert out["gap_id"] == "cg:abc123"


def test_build_hook_needs_implementation_for_unknown():
    out = build_capability({"capability": "non-recoverability-detector", "id": "cg:def456"})
    assert out["status"] == "needs_implementation"
    assert out["process"] is None
    assert out["gap_id"] == "cg:def456"


def test_register_build_hook():
    reg: dict = {}
    register_build_hook(reg)
    assert callable(reg.get("build_capability"))
