"""Honesty invariants for the sbml-decomposition workspace.

These tests gate the derive-first verdict foundation: investigation titles must
state their forward/backward direction, no authored verdict may be more
favorable than the discovery-engine's computed verdict, and both investigations
must ship a rendered report. They read the committed investigation YAML directly
so they run in CI with no server or heavy imports.
"""
import pathlib

import pytest
import yaml

INV = pathlib.Path("workspace/investigations")

# Verdict favorability ladder. A lower number is a weaker (less favorable)
# claim. `complete`/`closed` are lifecycle values, but when they appear as a
# verdict they read as a passing-level overclaim, so they grade at passing level.
_FAVOR = {
    "failing": 0,
    "in-progress": 1,
    "passing-with-caveats": 2,
    "passing": 3,
    "passed": 3,
    "pass": 3,
    "accepted": 3,
    "complete": 3,
    "closed": 3,
}


def _invs():
    for p in sorted(INV.glob("*/investigation.yaml")):
        yield p.parent.name, yaml.safe_load(p.read_text())


def test_investigation_titles_state_direction():
    dec = yaml.safe_load((INV / "sbml-decomposition/investigation.yaml").read_text())
    rec = yaml.safe_load((INV / "discovery-sbml-recovery/investigation.yaml").read_text())
    assert "Decomposition" in dec["title"]
    assert "Recovery" in rec["title"] and "Recover" not in dec["title"]


@pytest.mark.parametrize("name,spec", list(_invs()))
def test_no_verdict_overclaim(name, spec):
    """No authored verdict may be more favorable than the computed one.

    Derive-first: an investigation with no authored verdict cannot overclaim.
    Where an authored value remains, it must be no more favorable than the
    discovery-engine's `computed_verdict_status`. A conservative authored
    verdict (e.g. passing-with-caveats over computed passing) is allowed.
    """
    ex = spec.get("executive", {}) or {}
    authored = (ex.get("verdict_status") or "").strip().lower()
    computed = (ex.get("computed_verdict_status") or "").strip().lower()
    if not authored:
        return
    assert computed, f"{name}: authored verdict but no computed_verdict_status"
    assert _FAVOR.get(authored, 3) <= _FAVOR[computed], (
        f"{name}: authored '{authored}' is more favorable than computed '{computed}'"
    )


def test_both_investigations_have_reports():
    for slug in ("sbml-decomposition", "discovery-sbml-recovery"):
        rpt = INV / slug / "report.html"
        assert rpt.exists() and rpt.stat().st_size >= 20000, (
            f"{slug}: missing or empty report.html"
        )
