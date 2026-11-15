"""Phase 6 — Rust-engine frontier probe + honest-characterization grade.

None of these tests import viva_core or require a Rust toolchain: the probe
degrades without it, and the committed baseline JSON is what CI reads.
"""
import json
import pathlib

_BASELINE = pathlib.Path(
    "viva_sbml_decomposition/visualizations/rust_frontier.json")

_KEYS = {"viva_core_available", "native_composite_runs", "ode_speedup",
         "dispatch_speedup", "dispatch_note", "ode_note", "viva_hybrid_note",
         "requires_port", "caveat", "machine"}


def test_probe_degrades_without_viva_core():
    from viva_sbml_decomposition.harness.rust_frontier import rust_frontier_probe
    r = rust_frontier_probe(viva_core_python="/nonexistent/path")
    assert isinstance(r, dict) and _KEYS <= set(r)
    assert r["viva_core_available"] is False
    assert r["native_composite_runs"] is False
    assert r["requires_port"] is True
    assert r["ode_speedup"] is None
    assert isinstance(r["dispatch_note"], str) and r["dispatch_note"]
    assert isinstance(r["ode_note"], str) and r["ode_note"]


def test_committed_baseline_is_well_formed():
    r = json.loads(_BASELINE.read_text())
    assert _KEYS <= set(r)
    assert r["ode_speedup"] is None or isinstance(r["ode_speedup"], (int, float))
    assert isinstance(r["requires_port"], bool)
    assert isinstance(r["caveat"], str) and r["caveat"]
    assert isinstance(r["machine"], str) and r["machine"]


def test_honesty_grade_passes_on_honest_json():
    from viva_sbml_decomposition.evaluators import _rust_frontier_verdict
    honest = json.loads(_BASELINE.read_text())           # requires_port=True, ode_speedup=None
    passed, measured, _ = _rust_frontier_verdict(honest)
    assert passed is True


def test_honesty_grade_fails_on_false_ode_win():
    from viva_sbml_decomposition.evaluators import _rust_frontier_verdict
    false_win = {"requires_port": False, "native_composite_runs": True, "ode_speedup": 6.0,
                 "dispatch_note": "x", "ode_note": "x"}
    passed, measured, _ = _rust_frontier_verdict(false_win)
    assert passed is False
    assert measured == 6.0


def test_rust_frontier_study_is_code_graded_and_passes():
    import yaml
    sd = pathlib.Path(
        "workspace/investigations/sbml-decomposition/studies/rust-engine-frontier/study.yaml")
    d = yaml.safe_load(sd.read_text())
    assert d["investigation"] == "sbml-decomposition"
    canonical = next((r for r in (d.get("runs") or []) if r.get("canonical")), None)
    assert canonical is not None, "no canonical run"
    grade = (canonical.get("computed_outcomes") or {}).get("rust-frontier-characterized-honestly") or {}
    assert grade.get("result") == "PASS"
    assert grade.get("evaluated_by") == "code"
    assert grade.get("reconcile") == "no_authored"
