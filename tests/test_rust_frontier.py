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
