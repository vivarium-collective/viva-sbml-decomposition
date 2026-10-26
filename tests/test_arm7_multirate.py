"""Arm 7 — multi-rate scheduler stress test (process-bigraph #270 event-heap).

Asserts the scheduler-level claims, not biology: (1) the composite builds and runs
for a small N under BOTH schedulers and produces finite output that is BIT-IDENTICAL
between scan and heap (same math, different tick order); (2) ``multirate_scaling``
returns the documented shape and the heap/scan speedup rises with P and beats scan
at the larger P; (3) the figure renders interactive Plotly html.
"""
import numpy as np
import pytest
from process_bigraph import Composite, allocate_core
from viva_tellurium.composites import register_tellurium

from viva_sbml_decomposition.composites.arm7_multirate import (
    DEFAULT_N_INSTANCES,
    build_document,
    geometric_intervals,
    glycolysis_multirate,
    multirate_scaling,
)

HORIZON = 10.0  # short horizon keeps the test fast; the scheduler claim is N-driven


def _run(n_instances, scheduler, horizon=HORIZON):
    core = allocate_core()
    register_tellurium(core)
    comp = Composite({"state": build_document(n_instances, horizon),
                      "scheduler": scheduler}, core=core)
    comp.run(horizon)
    stores = comp.state["stores"]
    return {k: np.asarray(v, dtype=float) for k, v in stores.items()}


def test_geometric_intervals_are_distinct_and_ordered():
    ivs = geometric_intervals(8)
    assert len(ivs) == 8
    assert all(a < b for a, b in zip(ivs, ivs[1:]))        # strictly increasing
    assert len(set(ivs)) == 8                              # genuinely multi-rate


def test_stripped_generator_has_no_emitter_and_n_nodes():
    doc = glycolysis_multirate(n_instances=5)
    assert not any(k == "emitter" or k.startswith("emitter_") for k in doc)
    assert sum(1 for k in doc if k.startswith("inst_")) == 5
    # default generator uses the module default
    assert sum(1 for k in glycolysis_multirate() if k.startswith("inst_")) == DEFAULT_N_INSTANCES


def test_scan_and_heap_are_bit_identical_and_finite():
    n = 6
    scan = _run(n, "scan")
    heap = _run(n, "heap")
    assert set(scan) == set(heap) == {f"s{i}" for i in range(n)}
    for k in scan:
        assert scan[k].shape == (17,)
        assert np.all(np.isfinite(scan[k]))
        assert np.all(scan[k] >= -1e-9)                    # concentrations sensible
        # Same math, different scheduler -> identical to the bit.
        assert np.array_equal(scan[k], heap[k]), f"scan != heap for {k}"
    # The instances are independent + multi-rate, so different instances (different
    # intervals) need not hold identical state — but the fastest/finest should have
    # integrated to a sensible non-trivial trajectory.
    assert np.any(scan["s0"] > 0)


def test_multirate_scaling_shape_and_scheduler_win():
    rows = multirate_scaling(ns=(16, 64), horizon=HORIZON, repeats=2)
    assert [r["n"] for r in rows] == [16, 64]
    for r in rows:
        assert set(r) == {"n", "scan_s", "heap_s", "speedup"}
        assert r["scan_s"] > 0 and np.isfinite(r["scan_s"])
        assert r["heap_s"] > 0 and np.isfinite(r["heap_s"])
        assert r["speedup"] == pytest.approx(r["scan_s"] / r["heap_s"])
    # Heap's advantage over scan grows with P (the O(P^2) -> O(P log P) claim) and
    # the heap beats the stock scan at the larger P. Reported as measured.
    assert rows[-1]["speedup"] > rows[0]["speedup"]
    assert rows[-1]["speedup"] > 1.0


def test_multirate_figure_renders_plotly():
    from viva_sbml_decomposition.visualizations.multirate_scaling import MultirateScaling

    inst = MultirateScaling(config={}, core=allocate_core())
    out = inst.update(MultirateScaling.generate_state())
    assert set(out) == {"html"} and isinstance(out["html"], str)
    assert "Plotly.newPlot" in out["html"] and "viva_MultirateScaling" in out["html"]
    spec = inst.figure(MultirateScaling.generate_state())
    assert len(spec["data"]) == 3 and "layout" in spec
    import plotly.graph_objects as go

    go.Figure(spec)  # valid plotly spec
