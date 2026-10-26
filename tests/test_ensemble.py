"""Golden + property regression tests for the generic ``ensemble``.

The generic ensemble was A/B-verified against the retired same-rate (Ray) and
multi-rate (#270 heap) arms (max|diff| 0.0). Those comparisons are replaced by
frozen golden references plus the scheduler/honesty properties the arm tests carried.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pytest
from process_bigraph import Composite, allocate_core

from viva_sbml_decomposition.composites import ensemble as ens
from viva_sbml_decomposition.composites import abm
from viva_sbml_decomposition.composites.engine_monolithic import run_engine_monolithic
from viva_sbml_decomposition.models.glycolysis import GLYCOLYSIS_ANTIMONY, GLYCOLYSIS_META

SPECIES = GLYCOLYSIS_META["species"]
N = 3
RTOL = 1e-7

# Seeded draws: scale the two first global parameters by U(0.9, 1.1), per instance.
BASE_PARAMS = {"KeqAK": 0.45, "vGLK_VmGLK": 226.452}
GOLD_SAME_RATE = [   # final state of each of the N=3 instances (checksum = sum of all 17)
    {"ATP": 2.474943425068273, "F16P": 0.5856773311360585, "PEP": 0.07118005086329016,
     "checksum": 23.472713524616935},
    {"ATP": 2.4238861138450982, "F16P": 0.5597016278544201, "PEP": 0.06644413767724096,
     "checksum": 23.072125653634732},
    {"ATP": 2.5569901598797387, "F16P": 0.6284219033144269, "PEP": 0.07837189487538783,
     "checksum": 24.24016897292032},
]
GOLD_MULTIRATE = [   # n=4 geometric intervals, horizon 10, heap: (ATP final, rows, checksum)
    {"ATP": 2.508390624951976, "n": 17, "sum": 407.90397735262616},
    {"ATP": 2.508390624926786, "n": 17, "sum": 408.2452531834545},
    {"ATP": 2.5083906255125354, "n": 17, "sum": 417.69301307316454},
    {"ATP": 2.5091904762788237, "n": 17, "sum": 420.529},
]


def _draws(n):
    rng = np.random.default_rng(0)
    return [{p: v * float(rng.uniform(0.9, 1.1)) for p, v in BASE_PARAMS.items()}
            for _ in range(n)]


@pytest.fixture(autouse=True)
def _shutdown_ray():
    """Keep CI clean: tear down any Ray cluster a test started."""
    yield
    try:
        import ray
        if ray.is_initialized():
            ray.shutdown()
    except Exception:
        pass


def test_registered_and_carries_model():
    doc = ens.ensemble(n_instances=2)
    assert doc["inst_0"]["config"]["model"].strip() == GLYCOLYSIS_ANTIMONY.strip()
    assert not any(k.startswith("emitter") for k in doc)
    assert callable(abm.predator_prey_abm) and callable(abm.predator_prey_meanfield)


def test_same_rate_final_states_golden():
    res = ens.run_ensemble("glycolysis", "tellurium", N, param_draws=_draws(N))
    assert len(res) == N and res.backend in ("ray", "serial")
    for got, gold in zip(res, GOLD_SAME_RATE):
        assert set(got) == set(SPECIES)
        for k in ("ATP", "F16P", "PEP"):
            np.testing.assert_allclose(got[k], gold[k], rtol=RTOL, err_msg=k)
        np.testing.assert_allclose(sum(got.values()), gold["checksum"], rtol=RTOL)


def test_same_rate_full_trajectories_equal_the_direct_monolith():
    draws = _draws(N)
    res = ens.run_ensemble("glycolysis", "tellurium", N, param_draws=draws, full=True)
    d = 0.0
    for (traj, t), dr in zip(res, draws):
        ref, rt = run_engine_monolithic("glycolysis", "tellurium", parameter_overrides=dr)
        d = max(d, float(np.max(np.abs(t - rt))),
                max(float(np.max(np.abs(traj[s] - ref[s]))) for s in ref))
    assert d < 1e-9


def test_ensemble_runs_all_instances_and_draws_perturb_them():
    results = ens.run_ensemble(n_instances=4)
    assert len(results) == 4 and results.backend in ("ray", "serial")
    for res in results:
        assert set(res) == set(SPECIES) and len(SPECIES) == 17
        assert all(math.isfinite(float(v)) for v in res.values())
    # +/-10% hexokinase Vmax: integrable and distinct.
    pert = ens.run_ensemble(n_instances=2, param_draws=[{"vGLK_VmGLK": 226.452 * 1.10},
                                                        {"vGLK_VmGLK": 226.452 * 0.90}])
    a = np.array([pert[0][s] for s in SPECIES])
    b = np.array([pert[1][s] for s in SPECIES])
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and not np.allclose(a, b)


@pytest.mark.slow
def test_speedup_measurement_runs():
    r = ens.ensemble_speedup("glycolysis", "tellurium", 4, repeats=1)
    print("speedup:", r)
    assert r.serial_s > 0 and r.n_instances == 4
    assert r.backend in ("ray", "serial")
    if r.speedup_claimed:
        assert r.speedup > 0 and r.parallel_s > 0


def test_parallel_or_declares_serial():
    """HONESTY GATE. Either Ray delivers a real wall-clock speedup, or the result
    honestly declares a serial fallback with no speedup claim."""
    result = ens.ensemble_speedup(n_instances=8)
    try:
        import ray  # noqa: F401
        have_ray = True
    except ImportError:
        have_ray = False
    if (os.cpu_count() or 1) >= 4 and have_ray:
        if not result.speedup_claimed:      # one retry against timing noise
            result = ens.ensemble_speedup(n_instances=8)
        assert result.speedup_claimed is True, result
        assert result.setup_excluded is True and result.setup_s is not None
    if result.speedup_claimed:
        assert result.backend == "ray"
        assert result.parallel_s < result.serial_s
        assert result.speedup > 1.0
        assert math.isclose(result.speedup, result.serial_s / result.parallel_s, rel_tol=1e-6)
    else:
        assert result.backend == "serial" and result.speedup is None


def test_geometric_intervals_are_distinct_and_ordered():
    ivs = ens.geometric_intervals(8)
    assert len(ivs) == 8 and len(set(ivs)) == 8
    assert all(a < b for a, b in zip(ivs, ivs[1:]))
    assert ens._resolve_intervals("geometric", 4, 10.0) == ens.geometric_intervals(4, 10.0)


def test_multirate_heap_golden():
    if "scheduler" not in Composite.config_schema:
        pytest.skip("process-bigraph without #270 (no 'scheduler' config field)")
    got = ens.run_multirate("glycolysis", "tellurium", 4, "geometric", horizon=10.0,
                            scheduler="heap")
    assert len(got) == len(GOLD_MULTIRATE)
    for (traj, t), gold in zip(got, GOLD_MULTIRATE):
        assert len(t) == gold["n"]
        np.testing.assert_allclose(traj["ATP"][-1], gold["ATP"], rtol=RTOL)
        np.testing.assert_allclose(sum(float(np.sum(v)) for v in traj.values()),
                                   gold["sum"], rtol=RTOL)


def _final_stores(n, scheduler, horizon=10.0):
    comp = ens._make_composite(n, scheduler, horizon)
    comp.run(horizon)
    return ens._final_state(comp)


def test_scan_and_heap_are_bit_identical_and_finite():
    if "scheduler" not in Composite.config_schema:
        pytest.skip("process-bigraph without #270 (no 'scheduler' config field)")
    n = 6
    scan, heap = _final_stores(n, "scan"), _final_stores(n, "heap")
    assert set(scan) == set(heap) == {f"s{i}" for i in range(n)}
    for k in scan:
        assert scan[k].shape == (17,) and np.all(np.isfinite(scan[k]))
        assert np.all(scan[k] >= -1e-9)
        assert np.array_equal(scan[k], heap[k]), f"scan != heap for {k}"      # same math
    assert np.any(scan["s0"] > 0)


def test_multirate_scaling_shape_bit_identical_and_trend():
    if "scheduler" not in Composite.config_schema:
        pytest.skip("process-bigraph without #270 (no 'scheduler' config field)")
    rows = ens.multirate_scaling(ns=(8, 64), horizon=10.0, repeats=3)
    assert [r["n"] for r in rows] == [8, 64]
    for r in rows:
        assert set(r) == {"n", "scan_s", "heap_s", "speedup", "bit_identical"}
        assert r["scan_s"] > 0 and np.isfinite(r["scan_s"])
        assert r["heap_s"] > 0 and np.isfinite(r["heap_s"])
        assert r["speedup"] == pytest.approx(r["scan_s"] / r["heap_s"])
        assert r["bit_identical"] is True
    # Heap's advantage over scan GROWS with P (O(P^2) -> O(P log P)); a trend with a
    # floor, not a strict wall-clock win.
    assert rows[-1]["speedup"] > rows[0]["speedup"]
    assert rows[-1]["speedup"] > 0.9


def test_multirate_figure_renders_plotly():
    from viva_sbml_decomposition.visualizations.multirate_scaling import MultirateScaling

    inst = MultirateScaling(config={}, core=allocate_core())
    out = inst.update(MultirateScaling.generate_state())
    assert set(out) == {"html"} and isinstance(out["html"], str)
    assert "Plotly.newPlot" in out["html"] and "viva_MultirateScaling" in out["html"]
    spec = inst.figure(MultirateScaling.generate_state())
    assert len(spec["data"]) == 3 and "layout" in spec
    import plotly.graph_objects as go

    go.Figure(spec)


def test_multirate_list_intervals_and_validation():
    doc = ens.build_document(n_instances=2, intervals=[0.5, 2.0])
    assert [doc[f"inst_{i}"]["interval"] for i in range(2)] == [0.5, 2.0]
    with pytest.raises(ValueError):
        ens.build_document(n_instances=2, intervals=[0.5])
    with pytest.raises(ValueError):
        ens.build_document(model="glycolysis", engine="copasi", n_instances=2,
                           param_draws=[{"a": 1.0}, None])
