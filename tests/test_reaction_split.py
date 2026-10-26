"""Golden + property regression tests for the generic ``reaction_split``.

The generic composite was A/B-verified against the retired split/coupling arms
(aldolase tellurium/tellurium, tellurium/native, tellurium/COPASI: max|diff| 0.0 /
0.0 / ~1e-7). Those comparisons are replaced by frozen golden references plus the
properties the arm tests carried: convergence to the monolith as the coupling
interval shrinks, clean divergence out of the sqrt domain, a complete-and-disjoint
reaction partition, and native subset RHS halves summing to the full RHS.
"""
from __future__ import annotations

import functools

import numpy as np
import pytest

from _golden import assert_golden
from viva_sbml_decomposition.composites.engine_monolithic import run_engine_monolithic
from viva_sbml_decomposition.composites.reaction_split import (
    DEFAULT_COUPLING_INTERVAL, build_document, reaction_split, run_reaction_split,
    time_reaction_split)
from viva_sbml_decomposition.harness.accuracy import compare_to_reference
from viva_sbml_decomposition.models.glycolysis import GLYCOLYSIS_ANTIMONY, GLYCOLYSIS_META
from viva_sbml_decomposition.models.glycolysis_submodels import (
    LOWER_GLYCOLYSIS, SUBMODELS, UPPER_GLYCOLYSIS)
from viva_sbml_decomposition.processes import native_glycolysis as ng

SPECIES = GLYCOLYSIS_META["species"]


@functools.lru_cache(maxsize=None)
def _run(seam, engine_a, engine_b, dt):
    """Memoised run (several tests share the same expensive configurations)."""
    return run_reaction_split("glycolysis", seam, engine_a, engine_b, coupling_interval=dt)

GOLD_ALDOLASE_TT = {     # tellurium + tellurium, interval 0.0005
    "final": {"ATP": 2.4677100785424724, "F16P": 0.6168745876274058, "PEP": 0.07297508206544684},
    "checksum": 1428977.917047086, "n": 60001}
GOLD_ALDOLASE_TN = {     # tellurium upper + native lower, interval 0.005
    "final": {"ATP": 2.0796587463657192, "F16P": 0.7702095037914933, "PEP": 0.0699873856262165},
    "checksum": 134882.8634783272, "n": 6001}
GOLD_ALDOLASE_TC = {     # tellurium upper + COPASI lower, interval 0.005
    "final": {"ATP": 2.079658745433012, "F16P": 0.7702095019485735, "PEP": 0.06998738548688842},
    "checksum": 134882.86346574366, "n": 6001}
GOLD_MECHANICAL_005 = {  # mechanical seam, interval 0.05
    "final": {"ATP": 2.2843354569455223, "F16P": 0.6541448572779535, "PEP": 0.07763664884156604},
    "checksum": 14051.788752989249, "n": 601}
GOLD_MECHANICAL_0005 = {  # mechanical seam, interval 0.005
    "final": {"ATP": 2.4926831190946963, "F16P": 0.6017923001267751, "PEP": 0.0739433248515509},
    "checksum": 143404.57087947245, "n": 6001}


def test_aldolase_tellurium_golden():
    assert_golden(_run("aldolase", "tellurium", "tellurium", 0.0005), GOLD_ALDOLASE_TT)


def test_aldolase_tellurium_native_golden():
    assert_golden(_run("aldolase", "tellurium", "native", 0.005), GOLD_ALDOLASE_TN)


def test_mechanical_seam_golden():
    assert_golden(_run("mechanical", "tellurium", "tellurium", 0.05), GOLD_MECHANICAL_005)
    assert_golden(_run("mechanical", "tellurium", "tellurium", 0.005), GOLD_MECHANICAL_0005)


@pytest.mark.slow
def test_aldolase_tellurium_copasi_golden():
    # Runs in the abort-safe retrying subprocess; rtol 1e-5 (COPASI is not
    # bit-reproducible run to run, ~5e-8..1e-7 measured).
    assert_golden(_run("aldolase", "tellurium", "copasi", 0.005), GOLD_ALDOLASE_TC, rtol=1e-5)


# --- properties the arm tests carried ---------------------------------------------

# Interval -> error against the monolith (max relative error), measured:
#   aldolase 0.002 -> ~0.47 (outside T), 0.0005 -> ~0.10 (within T), T = 0.15.
#   mechanical 0.1 -> ~0.86 (outside T), 0.0125 -> ~0.11 (within T), T = 0.15.
TOLERANCE_T = 0.15


@pytest.fixture(scope="module")
def reference():
    return run_engine_monolithic("glycolysis", "tellurium", n_steps=300)


@pytest.mark.parametrize("seam,coarse,fine", [("aldolase", 0.002, 0.0005),
                                              ("mechanical", 0.1, 0.0125)])
def test_split_converges_to_monolith_as_interval_shrinks(reference, seam, coarse, fine):
    ref_traj, ref_t = reference
    errs = {}
    for dt in (coarse, fine):
        traj, t = _run(seam, "tellurium", "tellurium", dt)
        res = compare_to_reference(ref_traj, ref_t, traj, t)
        assert res.status == "ok", f"{seam} dt={dt} diverged"
        errs[dt] = res.max_rel_err
    # Splitting error shrinks with the interval, and T discriminates the two.
    assert errs[fine] < errs[coarse]
    assert errs[fine] < TOLERANCE_T, errs
    assert errs[coarse] > TOLERANCE_T, errs


def test_mechanical_interval_to_diff_is_monotone_against_finest_run():
    # Additive (Jacobi) coupling error is first order in the interval: against the
    # finest-interval run of the same partition the gap shrinks as dt shrinks.
    def final_gap(dt, fine):
        traj, _ = _run("mechanical", "tellurium", "tellurium", dt)
        return max(abs(traj[s][-1] - fine[s][-1]) for s in SPECIES)

    fine, _ = _run("mechanical", "tellurium", "tellurium", 0.001)
    gaps = [final_gap(dt, fine) for dt in (0.05, 0.01, 0.005)]
    assert gaps[0] > gaps[1] > gaps[2] > 0, gaps
    assert gaps[2] < gaps[0] / 3


def test_split_pairs_full_species_set_by_name(reference):
    traj, t = _run("aldolase", "tellurium", "tellurium", 0.005)
    assert set(traj) == set(SPECIES) and len(SPECIES) == 17
    assert t[0] == 0.0 and np.isclose(t[-1], GLYCOLYSIS_META["t_end"])
    for s in SPECIES:
        assert traj[s].shape == t.shape and np.all(np.isfinite(traj[s]))
    ref_traj, ref_t = reference
    with pytest.raises(ValueError):      # the harness pairs by name and refuses a gap
        compare_to_reference(ref_traj, ref_t, {k: v for k, v in traj.items() if k != "ATP"}, t)


def test_split_diverges_cleanly_out_of_domain(reference):
    # A whole 0.1 coupling interval in isolation drives the shared phosphate pool out
    # of the sqrt ADP-rule domain; SubmodelProcess turns the CVODE failure into a
    # non-finite delta, so the run completes and is reported as diverged.
    ref_traj, ref_t = reference
    traj, t = _run("aldolase", "tellurium", "tellurium", 0.1)
    res = compare_to_reference(ref_traj, ref_t, traj, t)
    assert res.status == "diverged" and not np.isfinite(res.max_rel_err)


def test_tellurium_native_reproduces_tellurium_tellurium_at_same_interval():
    # Same ODEs, lower half on a different engine: at a matched interval the common
    # splitting error cancels, so the two agree to ~1e-7. A broken partition or
    # native subset RHS would blow this up.
    a = _run("aldolase", "tellurium", "tellurium", 0.005)
    b = _run("aldolase", "tellurium", "native", 0.005)
    res = compare_to_reference(*a, *b)
    assert not res.diverged and res.max_rel_err < 1e-3, res.max_rel_err


def test_cross_engine_run_is_finite_and_sensible():
    traj, t = _run("aldolase", "tellurium", "native", 0.005)
    for s in SPECIES:
        v = traj[s]
        assert np.all(np.isfinite(v)) and np.all(v > -1e-6) and np.all(v < 1e4), s


def test_partition_is_a_biological_seam():
    assert [name for name, _ in SUBMODELS] == ["upper_glycolysis", "lower_glycolysis"]
    for _, group in SUBMODELS:
        assert len(group) >= 1


def test_reaction_partition_is_complete_and_disjoint():
    upper, lower = set(UPPER_GLYCOLYSIS), set(LOWER_GLYCOLYSIS)
    assert upper.isdisjoint(lower), f"overlap: {upper & lower}"
    assert upper | lower == set(ng.STOICH), (
        f"missing {set(ng.STOICH) - (upper | lower)}, extra {(upper | lower) - set(ng.STOICH)}")


def test_native_submodel_halves_sum_to_the_full_rhs():
    upper_rhs = ng.make_subset_rhs(UPPER_GLYCOLYSIS)
    lower_rhs = ng.make_subset_rhs(LOWER_GLYCOLYSIS)
    rng = np.random.default_rng(0)
    for y in (np.array([ng._P[s] for s in ng.SETTABLE], dtype=float),
              np.abs(rng.normal(1.0, 0.3, len(ng.SETTABLE)))):
        np.testing.assert_allclose(upper_rhs(0.0, y) + lower_rhs(0.0, y), ng._rhs(0.0, y),
                                   rtol=0, atol=1e-9)


def test_timed_split_reports_framework_split():
    traj, t, rt = time_reaction_split("glycolysis", "mechanical", "tellurium", "tellurium",
                                      0.1)
    assert all(np.all(np.isfinite(v)) for v in traj.values())
    assert 0.0 <= rt.overhead_fraction <= 1.0 and rt.per_process


# --- structure (no engine run) ----------------------------------------------------

@pytest.mark.parametrize("engine_b", ["tellurium", "copasi", "native"])
def test_config_model_and_shared_store(engine_b):
    doc = reaction_split(model="glycolysis", seam="aldolase", engine_b=engine_b)
    procs = [v for v in doc.values() if isinstance(v, dict) and v.get("_type") == "process"]
    assert len(procs) == 2
    for p in procs:
        assert p["config"]["model"] == GLYCOLYSIS_ANTIMONY
        assert p["inputs"] == {"species": ["species"]}
        assert p["outputs"] == {"species": ["species"]}
    assert "emitter" not in doc


def test_explicit_seam_and_errors():
    doc = build_document("glycolysis", {"upper": ["vGLT", "vGLK"], "lower": ["vPGI"]})
    assert doc["upper"]["config"]["reactions"] == ["vGLT", "vGLK"]
    with pytest.raises(ValueError):
        build_document("glycolysis", {"upper": ["vGLT"], "lower": ["vGLT"]})
    with pytest.raises(ValueError):
        build_document("glycolysis", "nope")


@pytest.mark.slow
def test_tellurium_copasi_reproduces_tellurium_tellurium_at_same_interval():
    # COPASI engine-swap drop-in: measured ~3.2e-7 (not bit-reproducible run to run).
    a = _run("aldolase", "tellurium", "tellurium", 0.005)
    b = _run("aldolase", "tellurium", "copasi", 0.005)
    res = compare_to_reference(*a, *b)
    assert not res.diverged and res.max_rel_err < 1e-3, res.max_rel_err
