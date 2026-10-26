"""A/B: the generic ``reaction_split`` reproduces split/coupling arms 2, 3, 6, 10-cross.

Full trajectories (every species, every timepoint) are compared, not a scalar.
"""
from __future__ import annotations

import numpy as np
import pytest

from viva_sbml_decomposition.composites import (arm2_operator_split, arm3_modular,
                                                arm6_coupling, arm10_copasi)
from viva_sbml_decomposition.composites.reaction_split import (
    build_document, reaction_split, run_reaction_split)
from viva_sbml_decomposition.models.glycolysis import GLYCOLYSIS_ANTIMONY, GLYCOLYSIS_META

TOL = 1e-9


def max_abs_diff(a, b):
    (traj_a, t_a), (traj_b, t_b) = a, b
    assert set(GLYCOLYSIS_META["species"]) <= set(traj_a)
    assert t_a.shape == t_b.shape
    d = float(np.max(np.abs(t_a - t_b)))
    for s in GLYCOLYSIS_META["species"]:
        assert traj_a[s].shape == traj_b[s].shape
        d = max(d, float(np.max(np.abs(traj_a[s] - traj_b[s]))))
    return d


def test_aldolase_tellurium_matches_arm3():
    d = max_abs_diff(run_reaction_split("glycolysis", "aldolase", "tellurium", "tellurium",
                                        coupling_interval=arm3_modular.DEFAULT_COUPLING_INTERVAL),
                     arm3_modular.run_arm())
    print("A/B arm3 max|diff| =", d)
    assert d < TOL


def test_aldolase_tellurium_native_matches_arm6():
    d = max_abs_diff(run_reaction_split("glycolysis", "aldolase", "tellurium", "native",
                                        coupling_interval=arm6_coupling.DEFAULT_COUPLING_INTERVAL),
                     arm6_coupling.run_arm())
    print("A/B arm6 max|diff| =", d)
    assert d < TOL


def test_mechanical_tellurium_vs_arm2_closest_faithful():
    # NOT bit-identical by construction: arm 2 is ONE OperatorSplitProcess doing a
    # sequential (Lie) split with overwrite coupling; reaction_split is two nodes with
    # additive (Jacobi) coupling over the same partition. The two schemes share the
    # first-order splitting error order but not its constant, so they agree only to
    # O(coupling_interval). Measured max|diff|: dt=0.05 -> 0.662 (arm 2 default),
    # 0.01 -> 0.186, 0.005 -> 0.095, 0.001 -> 0.029 (first-order convergence).
    # Assert the honest bounds and the convergence trend (a wrong partition would not
    # converge).
    diffs = {}
    for dt in (arm2_operator_split.DEFAULT_COUPLING_INTERVAL, 0.005):
        diffs[dt] = max_abs_diff(
            run_reaction_split("glycolysis", "mechanical", "tellurium", "tellurium",
                               coupling_interval=dt),
            arm2_operator_split.run_arm(coupling_interval=dt))
    print("A/B arm2 max|diff| =", diffs)
    assert diffs[0.05] < 1.0
    assert diffs[0.005] < 0.15
    assert diffs[0.005] < diffs[0.05] / 3


@pytest.mark.slow
def test_aldolase_tellurium_copasi_matches_arm10_cross():
    # Both sides run in the abort-safe retrying subprocess. 1e-6, not 1e-9: COPASI is
    # not bit-reproducible run-to-run (~5e-8..1e-7 measured against itself).
    d = max_abs_diff(run_reaction_split("glycolysis", "aldolase", "tellurium", "copasi",
                                        coupling_interval=arm10_copasi.DEFAULT_COUPLING_INTERVAL),
                     arm10_copasi.run_cross_engine())
    print("A/B arm10-cross max|diff| =", d)
    assert d < 1e-6


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
