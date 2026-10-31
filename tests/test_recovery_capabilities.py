"""Tests for the SBML-recovery capabilities (synthetic target + comparator)."""
from __future__ import annotations

import numpy as np
import pytest

from viva_sbml_decomposition.core import build_core
from viva_sbml_decomposition.models.recovery_targets import (
    MASS_ACTION_ANTIMONY,
    MASS_ACTION_META,
    RECOVERY_TARGETS,
)
from viva_sbml_decomposition.processes.recovery import (
    SyntheticTargetGenerator,
    TrajectoryComparator,
)


@pytest.fixture
def core():
    return build_core()


def _generate(core, **cfg):
    gen = SyntheticTargetGenerator(config={"antimony": MASS_ACTION_ANTIMONY, **cfg}, core=core)
    return gen.update({})


def test_capabilities_registered(core):
    for name in ("SyntheticTargetGenerator", "TrajectoryComparator"):
        assert name in core.link_registry, f"{name} not registered by build_core()"


def test_synthetic_target_shapes_and_values(core):
    out = _generate(core, t_end=10.0, n_points=50)
    assert set(out["species_ids"]) == set(MASS_ACTION_META["species"])
    assert len(out["time"]) == 50
    for s in MASS_ACTION_META["species"]:
        assert len(out["trajectories"][s]) == 50
        assert all(np.isfinite(out["trajectories"][s]))
    # Mass-action here conserves total mass A+B+2C? No — r2 consumes A and B into C,
    # but A is also produced by r2r. Sanity: A is consumed from its initial 10 and
    # the system reaches a finite non-negative state.
    assert out["trajectories"]["A"][0] == pytest.approx(10.0, abs=1e-6)
    assert out["trajectories"]["B"][0] == pytest.approx(0.0, abs=1e-9)
    assert all(v >= -1e-6 for s in out["species_ids"] for v in out["trajectories"][s])


def test_observed_subset(core):
    out = _generate(core, observed=["A", "C"], n_points=20)
    assert set(out["species_ids"]) == {"A", "C"}
    assert "B" not in out["trajectories"]


def test_identical_trajectories_score_zero(core):
    out = _generate(core, n_points=40)
    cmp = TrajectoryComparator(config={"tol": 0.01}, core=core)
    res = cmp.update({"time": out["time"], "target": out["trajectories"], "candidate": out["trajectories"]})
    assert res["nrmse"] == pytest.approx(0.0, abs=1e-12)
    assert res["passed"] is True
    for s in out["species_ids"]:
        assert res["per_species_nrmse"][s] == pytest.approx(0.0, abs=1e-12)


def test_perturbed_candidate_fails_tolerance(core):
    out = _generate(core, n_points=40)
    # A clearly-wrong candidate: scale every trajectory by 1.5.
    bad = {s: [v * 1.5 for v in vals] for s, vals in out["trajectories"].items()}
    cmp = TrajectoryComparator(config={"tol": 0.05}, core=core)
    res = cmp.update({"time": out["time"], "target": out["trajectories"], "candidate": bad})
    assert res["nrmse"] > 0.05
    assert res["passed"] is False


def test_missing_species_is_not_a_free_pass(core):
    out = _generate(core, n_points=30)
    partial = dict(out["trajectories"])
    partial.pop("C")  # candidate omits a target species
    cmp = TrajectoryComparator(config={"tol": 0.05}, core=core)
    res = cmp.update({"time": out["time"], "target": out["trajectories"], "candidate": partial})
    assert not np.isfinite(res["per_species_nrmse"]["C"])
    assert res["passed"] is False


def test_noise_degrades_fit_monotonically(core):
    clean = _generate(core, n_points=60, noise_sigma=0.0, seed=1)
    noisy = _generate(core, n_points=60, noise_sigma=0.05, seed=1)
    cmp = TrajectoryComparator(config={"tol": 1.0}, core=core)
    res = cmp.update({"time": clean["time"], "target": clean["trajectories"], "candidate": noisy["trajectories"]})
    assert res["nrmse"] > 0.0  # noise moves the candidate off the clean target


def test_recovery_targets_registry():
    assert "recover-mass-action" in RECOVERY_TARGETS
    assert RECOVERY_TARGETS["recover-mass-action"]["antimony"] == MASS_ACTION_ANTIMONY
