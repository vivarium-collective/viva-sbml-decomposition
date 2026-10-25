import numpy as np
import pytest

from viva_sbml_decomposition.harness.accuracy import compare_to_reference


def _ref():
    t = np.linspace(0, 10, 101)
    return {"A": np.exp(-t) + 1.0, "B": 2.0 - np.exp(-t)}, t


def test_identical_trajectories_zero_error():
    traj, t = _ref()
    r = compare_to_reference(traj, t, dict(traj), t)
    assert all(v == 0 for v in r.per_species_rmse.values())
    assert r.max_rel_err == 0
    assert not r.diverged and r.status == "ok"


def test_resamples_onto_common_grid():
    traj, t = _ref()
    t2 = np.linspace(0, 10, 11)
    arm = {k: np.interp(t2, t, v) for k, v in traj.items()}
    r = compare_to_reference(traj, t, arm, t2)
    assert np.isfinite(r.max_rel_err)
    assert all(np.isfinite(v) for v in r.per_species_rmse.values())


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_marks_diverged(bad):
    traj, t = _ref()
    arm = {k: v.copy() for k, v in traj.items()}
    arm["A"][5] = bad
    r = compare_to_reference(traj, t, arm, t)
    assert r.diverged and r.status == "diverged"
    assert not np.isnan(r.max_rel_err)


def test_species_paired_by_name_not_position():
    traj, t = _ref()
    arm = {"B": traj["B"], "A": traj["A"]}
    r = compare_to_reference(traj, t, arm, t)
    assert r.per_species_rmse == {"A": 0.0, "B": 0.0}
    with pytest.raises(ValueError):
        compare_to_reference(traj, t, {"A": traj["A"]}, t)
    with pytest.raises(ValueError):
        compare_to_reference(traj, t, {**traj, "C": traj["A"]}, t)


def test_nonoverlapping_spans_raise():
    traj, t = _ref()
    with pytest.raises(ValueError):
        compare_to_reference(traj, t, dict(traj), t + 100)
