import numpy as np
import tellurium as te

from viva_sbml_decomposition.composites.arm1_monolithic import run_arm
from viva_sbml_decomposition.harness.accuracy import compare_to_reference
from viva_sbml_decomposition.models.glycolysis import GLYCOLYSIS_ANTIMONY, GLYCOLYSIS_META


def test_arm1_emits_reference_trajectory():
    names = GLYCOLYSIS_META["species"]
    traj, t = run_arm()
    assert set(traj) == set(names) and len(names) == 17
    assert t[0] == 0.0 and np.isclose(t[-1], GLYCOLYSIS_META["t_end"])
    for s in names:
        assert traj[s].shape == t.shape and np.all(np.isfinite(traj[s]))

    r = te.loada(GLYCOLYSIS_ANTIMONY)
    r.integrator.absolute_tolerance = 1e-12
    r.integrator.relative_tolerance = 1e-8
    r.timeCourseSelections = ["time"] + [f"[{s}]" for s in names]
    res = np.asarray(r.simulate(0, GLYCOLYSIS_META["t_end"], 3001), dtype=float)
    direct = {s: np.interp(t, res[:, 0], res[:, i + 1]) for i, s in enumerate(names)}
    for s in names:
        scale = max(np.max(np.abs(direct[s])), 1e-9)
        assert np.max(np.abs(traj[s] - direct[s])) / scale < 1e-3, s
    out = compare_to_reference(direct, t, traj, t)
    assert out.status == "ok"
