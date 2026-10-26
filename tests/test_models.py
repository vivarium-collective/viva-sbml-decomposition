import math

import numpy as np
import pytest
import tellurium as te

from viva_sbml_decomposition.models.glycolysis import GLYCOLYSIS_ANTIMONY, GLYCOLYSIS_META

# Converged values (t=1000, tight tolerances) from a clean run of the reference model.
STEADY_STATE = {
    "F16P": 0.601907640,
    "ATP": 2.50839062,
    "P": 6.30888164,
    "PYR": 8.52315246,
    "NADH": 0.0444402330,
}


def _simulate():
    r = te.loada(GLYCOLYSIS_ANTIMONY)
    r.integrator.absolute_tolerance = 1e-12
    r.integrator.relative_tolerance = 1e-8
    names = GLYCOLYSIS_META["species"]
    r.timeCourseSelections = ["time"] + [f"[{s}]" for s in names]
    return r, np.asarray(r.simulate(0, GLYCOLYSIS_META["t_end"], 200), dtype=float)


def test_glycolysis_integrates_monolithic():
    r, res = _simulate()
    names = GLYCOLYSIS_META["species"]
    assert set(names) == set(r.getFloatingSpeciesIds())
    assert len(r.getBoundarySpeciesIds()) == 9
    assert np.all(np.isfinite(res))
    assert np.all(res[:, 1:] >= -1e-6)
    assert math.isclose(res[-1, 0], GLYCOLYSIS_META["t_end"])


def test_glycolysis_reaches_pinned_steady_state():
    _, res = _simulate()
    names = GLYCOLYSIS_META["species"]
    np.testing.assert_allclose(res[-1, 1:], res[-2, 1:], rtol=1e-6, atol=1e-9)
    for sp, val in STEADY_STATE.items():
        assert res[-1, 1 + names.index(sp)] == pytest.approx(val, rel=1e-4)
