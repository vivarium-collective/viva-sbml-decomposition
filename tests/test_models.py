import math

import numpy as np
import tellurium as te

from viva_sbml_decomposition.models.glycolysis import GLYCOLYSIS_ANTIMONY, GLYCOLYSIS_META


def test_glycolysis_integrates_monolithic():
    r = te.loada(GLYCOLYSIS_ANTIMONY)
    r.integrator.absolute_tolerance = 1e-12
    r.integrator.relative_tolerance = 1e-8
    names = GLYCOLYSIS_META["species"]
    r.timeCourseSelections = ["time"] + [f"[{s}]" for s in names]
    res = r.simulate(0, GLYCOLYSIS_META["t_end"], 200)
    assert list(res.colnames) == ["time"] + [f"[{s}]" for s in names]
    final = np.asarray(res[-1, 1:], dtype=float)
    assert np.all(np.isfinite(final))
    assert np.all(final >= 0)
    assert math.isclose(res[-1, 0], GLYCOLYSIS_META["t_end"])
