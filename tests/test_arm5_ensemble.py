"""Arm 5 — ensemble on a real parallel (Ray) backend.

HONESTY GATE (Review Focus): on a multi-core host with Ray, the measured
wall-clock speedup must be real (parallel < serial via actual OS processes).
If Ray is unavailable/unusable or yields no win, the result MUST record
``backend == "serial"`` and ``speedup_claimed is False`` and must NOT report a
speedup — a serial fallback silently claiming a speedup is the failure mode
these tests exist to prevent.
"""
import math
import os

import numpy as np
import pytest

from viva_sbml_decomposition.composites.arm5_ensemble import (
    ensemble_speedup,
    glycolysis_ensemble,
)
from viva_sbml_decomposition.models.glycolysis import GLYCOLYSIS_META

SPECIES = GLYCOLYSIS_META["species"]


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


def test_ensemble_runs_all_instances():
    n = 4
    results = glycolysis_ensemble(n_instances=n)
    assert len(results) == n
    assert results.backend in ("ray", "serial")
    for res in results:
        assert set(res) == set(SPECIES) and len(SPECIES) == 17
        assert all(math.isfinite(float(v)) for v in res.values())


def test_ensemble_param_draws_perturb_instances():
    # Two draws that scale a rate constant differently must give different states.
    # Mild +/-10% perturbations of hexokinase Vmax: integrable and distinct.
    draws = [
        {"vGLK_VmGLK": 226.452 * 1.10},
        {"vGLK_VmGLK": 226.452 * 0.90},
    ]
    results = glycolysis_ensemble(n_instances=2, param_draws=draws)
    assert len(results) == 2
    a = np.array([results[0][s] for s in SPECIES])
    b = np.array([results[1][s] for s in SPECIES])
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    assert not np.allclose(a, b)


def test_ensemble_parallel_or_declares_serial():
    """HONESTY GATE. Either Ray delivers a real wall-clock speedup, or the
    result honestly declares a serial fallback with no speedup claim."""
    result = ensemble_speedup(n_instances=8)

    # On a multicore host where Ray imports, a Ray regression must not pass
    # silently via the serial-declared branch.
    try:
        import ray  # noqa: F401
        have_ray = True
    except ImportError:
        have_ray = False
    if (os.cpu_count() or 1) >= 4 and have_ray:
        if not result.speedup_claimed:  # one retry against timing noise
            result = ensemble_speedup(n_instances=8)
        assert result.speedup_claimed is True, result
        assert result.setup_excluded is True and result.setup_s is not None

    # The honesty invariant, whichever branch we land in.
    if result.speedup_claimed:
        # A claimed speedup must be a REAL Ray parallel win.
        assert result.backend == "ray"
        assert result.parallel_s is not None and result.serial_s is not None
        assert result.parallel_s < result.serial_s
        assert result.speedup is not None and result.speedup > 1.0
        assert math.isclose(result.speedup, result.serial_s / result.parallel_s, rel_tol=1e-6)
    else:
        # Serial fallback: declare it and report NO speedup.
        assert result.backend == "serial"
        assert result.speedup is None
