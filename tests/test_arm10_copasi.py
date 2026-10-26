"""Arm 10: COPASI as a SECOND SBML engine + a cross-engine coupling composite.

Two things are under test:

1. **COPASI-mono** (``glycolysis_copasi_mono``) — the whole Teusink model run by
   COPASI/basico (LSODA) through process-bigraph, validated against the arm-1
   libroadrunner/CVODE reference. Two different integrators agree to a small but
   non-zero residual (measured max-relative-error ~1.7e-6), so the accuracy gate
   ``MONO_TOL`` sits well above that and tracks a genuine engine swap, not
   integrator noise.

2. **Cross-engine coupling** (``glycolysis_cross_engine``) — the arm-3
   aldolase-seam decomposition with upper glycolysis integrated by a
   Tellurium/libroadrunner ``SubmodelProcess`` and lower glycolysis by the COPASI
   process, BOTH sharing one ``species`` store and reconciling additive deltas
   each coupling interval (arm-6's cross-engine demonstration with COPASI swapped
   in for the native engine). Two genuinely different SBML engines interoperate
   through the pbg typed store.

Tolerances (chosen + recorded here):

* ``MONO_TOL = 1e-3`` — COPASI-mono vs. the libroadrunner reference. Measured
  ~1.7e-6; ~3 orders of headroom.
* ``ENGINE_SWAP_TOL = 1e-3`` — cross-engine (COPASI lower) vs. arm 3
  (Tellurium lower) at the *same* coupling interval. This is the clean
  engine-swap proof: with the engine the only thing changed, the two agree to
  machine-noise levels (measured ~1e-12), so this gate is a wide bound that
  still fails hard if the COPASI node is wired wrongly. It isolates the engine
  substitution from the (shared) operator-split error.
* ``CROSS_TOL = 0.5`` — cross-engine vs. the reference at the fine interval.
  The aldolase seam cuts the strong phosphate/energy coupling, so additive
  (Jacobi) splitting across it carries a large first-order error that shrinks
  with the interval — this is a property of the SEAM, identical to arm 3
  (``arm3`` ships ``TOLERANCE_T=0.15`` only at its 0.0005 interval), NOT a COPASI
  defect. Convergence is asserted two ways: the fine run beats the coarse run
  (monotone trend toward the reference) AND the fine run is within ``CROSS_TOL``
  while the coarse run is not (a gate that would fail a non-converging arm).
  Measured: coarse(0.005) ~1.41, fine(0.001) ~0.21.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from viva_sbml_decomposition.composites import arm3_modular
from viva_sbml_decomposition.composites.arm1_monolithic import run_arm as run_monolithic
from viva_sbml_decomposition.harness.accuracy import compare_to_reference
from viva_sbml_decomposition.models.glycolysis import GLYCOLYSIS_META

MONO_TOL = 1e-3
ENGINE_SWAP_TOL = 1e-3
CROSS_TOL = 0.5

COARSE_INTERVAL = 0.005   # ~1.41 max rel err (outside CROSS_TOL): domain-safe, fast
FINE_INTERVAL = 0.001     # ~0.21 max rel err (within CROSS_TOL): the convergence point


@pytest.fixture(scope="module")
def runs():
    """Compute the heavy arm-10 runs once and share them across the module.

    The cross-engine fine run (0.001 = 30k steps) is the slow one, so everything
    is run once here instead of per-test.
    """
    from viva_sbml_decomposition.composites.arm10_copasi import (
        run_copasi_mono,
        run_cross_engine,
    )

    ref_traj, ref_t = run_monolithic(n_steps=300)
    mono = run_copasi_mono(n_steps=300)
    cross_coarse = run_cross_engine(coupling_interval=COARSE_INTERVAL)
    cross_fine = run_cross_engine(coupling_interval=FINE_INTERVAL)
    arm3_coarse = arm3_modular.run_arm(coupling_interval=COARSE_INTERVAL)
    return SimpleNamespace(
        ref=(ref_traj, ref_t),
        mono=mono,
        cross_coarse=cross_coarse,
        cross_fine=cross_fine,
        arm3_coarse=arm3_coarse,
    )


# --------------------------------------------------------------------------- #
# COPASI-mono: second engine on the whole model, validated vs. libroadrunner.  #
# --------------------------------------------------------------------------- #

def test_copasi_mono_runs_through_process_bigraph(runs):
    traj, t = runs.mono
    assert set(traj) == set(GLYCOLYSIS_META["species"])
    assert t[0] == pytest.approx(0.0)
    assert t[-1] == pytest.approx(GLYCOLYSIS_META["t_end"], rel=1e-6)
    for s, arr in traj.items():
        assert arr.shape == t.shape
        assert np.all(np.isfinite(arr)), f"{s} non-finite"


def test_copasi_mono_matches_libroadrunner_reference(runs):
    ref_traj, ref_t = runs.ref
    traj, t = runs.mono
    res = compare_to_reference(ref_traj, ref_t, traj, t)
    assert not res.diverged
    assert res.max_rel_err < MONO_TOL, (
        f"COPASI-mono max_rel_err={res.max_rel_err:.3e} exceeds {MONO_TOL:.0e}"
    )


def test_copasi_mono_generator_builds_stripped_doc():
    from viva_sbml_decomposition.composites.arm10_copasi import glycolysis_copasi_mono

    doc = glycolysis_copasi_mono()
    assert "emitter" not in doc
    assert any(node.get("address") == "local:CopasiProcess"
               for node in doc.values() if isinstance(node, dict))


# --------------------------------------------------------------------------- #
# Cross-engine coupling: Tellurium + COPASI sharing one store.                 #
# --------------------------------------------------------------------------- #

def test_cross_engine_builds_and_runs_through_process_bigraph(runs):
    traj, t = runs.cross_coarse
    assert set(traj) == set(GLYCOLYSIS_META["species"])
    assert np.isclose(t[-1], GLYCOLYSIS_META["t_end"])
    for s, arr in traj.items():
        assert arr.shape == t.shape
        assert np.all(np.isfinite(arr)), f"{s} non-finite"
        assert np.all(arr > -1e-6), f"{s} went substantially negative"


def test_cross_engine_reproduces_arm3_at_same_interval(runs):
    """COPASI-lower reproduces Tellurium-lower at the same interval (clean swap).

    The cross-engine composite and arm 3 differ in ONE thing — the lower-half
    engine (COPASI vs. Tellurium) — so at a matched coupling interval they must
    agree to machine-noise levels. This isolates the engine substitution from the
    (shared) aldolase-seam splitting error and would fail hard on a mis-wired
    COPASI node.
    """
    cross_traj, cross_t = runs.cross_coarse
    arm3_traj, arm3_t = runs.arm3_coarse
    res = compare_to_reference(arm3_traj, arm3_t, cross_traj, cross_t)
    assert not res.diverged
    assert res.max_rel_err < ENGINE_SWAP_TOL, (
        f"COPASI-lower vs Tellurium-lower (same interval) max_rel_err="
        f"{res.max_rel_err:.3e} exceeds {ENGINE_SWAP_TOL:.0e} — the engine swap "
        f"is not clean"
    )


def test_cross_engine_converges_to_reference(runs):
    """Two different engines sharing one store converge to the reference.

    Monotone trend (fine beats coarse) AND the fine run is within CROSS_TOL while
    the coarse run is not — a gate that genuinely asserts convergence rather than
    passing vacuously.
    """
    ref_traj, ref_t = runs.ref
    coarse = compare_to_reference(ref_traj, ref_t, *runs.cross_coarse)
    fine = compare_to_reference(ref_traj, ref_t, *runs.cross_fine)
    assert not fine.diverged and not coarse.diverged
    assert fine.max_rel_err < coarse.max_rel_err, (
        f"fine {fine.max_rel_err:.3e} not better than coarse {coarse.max_rel_err:.3e}"
    )
    assert fine.max_rel_err < CROSS_TOL, (
        f"fine cross-engine max_rel_err={fine.max_rel_err:.3e} exceeds {CROSS_TOL}"
    )
    assert coarse.max_rel_err > CROSS_TOL, (
        f"coarse {coarse.max_rel_err:.3e} already within {CROSS_TOL}; "
        f"the convergence gate is vacuous"
    )


def test_cross_engine_uses_two_distinct_engines():
    from viva_sbml_decomposition.composites.arm10_copasi import glycolysis_cross_engine

    doc = glycolysis_cross_engine()
    addresses = {k: v["address"] for k, v in doc.items()
                 if isinstance(v, dict) and "address" in v}
    tell = [k for k, a in addresses.items() if a == "local:SubmodelProcess"]
    cop = [k for k, a in addresses.items() if a == "local:CopasiProcess"]
    assert len(tell) == 1, f"expected one Tellurium node: {addresses}"
    assert len(cop) == 1, f"expected one COPASI node: {addresses}"
    assert tell[0] != cop[0]
    # Both share the single species store seeded with the full species set.
    assert set(doc["species"]) == set(GLYCOLYSIS_META["species"])
    for node in (doc[tell[0]], doc[cop[0]]):
        assert node["inputs"]["species"] == ["species"]
        assert node["outputs"]["species"] == ["species"]
    assert "emitter" not in doc


# --------------------------------------------------------------------------- #
# Measurement helper.                                                          #
# --------------------------------------------------------------------------- #

def test_measurement_helper_reports_accuracy_and_timing():
    """measure_copasi() returns the mono accuracy/timing and a finite cross number."""
    from viva_sbml_decomposition.composites.arm10_copasi import measure_copasi

    m = measure_copasi(n_steps=300, coupling_interval=COARSE_INTERVAL, repeats=1)
    assert not m.mono_diverged
    assert m.mono_max_rel_err < MONO_TOL
    assert m.mono_time_s > 0.0
    assert not m.cross_diverged
    assert np.isfinite(m.cross_max_rel_err)
