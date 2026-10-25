"""Arm 4 (fully native pbg glycolysis) accuracy + leaf-equivalence tests.

Arm 4 reimplements the Teusink glycolysis ODE RHS as native Python/numpy inside
a process-bigraph process (no libroadrunner at runtime) and integrates it with
scipy LSODA. Two composite generators differ only in the species leaf
representation (``map[float]`` vs one numpy ``array``). These tests assert:

1. Each variant, run through process-bigraph, reproduces the arm-1 libroadrunner
   reference within the provisional tolerance ``T`` (species paired by name,
   full 17-species set round-tripping through the RAM emitter).
2. The two variants produce the same trajectory as each other (they are the
   same math) to within a tight tolerance.

``T`` is provisional and recorded here. The observed max relative error of the
native port against the reference is ~1e-7 (machine-level agreement with
libroadrunner at the arm-1 integrator tolerances); ``T = 1e-5`` leaves a modest
margin while still asserting genuine agreement — it is NOT set loose enough to
pass a wrong RHS.
"""
from __future__ import annotations

import numpy as np
import pytest

from viva_sbml_decomposition.composites import arm4_native
from viva_sbml_decomposition.composites.arm1_monolithic import run_arm as run_monolithic
from viva_sbml_decomposition.harness.accuracy import compare_to_reference
from viva_sbml_decomposition.models.glycolysis import GLYCOLYSIS_META

# Provisional accuracy tolerance (max relative error vs the arm-1 reference).
T = 1e-5
# The two leaf variants run byte-for-byte identical math; require tight agreement.
VARIANT_AGREEMENT_TOL = 1e-9


@pytest.fixture(scope="module")
def reference():
    return run_monolithic()


@pytest.mark.parametrize("leaf", ["scalar", "array"])
def test_arm4_matches_reference_within_tolerance(reference, leaf):
    ref_traj, ref_t = reference
    arm_traj, arm_t = arm4_native.run_arm(leaf=leaf)

    # Full species set round-trips, paired by name.
    assert set(arm_traj) == set(GLYCOLYSIS_META["species"])
    assert set(arm_traj) == set(ref_traj)

    result = compare_to_reference(ref_traj, ref_t, arm_traj, arm_t)
    assert not result.diverged, f"{leaf} variant diverged"
    assert result.max_rel_err < T, (
        f"{leaf} variant max_rel_err {result.max_rel_err:.3e} >= T {T:.1e}"
    )


def test_arm4_scalar_and_array_variants_agree():
    scalar_traj, scalar_t = arm4_native.run_arm(leaf="scalar")
    array_traj, array_t = arm4_native.run_arm(leaf="array")

    assert set(scalar_traj) == set(array_traj)
    np.testing.assert_allclose(scalar_t, array_t)

    for name in scalar_traj:
        a = np.asarray(scalar_traj[name], dtype=float)
        b = np.asarray(array_traj[name], dtype=float)
        scale = max(float(np.max(np.abs(a))), 1e-12)
        max_rel = float(np.max(np.abs(a - b))) / scale
        assert max_rel < VARIANT_AGREEMENT_TOL, (
            f"species {name}: scalar vs array max_rel {max_rel:.3e} "
            f">= {VARIANT_AGREEMENT_TOL:.1e}"
        )
