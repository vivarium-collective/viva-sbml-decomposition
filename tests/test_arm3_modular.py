"""Arm 3: modular submodels across a biological seam — convergence to arm 1.

The SAME 17-species Teusink glycolysis model as arm 1, partitioned along the
textbook **aldolase seam** into two biologically-meaningful submodels:

* **upper glycolysis** — the preparatory / ATP-investment phase (glucose uptake
  through fructose-1,6-bisphosphate) plus the G6P storage sinks;
* **lower glycolysis** — the payoff phase (F16P -> pyruvate/ethanol, ATP- and
  NADH-producing) plus fermentation and the ATPase.

Each submodel is its own ``TelluriumProcess`` (one libroadrunner instance over
its reaction subset). They share the boundary metabolites — F16P, the
P/ATP/ADP/AMP energy pool, and NAD/NADH — through a common vivarium ``species``
store, reconciled **additively** (each submodel's per-substep delta is summed).

This seam cuts the strong phosphate/energy coupling that fed the sqrt ADP
assignment rule, which made it *intractable* for arm 2's pure Lie split (the
isolated sub-model left the sqrt domain and CVODE blew up for every interval).
Arm 3 makes the same seam converge by sub-stepping: each coupling interval is
advanced in ``n_substeps`` small additive steps, so neither submodel drifts the
shared energy pool out of the sqrt domain before reconciliation. The residual
is genuine first-order additive-splitting error that shrinks as the substep
shrinks (verified below); it is NOT manufactured.
"""
import numpy as np
import pytest

from viva_sbml_decomposition.composites.arm1_monolithic import run_arm as run_monolithic
from viva_sbml_decomposition.composites.arm3_modular import (
    glycolysis_modular,
    run_arm as run_modular,
)
from viva_sbml_decomposition.harness.accuracy import compare_to_reference
from viva_sbml_decomposition.models.glycolysis import GLYCOLYSIS_META
from viva_sbml_decomposition.models.glycolysis_submodels import SUBMODELS

# Shared coupling interval; the discriminating knob is the number of internal
# additive sub-steps per interval (smaller substep -> smaller splitting error).
COUPLING_INTERVAL = 0.1
COARSE_SUBSTEPS = 50    # dt = 2.0e-3  -> ~0.47 max rel err (outside T)
FINE_SUBSTEPS = 200     # dt = 5.0e-4  -> ~0.10 max rel err (within T)

# Provisional tolerance for the fine substep, calibrated in the task-7 report.
# The coarse substep sits OUTSIDE it so the bound discriminates. T is looser
# than arm 2's 0.10 because additive (Jacobi) coupling across the STRONG energy
# seam is fundamentally less accurate than arm 2's sequential split across the
# WEAK storage seam — that difference is the scientific point of this arm.
TOLERANCE_T = 0.15


def test_arm3_matches_reference_within_tolerance():
    ref_traj, ref_t = run_monolithic()

    coarse_traj, coarse_t = run_modular(
        coupling_interval=COUPLING_INTERVAL, n_substeps=COARSE_SUBSTEPS
    )
    fine_traj, fine_t = run_modular(
        coupling_interval=COUPLING_INTERVAL, n_substeps=FINE_SUBSTEPS
    )

    coarse = compare_to_reference(ref_traj, ref_t, coarse_traj, coarse_t)
    fine = compare_to_reference(ref_traj, ref_t, fine_traj, fine_t)

    # Both configurations integrate to finite trajectories over the full span
    # (the sub-stepping keeps the strong-coupling seam in the sqrt domain).
    assert coarse.status == "ok", "coarse modular split diverged"
    assert fine.status == "ok", "fine modular split diverged"

    # Accuracy IMPROVES as the substep shrinks — the real signature of
    # additive-splitting error vanishing with the substep.
    assert fine.max_rel_err < coarse.max_rel_err, (
        f"fine {fine.max_rel_err:.4e} not better than coarse "
        f"{coarse.max_rel_err:.4e}"
    )

    # The fine substep is within T; the coarse substep is outside it — so T
    # genuinely asserts something rather than passing vacuously.
    assert fine.max_rel_err < TOLERANCE_T, (
        f"fine {fine.max_rel_err:.4e} exceeds T={TOLERANCE_T}"
    )
    assert coarse.max_rel_err > TOLERANCE_T, (
        f"coarse {coarse.max_rel_err:.4e} within T={TOLERANCE_T}; "
        "T does not discriminate"
    )


def test_arm3_pairs_full_species_set_by_name():
    names = GLYCOLYSIS_META["species"]
    traj, t = run_modular(
        coupling_interval=COUPLING_INTERVAL, n_substeps=COARSE_SUBSTEPS
    )

    # The full 17-species set round-trips through the shared store by name.
    assert set(traj) == set(names) and len(names) == 17
    assert t[0] == 0.0 and np.isclose(t[-1], GLYCOLYSIS_META["t_end"])
    for s in names:
        assert traj[s].shape == t.shape and np.all(np.isfinite(traj[s]))

    # The accuracy harness pairs species by name and raises on any mismatch:
    # drop one species and confirm the comparison refuses to proceed.
    ref_traj, ref_t = run_monolithic()
    dropped = {k: v for k, v in traj.items() if k != "ATP"}
    with pytest.raises(ValueError):
        compare_to_reference(ref_traj, ref_t, dropped, t)


def test_arm3_partition_is_a_biological_seam():
    # Two biologically-named submodels at the aldolase seam, each its own
    # non-empty reaction group; together they partition the network.
    assert [name for name, _ in SUBMODELS] == ["upper_glycolysis", "lower_glycolysis"]
    assert len(SUBMODELS) >= 2
    for _, group in SUBMODELS:
        assert len(group) >= 1


def test_arm3_generator_strips_emitter():
    doc = glycolysis_modular()
    assert "submodels" in doc
    assert not any(k == "emitter" or k.startswith("emitter_") for k in doc)
