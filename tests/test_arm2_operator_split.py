"""Arm 2: operator-split Tellurium — convergence toward the arm-1 reference.

The SAME glycolysis model as arm 1, advanced as a Lie operator split: the
reaction network is partitioned into two complementary groups (operator A =
the core glycolytic + fermentation network; operator B = the glycogen /
trehalose storage sinks branching off G6P). Each vivarium step advances
operator A over ``coupling_interval``, hands its state to operator B through
the shared ``species`` store, then advances B over the same interval. The
splitting error is real and first-order in ``coupling_interval``; it must
shrink as the interval shrinks.
"""
import numpy as np

from viva_sbml_decomposition.composites.arm1_monolithic import run_arm as run_monolithic
from viva_sbml_decomposition.composites.arm2_operator_split import (
    glycolysis_operator_split,
    run_arm as run_operator_split,
)
from viva_sbml_decomposition.harness.accuracy import compare_to_reference
from viva_sbml_decomposition.models.glycolysis import GLYCOLYSIS_META

# Two coupling intervals, both in the operator split's stable regime.
COARSE_INTERVAL = 0.1
FINE_INTERVAL = 0.0125

# Provisional tolerance for the fine interval, calibrated here (see task-6
# report). The coarse interval sits OUTSIDE it so the bound discriminates.
TOLERANCE_T = 0.10


def test_arm2_converges_to_reference_as_interval_shrinks():
    ref_traj, ref_t = run_monolithic()

    coarse_traj, coarse_t = run_operator_split(coupling_interval=COARSE_INTERVAL)
    fine_traj, fine_t = run_operator_split(coupling_interval=FINE_INTERVAL)

    coarse = compare_to_reference(ref_traj, ref_t, coarse_traj, coarse_t)
    fine = compare_to_reference(ref_traj, ref_t, fine_traj, fine_t)

    # Both arms integrate to finite trajectories over the full span.
    assert coarse.status == "ok", "coarse operator split diverged"
    assert fine.status == "ok", "fine operator split diverged"

    # Accuracy IMPROVES as the coupling interval shrinks — the real signature
    # of operator-splitting error vanishing with the interval.
    assert fine.max_rel_err < coarse.max_rel_err, (
        f"fine {fine.max_rel_err:.4e} not better than coarse "
        f"{coarse.max_rel_err:.4e}"
    )

    # The fine interval is within the provisional tolerance T, and the coarse
    # interval is outside it — so T genuinely asserts something.
    assert fine.max_rel_err < TOLERANCE_T, (
        f"fine {fine.max_rel_err:.4e} exceeds T={TOLERANCE_T}"
    )
    assert coarse.max_rel_err > TOLERANCE_T, (
        f"coarse {coarse.max_rel_err:.4e} within T={TOLERANCE_T}; "
        "T does not discriminate"
    )


def test_arm2_run_emits_full_species_set():
    names = GLYCOLYSIS_META["species"]
    traj, t = run_operator_split(coupling_interval=COARSE_INTERVAL)
    assert set(traj) == set(names) and len(names) == 17
    assert t[0] == 0.0 and np.isclose(t[-1], GLYCOLYSIS_META["t_end"])
    for s in names:
        assert traj[s].shape == t.shape and np.all(np.isfinite(traj[s]))


def test_arm2_generator_strips_emitter():
    doc = glycolysis_operator_split()
    assert "operator_split" in doc
    assert not any(k == "emitter" or k.startswith("emitter_") for k in doc)
