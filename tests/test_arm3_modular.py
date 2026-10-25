"""Arm 3: modular submodels as a GENUINE multi-process composite — convergence
to the arm-1 reference.

The SAME 17-species Teusink glycolysis model as arm 1, partitioned along the
textbook **aldolase seam** into two biologically-meaningful submodels, each its
own process-bigraph node:

* **upper glycolysis** — the preparatory / ATP-investment phase (glucose uptake
  through fructose-1,6-bisphosphate) plus the G6P storage sinks;
* **lower glycolysis** — the payoff phase (F16P -> pyruvate/ethanol, ATP- and
  NADH-producing) plus fermentation and the ATPase.

The two ``SubmodelProcess`` nodes share a single ``species`` store and couple
only through it: each reads the shared state, integrates its reactions over the
coupling interval, and writes its per-species **delta** back. Because the store's
``map[float]`` applies numeric updates additively and sums deltas written to the
same path, the two submodels reconcile **additively** through the bigraph apply
path (a Jacobi / additive operator split — the mosaic-compose dFBA-coculture
pattern). An ``EnergyStep`` node re-derives ATP/ADP/AMP from the reconciled pool.

This seam cuts the strong phosphate/energy coupling that fed the sqrt ADP
assignment rule, which made it *intractable* for arm 2's pure Lie split (the
isolated sub-model left the sqrt domain and CVODE blew up for every interval).
Arm 3 makes the same seam converge by reconciling through the store frequently
(small coupling interval), so neither submodel drifts the shared energy pool out
of the sqrt domain before the other's delta lands. The residual is genuine
first-order additive-splitting error that shrinks with the coupling interval
(verified below); it is NOT manufactured.
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

# The discriminating knob is the coupling interval = the scheduler step at which
# both submodel nodes reconcile their deltas through the shared store. Smaller
# interval -> smaller additive-splitting error.
COARSE_INTERVAL = 0.002   # ~0.47 max rel err (outside T)
FINE_INTERVAL = 0.0005    # ~0.10 max rel err (within T)
CHEAP_INTERVAL = 0.005    # domain-safe, fast; for the round-trip / finiteness check
DIVERGE_INTERVAL = 0.1    # one whole interval in isolation -> leaves the sqrt domain

# Provisional tolerance for the fine interval, calibrated in the task-7 report.
# Looser than arm 2's 0.10 because additive (Jacobi) coupling across the STRONG
# energy seam is fundamentally less accurate than arm 2's sequential split across
# the WEAK storage seam — that difference is the scientific point of this arm.
TOLERANCE_T = 0.15


def test_arm3_matches_reference_within_tolerance():
    ref_traj, ref_t = run_monolithic()

    coarse_traj, coarse_t = run_modular(coupling_interval=COARSE_INTERVAL)
    fine_traj, fine_t = run_modular(coupling_interval=FINE_INTERVAL)

    coarse = compare_to_reference(ref_traj, ref_t, coarse_traj, coarse_t)
    fine = compare_to_reference(ref_traj, ref_t, fine_traj, fine_t)

    # Both configurations integrate to finite trajectories over the full span
    # (frequent store reconciliation keeps the strong-coupling seam in domain).
    assert coarse.status == "ok", "coarse modular composite diverged"
    assert fine.status == "ok", "fine modular composite diverged"

    # Accuracy IMPROVES as the coupling interval shrinks — the real signature of
    # additive-splitting error vanishing with the reconciliation interval.
    assert fine.max_rel_err < coarse.max_rel_err, (
        f"fine {fine.max_rel_err:.4e} not better than coarse "
        f"{coarse.max_rel_err:.4e}"
    )

    # The fine interval is within T; the coarse interval is outside it — so T
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
    traj, t = run_modular(coupling_interval=CHEAP_INTERVAL)

    # The full 17-species set round-trips through the shared store by name — the
    # 14 settable species carried by the submodel nodes plus ATP/ADP/AMP derived
    # by the EnergyStep node.
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


def test_arm3_diverges_cleanly_out_of_domain():
    # A whole coupling interval in isolation drives the shared phosphate pool out
    # of the sqrt ADP-rule domain; CVODE fails. The SubmodelProcess guard converts
    # that into a non-finite delta, so the composite runs to completion and the
    # arm surfaces as *diverged* (status + inf) rather than crashing.
    ref_traj, ref_t = run_monolithic()
    traj, t = run_modular(coupling_interval=DIVERGE_INTERVAL)
    result = compare_to_reference(ref_traj, ref_t, traj, t)
    assert result.status == "diverged"
    assert result.diverged and not np.isfinite(result.max_rel_err)


def test_arm3_partition_is_a_biological_seam():
    # Two biologically-named submodels at the aldolase seam, each its own
    # non-empty reaction group; together they partition the network.
    assert [name for name, _ in SUBMODELS] == ["upper_glycolysis", "lower_glycolysis"]
    assert len(SUBMODELS) >= 2
    for _, group in SUBMODELS:
        assert len(group) >= 1


def test_arm3_generator_is_multiprocess_through_a_shared_store():
    doc = glycolysis_modular()

    # A shared species store and SEPARATE process nodes per submodel — the
    # multi-process decomposition, not one process doing internal reconciliation.
    assert "species" in doc and isinstance(doc["species"], dict)
    assert doc["upper_glycolysis"]["_type"] == "process"
    assert doc["lower_glycolysis"]["_type"] == "process"
    # Both submodels wire their species port to the one shared store.
    assert doc["upper_glycolysis"]["outputs"]["species"] == ["species"]
    assert doc["lower_glycolysis"]["outputs"]["species"] == ["species"]
    # Emitter stripped for the workbench.
    assert not any(k == "emitter" or k.startswith("emitter_") for k in doc)
