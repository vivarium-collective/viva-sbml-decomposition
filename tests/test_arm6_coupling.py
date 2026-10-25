"""Arm 6: the COUPLING DEMONSTRATION — a composition the monolith can't express.

Arm 6 takes arm 3's modular decomposition and swaps ONE submodel node (lower
glycolysis) for a fully NATIVE process-bigraph process computing that submodel's
reactions, while the OTHER submodel (upper glycolysis) stays a
Tellurium/libroadrunner :class:`SubmodelProcess`. Both share the same ``species``
store and reconcile additively through the bigraph apply path. You cannot mix a
hand-written native process and an SBML submodel inside one libroadrunner model —
so this is a real composition the monolithic arm 1 **structurally** cannot
represent. That impossibility is structural, NOT numeric: arm 6 solves the same
ODEs as arm 1 and reproduces arm 3 (the matching additive-split decomposition) to
~1e-7 at the same coupling interval.

Arm 6 is kept OFF the accuracy axis because it is the composability existence
proof, not an accuracy result — it has no dedicated monolithic benchmark. These
tests assert that it (1) builds, runs, and emits finite, sensible output for the
full species set; (2) is a genuine cross-engine composition (both a Tellurium and
a native node, distinct, on the shared store — not arm 3 relabeled); (3) rests on
a complete, disjoint reaction partition whose two halves' native RHS sum to the
full network RHS; and (4) reproduces the additive-split reference (arm 3) at the
same interval, which would catch a broken partition.
"""
from __future__ import annotations

import numpy as np
import pytest

from viva_sbml_decomposition.composites import arm3_modular, arm6_coupling
from viva_sbml_decomposition.harness.accuracy import compare_to_reference
from viva_sbml_decomposition.harness.composability import score_arm
from viva_sbml_decomposition.models.glycolysis import GLYCOLYSIS_META
from viva_sbml_decomposition.models.glycolysis_submodels import (
    LOWER_GLYCOLYSIS,
    UPPER_GLYCOLYSIS,
)
from viva_sbml_decomposition.processes import native_glycolysis as ng

# arm 6 and arm 3 are compared at the SAME coupling interval so the (common)
# additive-splitting error cancels and the comparison isolates the engine swap.
INTERVAL = arm6_coupling.DEFAULT_COUPLING_INTERVAL

# arm 6 reproduces arm 3 to ~1e-7 (the native lower half matches the Tellurium
# lower half, both solving the same ODEs). A loose 1e-3 bound passes that with a
# wide margin yet still fails hard on a broken reaction partition (which gives an
# order-1 error or a diverged run).
ARM6_VS_ARM3_TOL = 1e-3


@pytest.fixture(scope="module")
def arm6_run():
    """Run arm 6 once (the heavy native+Tellurium hybrid) and share it."""
    return arm6_coupling.run_arm(coupling_interval=INTERVAL)


def test_arm6_runs_a_coupling_the_monolith_cannot(arm6_run):
    names = GLYCOLYSIS_META["species"]
    traj, t = arm6_run

    # The full 17-species set round-trips through the shared store by name.
    assert set(traj) == set(names) and len(names) == 17
    assert t[0] == 0.0 and np.isclose(t[-1], GLYCOLYSIS_META["t_end"])

    # Finite + sensible: every species is finite, bounded, and non-negative to a
    # small tolerance over the whole run.
    for s in names:
        v = traj[s]
        assert v.shape == t.shape
        assert np.all(np.isfinite(v)), f"{s} produced non-finite values"
        assert np.all(v > -1e-6), f"{s} went substantially negative"
        assert np.all(v < 1e4), f"{s} blew up (unbounded)"

    # Arm 6 is accuracy-exempt (composability existence proof, no benchmark).
    s6 = score_arm("arm6")
    assert s6.accuracy is None and s6.accuracy_exempt is True


def test_arm6_is_a_genuine_cross_engine_composition():
    # Prove it is NOT arm 3 relabeled: the Tellurium node (local:SubmodelProcess)
    # and the native node (local:NativeSubmodelProcess) must BOTH be present as
    # DISTINCT nodes. Exact addresses — ".endswith('SubmodelProcess')" would be
    # satisfied by the native node alone and would miss a dropped Tellurium half.
    doc = arm6_coupling.glycolysis_coupled()
    addresses = {
        k: v["address"]
        for k, v in doc.items()
        if isinstance(v, dict) and "address" in v
    }
    tellurium_nodes = [k for k, a in addresses.items() if a == "local:SubmodelProcess"]
    native_nodes = [k for k, a in addresses.items() if a == "local:NativeSubmodelProcess"]
    assert len(tellurium_nodes) == 1, f"expected one Tellurium node: {addresses}"
    assert len(native_nodes) == 1, f"expected one native node: {addresses}"
    assert tellurium_nodes[0] != native_nodes[0]

    # Both submodel nodes share the single species store (couple only through it).
    proc_nodes = [
        v for v in doc.values()
        if isinstance(v, dict) and v.get("_type") == "process"
    ]
    assert len(proc_nodes) == 2
    for node in proc_nodes:
        assert node["inputs"]["species"] == ["species"]
        assert node["outputs"]["species"] == ["species"]

    # A single shared species store seeded with the full species set.
    assert "species" in doc and isinstance(doc["species"], dict)
    assert set(doc["species"]) == set(GLYCOLYSIS_META["species"])

    # Emitter stripped for the workbench generator.
    assert not any(k == "emitter" or k.startswith("emitter_") for k in doc)


def test_arm6_reaction_partition_is_complete_and_disjoint():
    # The upper/lower split must be a true partition of the network: together they
    # cover every reaction, and they share none (a dropped or duplicated reaction
    # would give finite-but-wrong output that the run test alone cannot catch).
    upper = set(UPPER_GLYCOLYSIS)
    lower = set(LOWER_GLYCOLYSIS)
    assert upper.isdisjoint(lower), f"overlap: {upper & lower}"
    assert upper | lower == set(ng.STOICH), (
        f"partition does not cover the network: missing "
        f"{set(ng.STOICH) - (upper | lower)}, extra {(upper | lower) - set(ng.STOICH)}"
    )


def test_arm6_native_submodel_halves_sum_to_the_full_rhs():
    # Structural integrity of the native subset machinery: the upper-subset RHS
    # plus the lower-subset RHS must equal the full native network RHS at any
    # state. This pins the partition to the real kinetics (a dropped reaction
    # under-counts, a duplicated reaction double-counts) independent of any run.
    upper_rhs = ng.make_subset_rhs(UPPER_GLYCOLYSIS)
    lower_rhs = ng.make_subset_rhs(LOWER_GLYCOLYSIS)
    rng = np.random.default_rng(0)
    states = [
        np.array([ng._P[s] for s in ng.SETTABLE], dtype=float),      # initial
        np.abs(rng.normal(1.0, 0.3, len(ng.SETTABLE))),              # perturbed
    ]
    for y in states:
        full = ng._rhs(0.0, y)
        halves = upper_rhs(0.0, y) + lower_rhs(0.0, y)
        np.testing.assert_allclose(halves, full, rtol=0, atol=1e-9)


def test_arm6_reproduces_the_additive_split_reference(arm6_run):
    # arm 6 solves the SAME ODEs as arm 3 with the lower half on a different
    # engine; at the same coupling interval the common additive-splitting error
    # cancels and the two trajectories must agree to ~1e-7. A broken reaction
    # partition would blow this up (order-1 error or divergence) — so this is the
    # end-to-end correctness check the finiteness test cannot provide. It does NOT
    # put arm 6 on the accuracy axis: the impossibility vs the monolith is
    # structural, and arm 6 carries no dedicated accuracy benchmark.
    a6_traj, a6_t = arm6_run
    a3_traj, a3_t = arm3_modular.run_arm(coupling_interval=INTERVAL)

    result = compare_to_reference(a3_traj, a3_t, a6_traj, a6_t)
    assert not result.diverged, "arm 6 diverged against the additive-split reference"
    assert result.max_rel_err < ARM6_VS_ARM3_TOL, (
        f"arm6 vs arm3 max_rel_err {result.max_rel_err:.3e} >= {ARM6_VS_ARM3_TOL:.1e} "
        "— the reaction partition or the native subset RHS is likely broken"
    )
