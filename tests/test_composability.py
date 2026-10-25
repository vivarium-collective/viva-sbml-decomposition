"""Composability scoring rubric tests (Task 10).

The composability axis of the 3-way tradeoff (accuracy / speed / composability)
is a *rubric judgment* per arm, not a measured number. These tests pin the
ordering the rubric encodes (monolith lowest; modular/native highest; arm 6 the
demonstrated maximum) and the accuracy-exemption flag arm 6 carries so the
Task-12 figure keeps it off the accuracy axis.

The rubric criteria themselves live (documented) in
:mod:`viva_sbml_decomposition.harness.composability`; these tests only assert the
direction and the exemption, never reverse-engineer the individual scores.
"""
from __future__ import annotations

from viva_sbml_decomposition.harness.composability import (
    ARM_IDS,
    ComposabilityScore,
    score_arm,
)


def test_composability_scores_monotonic():
    s1 = score_arm("arm1")  # monolithic
    s2 = score_arm("arm2")  # operator-split
    s3 = score_arm("arm3")  # modular submodels
    s4 = score_arm("arm4")  # native pbg
    s6 = score_arm("arm6")  # coupling demonstration

    assert isinstance(s1, ComposabilityScore)

    # Monolith scores LOWEST on substitutability + coupling_readiness: one opaque
    # SBML blob, nothing to swap or wire.
    assert s1.substitutability < s3.substitutability
    assert s1.substitutability < s4.substitutability
    assert s1.coupling_readiness < s3.coupling_readiness
    assert s1.coupling_readiness < s4.coupling_readiness

    # Operator-split is low-ish: above the monolith, below modular/native.
    assert s1.substitutability <= s2.substitutability < s3.substitutability
    assert s1.coupling_readiness <= s2.coupling_readiness < s3.coupling_readiness

    # Modular (arm3) and native (arm4) both score HIGH (named, swappable,
    # store-coupled nodes) — the two high poles of the spectrum.
    assert s3.substitutability == s4.substitutability
    assert s3.coupling_readiness == s4.coupling_readiness

    # Arm 6 demonstrates the MAXIMUM — it literally IS a substitution + coupling.
    assert s6.substitutability >= s3.substitutability
    assert s6.substitutability >= s4.substitutability
    assert s6.coupling_readiness >= s3.coupling_readiness
    assert s6.coupling_readiness >= s4.coupling_readiness
    # And strictly beyond the monolith on every axis.
    assert s6.substitutability > s1.substitutability
    assert s6.coupling_readiness > s1.coupling_readiness


def test_every_arm_scores_on_the_documented_scale():
    # Every arm id resolves to a score whose three fields are on the fixed
    # 0..3 (Low/Med/High/Max) integer scale documented in the rubric module.
    from viva_sbml_decomposition.harness.composability import MIN_LEVEL, MAX_LEVEL

    for aid in ARM_IDS:
        s = score_arm(aid)
        for field in (s.substitutability, s.coupling_readiness, s.multiscale_fit):
            assert MIN_LEVEL <= int(field) <= MAX_LEVEL
        assert s.rationale  # a documented justification string


def test_arm6_excluded_from_accuracy_axis():
    # The arm-6 record flags itself off the accuracy axis (accuracy None /
    # accuracy_exempt True) — the flag the Task-12 viz assembler will consume.
    s6 = score_arm("arm6")
    assert s6.accuracy_exempt is True
    assert s6.accuracy is None

    # Every other arm HAS a monolithic reference, so it is NOT exempt.
    for aid in ["arm1", "arm2", "arm3", "arm4", "arm5"]:
        assert score_arm(aid).accuracy_exempt is False
