"""Tests for the SBML-recovery capabilities (synthetic target + comparator)."""
from __future__ import annotations

import numpy as np
import pytest

from viva_sbml_decomposition.core import build_core
from viva_sbml_decomposition.models.recovery_targets import (
    MASS_ACTION_ANTIMONY,
    MASS_ACTION_META,
    RECOVERY_TARGETS,
)
from viva_sbml_decomposition.composites.engine_monolithic import run_engine_monolithic
from viva_sbml_decomposition.composites.recovery import recover_candidate, recover_mass_action
from viva_sbml_decomposition.processes.recovery import (
    RateLawFitter,
    ReactionProposer,
    StructuralScorer,
    SyntheticTargetGenerator,
    TrajectoryComparator,
)

# Mass-action structure with WRONG rate constants — the build side must fit these
# back to the truth (k1=0.5, k2=0.3, k2r=0.1) from the target data.
CAND_BADPARAMS = """
model cand_badparams()
  species A, B, C;
  A = 10.0; B = 0.0; C = 0.0;
  k1 = 1.0; k2 = 1.0; k2r = 1.0;
  r1: A -> B; k1*A;
  r2: A + B -> C; k2*A*B;
  r2r: C -> A + B; k2r*C;
end
""".strip()

# Candidate variants of the mass-action oracle for structural scoring.
CAND_MISSING_R2R = """
model cand_missing()
  species A, B, C;
  A = 10.0; B = 0.0; C = 0.0;
  k1 = 0.5; k2 = 0.3;
  r1: A -> B; k1*A;
  r2: A + B -> C; k2*A*B;
end
""".strip()

CAND_WRONG_PARAM = MASS_ACTION_ANTIMONY.replace("k1 = 0.5;", "k1 = 0.9;")

CAND_SPURIOUS = """
model cand_spurious()
  species A, B, C;
  A = 10.0; B = 0.0; C = 0.0;
  k1 = 0.5; k2 = 0.3; k2r = 0.1; k3 = 0.05;
  r1: A -> B; k1*A;
  r2: A + B -> C; k2*A*B;
  r2r: C -> A + B; k2r*C;
  r3: B -> C; k3*B;
end
""".strip()


@pytest.fixture
def core():
    return build_core()


def _generate(core, **cfg):
    gen = SyntheticTargetGenerator(config={"antimony": MASS_ACTION_ANTIMONY, **cfg}, core=core)
    return gen.update({})


def test_capabilities_registered(core):
    for name in ("SyntheticTargetGenerator", "TrajectoryComparator"):
        assert name in core.link_registry, f"{name} not registered by build_core()"


def test_synthetic_target_shapes_and_values(core):
    out = _generate(core, t_end=10.0, n_points=50)
    assert set(out["species_ids"]) == set(MASS_ACTION_META["species"])
    assert len(out["time"]) == 50
    for s in MASS_ACTION_META["species"]:
        assert len(out["trajectories"][s]) == 50
        assert all(np.isfinite(out["trajectories"][s]))
    # Mass-action here conserves total mass A+B+2C? No — r2 consumes A and B into C,
    # but A is also produced by r2r. Sanity: A is consumed from its initial 10 and
    # the system reaches a finite non-negative state.
    assert out["trajectories"]["A"][0] == pytest.approx(10.0, abs=1e-6)
    assert out["trajectories"]["B"][0] == pytest.approx(0.0, abs=1e-9)
    assert all(v >= -1e-6 for s in out["species_ids"] for v in out["trajectories"][s])


def test_observed_subset(core):
    out = _generate(core, observed=["A", "C"], n_points=20)
    assert set(out["species_ids"]) == {"A", "C"}
    assert "B" not in out["trajectories"]


def test_identical_trajectories_score_zero(core):
    out = _generate(core, n_points=40)
    cmp = TrajectoryComparator(config={"tol": 0.01}, core=core)
    res = cmp.update({"time": out["time"], "target": out["trajectories"], "candidate": out["trajectories"]})
    assert res["nrmse"] == pytest.approx(0.0, abs=1e-12)
    assert res["passed"] is True
    for s in out["species_ids"]:
        assert res["per_species_nrmse"][s] == pytest.approx(0.0, abs=1e-12)


def test_perturbed_candidate_fails_tolerance(core):
    out = _generate(core, n_points=40)
    # A clearly-wrong candidate: scale every trajectory by 1.5.
    bad = {s: [v * 1.5 for v in vals] for s, vals in out["trajectories"].items()}
    cmp = TrajectoryComparator(config={"tol": 0.05}, core=core)
    res = cmp.update({"time": out["time"], "target": out["trajectories"], "candidate": bad})
    assert res["nrmse"] > 0.05
    assert res["passed"] is False


def test_missing_species_is_not_a_free_pass(core):
    out = _generate(core, n_points=30)
    partial = dict(out["trajectories"])
    partial.pop("C")  # candidate omits a target species
    cmp = TrajectoryComparator(config={"tol": 0.05}, core=core)
    res = cmp.update({"time": out["time"], "target": out["trajectories"], "candidate": partial})
    assert not np.isfinite(res["per_species_nrmse"]["C"])
    assert res["passed"] is False


def test_noise_degrades_fit_monotonically(core):
    clean = _generate(core, n_points=60, noise_sigma=0.0, seed=1)
    noisy = _generate(core, n_points=60, noise_sigma=0.05, seed=1)
    cmp = TrajectoryComparator(config={"tol": 1.0}, core=core)
    res = cmp.update({"time": clean["time"], "target": clean["trajectories"], "candidate": noisy["trajectories"]})
    assert res["nrmse"] > 0.0  # noise moves the candidate off the clean target


def test_recovery_targets_registry():
    assert "recover-mass-action" in RECOVERY_TARGETS
    assert RECOVERY_TARGETS["recover-mass-action"]["antimony"] == MASS_ACTION_ANTIMONY


# --- StructuralScorer -------------------------------------------------------

def _score(core, candidate, truth=MASS_ACTION_ANTIMONY, **cfg):
    scorer = StructuralScorer(config=cfg, core=core)
    return scorer.update({"candidate_antimony": candidate, "truth_antimony": truth})


def test_structural_scorer_registered(core):
    assert "StructuralScorer" in core.link_registry


def test_structural_identity_fully_recovered(core):
    res = _score(core, MASS_ACTION_ANTIMONY)
    assert res["reactions_recovered"] == pytest.approx(1.0)
    assert res["spurious_count"] == 0
    assert res["param_max_rel_err"] == pytest.approx(0.0, abs=1e-9)
    assert res["structure_recovered"] is True
    assert res["parsimonious"] is True


def test_structural_missing_reaction_not_recovered(core):
    res = _score(core, CAND_MISSING_R2R)
    assert res["reactions_recovered"] < 1.0  # 2 of 3 truth reactions
    assert res["structure_recovered"] is False
    assert res["parsimonious"] is True  # fewer reactions, still parsimonious


def test_structural_wrong_parameter_flagged(core):
    res = _score(core, CAND_WRONG_PARAM, param_tol=0.1)
    # k1 0.5 -> 0.9 is an 80% error on a matched reaction.
    assert res["param_max_rel_err"] > 0.1
    assert res["structure_recovered"] is False


def test_structural_spurious_reaction_not_parsimonious(core):
    res = _score(core, CAND_SPURIOUS)
    assert res["spurious_count"] >= 1
    assert res["parsimonious"] is False
    assert res["structure_recovered"] is False  # require_no_spurious default


# --- recover_mass_action composite + end-to-end loop ------------------------

def test_recover_mass_action_composite_builds():
    doc = recover_mass_action()
    assert isinstance(doc, dict) and doc, "composite should return a non-empty document"


def test_recovery_loop_identity_end_to_end(core):
    # Oracle target (hidden ground truth) on a 101-point grid.
    gen = SyntheticTargetGenerator(
        config={"antimony": MASS_ACTION_ANTIMONY, "t_end": 10.0, "n_points": 101,
                "observed": ["A", "B", "C"]},
        core=core,
    )
    target = gen.update({})
    # Candidate run via the recover_mass_action composite path (identity: candidate == truth).
    traj, t = run_engine_monolithic(MASS_ACTION_ANTIMONY, "tellurium", n_steps=100, t_end=10.0)
    candidate = {s: [float(v) for v in traj[s]] for s in ["A", "B", "C"]}
    # Trajectory comparison: identity recovers the target to tight tolerance.
    cmp = TrajectoryComparator(config={"tol": 0.02, "observed": ["A", "B", "C"]}, core=core)
    traj_res = cmp.update({"time": target["time"], "target": target["trajectories"], "candidate": candidate})
    assert traj_res["passed"] is True, traj_res
    assert traj_res["nrmse"] < 0.02
    # Structural recovery: candidate IS the truth -> fully recovered.
    struct = _score(core, MASS_ACTION_ANTIMONY)
    assert struct["structure_recovered"] is True
    assert struct["parsimonious"] is True


# --- the full oracle ladder + generic composite -----------------------------

def test_all_four_oracles_simulate():
    import tellurium as te
    from viva_sbml_decomposition.models.recovery_targets import RECOVERY_TARGETS
    assert {"recover-mass-action", "recover-enzyme-kinetics", "recover-gene-switch",
            "recover-oscillator"} <= set(RECOVERY_TARGETS)
    for slug, entry in RECOVERY_TARGETS.items():
        r = te.loada(entry["antimony"]); r.reset()
        res = r.simulate(0, entry["meta"]["t_end"], 50)
        assert res.shape[0] == 50, slug


def test_recover_candidate_composite_for_each_target():
    from viva_sbml_decomposition.models.recovery_targets import RECOVERY_TARGETS
    for slug in RECOVERY_TARGETS:
        doc = recover_candidate(target=slug)
        assert isinstance(doc, dict) and doc, slug


# --- build side: RateLawFitter + ReactionProposer ---------------------------

def test_rate_law_fitter_recovers_mass_action_params(core):
    out = RateLawFitter(
        config={"candidate_antimony": CAND_BADPARAMS, "target": "recover-mass-action", "max_iter": 800},
        core=core,
    ).update({})
    assert out["nrmse"] < 0.03, out["nrmse"]
    fp = out["fitted_params"]
    assert fp["k1"] == pytest.approx(0.5, rel=0.2), fp
    assert fp["k2"] == pytest.approx(0.3, rel=0.25), fp
    assert fp["k2r"] == pytest.approx(0.1, rel=0.4), fp


def test_fit_then_structure_recovers_non_identity(core):
    # Non-identity recovery: wrong params -> fit -> structurally recovered.
    fitted = RateLawFitter(
        config={"candidate_antimony": CAND_BADPARAMS, "target": "recover-mass-action", "max_iter": 800},
        core=core,
    ).update({})["fitted_antimony"]
    sc = StructuralScorer(config={"param_tol": 0.25}, core=core).update(
        {"candidate_antimony": fitted, "truth_antimony": MASS_ACTION_ANTIMONY}
    )
    assert sc["structure_recovered"] is True, sc


def test_reaction_proposer_produces_valid_candidate(core):
    import tellurium as te
    out = ReactionProposer(config={"species": ["A", "B", "C"], "initial": {"A": 10.0}}, core=core).update({})
    assert out["n_reactions"] == 6  # 3*2 unimolecular conversions
    r = te.loada(out["candidate_antimony"]); r.reset()
    res = r.simulate(0, 5, 10)
    assert res.shape[0] == 10


def test_build_side_registered(core):
    for name in ("RateLawFitter", "ReactionProposer"):
        assert name in core.link_registry


# --- gradient-matching closes the oscillator gap ----------------------------

def _wrong_2x(antimony):
    import tellurium as te
    rr = te.loada(antimony)
    for p in rr.getGlobalParameterIds():
        rr[p] = float(rr[p]) * 2.0
    return te.sbmlToAntimony(rr.getCurrentSBML())


def test_oscillator_recovered_via_gradient_matching(core):
    from viva_sbml_decomposition.models.recovery_targets import OSCILLATOR_ANTIMONY
    out = RateLawFitter(
        config={"candidate_antimony": _wrong_2x(OSCILLATOR_ANTIMONY), "target": "recover-oscillator"},
        core=core,
    ).update({})
    assert out["nrmse"] < 0.05, out["nrmse"]
    assert out["fitted_params"]["A"] == pytest.approx(1.0, abs=0.05), out["fitted_params"]
    assert out["fitted_params"]["B"] == pytest.approx(3.0, abs=0.1), out["fitted_params"]


def test_trajectory_only_cannot_recover_oscillator(core):
    # Documents WHY gradient matching is needed: raw-trajectory LSQ is ill-conditioned
    # on a limit cycle and fails from a 2x-wrong start.
    from viva_sbml_decomposition.models.recovery_targets import OSCILLATOR_ANTIMONY
    out = RateLawFitter(
        config={"candidate_antimony": _wrong_2x(OSCILLATOR_ANTIMONY),
                "target": "recover-oscillator", "method": "trajectory"},
        core=core,
    ).update({})
    assert out["nrmse"] > 0.1, out["nrmse"]


def test_auto_recovers_every_rung(core):
    from viva_sbml_decomposition.models.recovery_targets import RECOVERY_TARGETS
    for slug, entry in RECOVERY_TARGETS.items():
        if entry["meta"].get("stochastic"):
            continue  # adversarial — not a recovery rung
        out = RateLawFitter(
            config={"candidate_antimony": _wrong_2x(entry["antimony"]), "target": slug},
            core=core,
        ).update({})
        assert out["nrmse"] < 0.05, (slug, out["nrmse"])


# --- deepened proposer (SINDy) + non-recoverability detector ----------------

def test_sindy_recovers_mass_action_structure_from_data(core):
    out = ReactionProposer(config={"mode": "from_data", "target": "recover-mass-action"}, core=core).update({})
    assert out["n_reactions"] == 3  # A->B, A+B->C, C->A+B — no spurious
    sc = StructuralScorer(config={"param_tol": 0.1}, core=core).update(
        {"candidate_antimony": out["candidate_antimony"],
         "truth_antimony": __import__("viva_sbml_decomposition.models.recovery_targets",
                                      fromlist=["MASS_ACTION_ANTIMONY"]).MASS_ACTION_ANTIMONY})
    assert sc["structure_recovered"] is True
    assert sc["reactions_recovered"] == pytest.approx(1.0)


def test_non_recoverability_detector(core):
    from viva_sbml_decomposition.processes.recovery import NonRecoverabilityDetector
    # Deterministic flat-SBML systems read recoverable.
    for slug in ("recover-mass-action", "recover-oscillator"):
        out = NonRecoverabilityDetector(config={"target": slug}, core=core).update({})
        assert out["non_recoverable"] is False, (slug, out)
        assert out["min_r2"] > 0.9
    # A single stochastic (low-copy) trajectory is NOT recoverable as deterministic flat SBML.
    adv = NonRecoverabilityDetector(config={"target": "recover-stochastic-abm"}, core=core).update({})
    assert adv["non_recoverable"] is True
    assert adv["min_r2"] < 0.9
