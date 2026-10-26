"""Arm 9 (adversarial): stochastic ABM vs its mean-field SBML/Tellurium reduction.

Arm 9 is the case built to BREAK the naive "any model is an SBML model" claim.
A well-mixed stochastic Lotka-Volterra ABM (integer counts, exact Gillespie SSA)
is compared to its deterministic mean-field ODE run in libroadrunner/CVODE. The
tests assert the quantified feasibility finding:

* the ABM runs as a genuine pbg process over INTEGER populations, and predator
  collapse (extinction) is reachable at small population;
* the mean-field ODE runs finite and NEVER goes extinct (it orbits forever);
* at a LARGE population scale the SSA ensemble mean matches the ODE within the
  recorded tolerance ``T_LARGE`` with no extinction (reduction VALID);
* at a SMALL population scale the reduction BREAKS — a large fraction of runs see
  the predators collapse (and a substantial fraction go totally extinct) while
  the ODE persists, and the ensemble mean diverges far more than at large scale
  (reduction INVALID).

Extinction is measured honestly as two fractions: only PREY extinction is
absorbing, while predator-only extinction leaves the prey to grow (bounded by a
documented cap), so the ABM is NOT frozen on predator extinction.
``predator_extinct_fraction`` (the LV collapse the ODE can never show) is the
headline; ``total_extinct_fraction`` (both species zero) is reported alongside.

Tolerances are recorded here. Measured values (seed 0): LARGE K=400, 80 runs ->
max rel err 0.049, predator/total extinction 0.00; SMALL K=8, 200 runs ->
predator_extinct 0.865, total_extinct 0.510, mean diverges several-fold. The SSA
is seed-controlled so these are reproducible; thresholds leave a wide margin so
the assertions are not flaky, while still failing if the ABM did NOT converge to
the ODE at large N or did NOT break at small N.
"""
from __future__ import annotations

import numpy as np
from process_bigraph import Composite, allocate_core

from viva_sbml_decomposition.composites import arm9_abm
from viva_sbml_decomposition.composites.arm9_abm import (
    LARGE_SCALE,
    MEANFIELD_SPECIES,
    SMALL_SCALE,
    abm_ensemble,
    meanfield_trajectory,
)
from viva_sbml_decomposition.processes.predator_prey_abm import (
    register_predator_prey_abm,
)

# Recorded tolerances (max relative error of the SSA ensemble mean vs the ODE,
# measured on the same density grid). LARGE: measured 0.049, assert < 0.12.
T_LARGE = 0.12
# SMALL: a non-trivial share of runs must see the predators collapse (measured
# predator_extinct 0.865, total_extinct 0.510) — assert the reduction is broken.
SMALL_PRED_EXTINCT_MIN = 0.4
SMALL_TOTAL_EXTINCT_MIN = 0.2
# Canonical ensemble sizes (match the committed figure dataset).
LARGE_N = 80
SMALL_N = 200


def _rel_err_mean_vs_ode(res, ode_prey, ode_pred):
    """Max relative error of a density ensemble mean against the ODE density."""
    md = res.mean_density  # (2, n_points+1)
    denom = max(float(ode_prey.max()), float(ode_pred.max()), 1e-9)
    stacked = np.vstack([ode_prey, ode_pred])
    return float(np.max(np.abs(md - stacked)) / denom)


def test_abm_runs_as_process_with_integer_populations():
    """The ABM runs through process-bigraph and the population leaf is integer-valued."""
    core = allocate_core()
    register_predator_prey_abm(core)
    doc = arm9_abm._build_abm_document(scale=LARGE_SCALE, n_points=arm9_abm.N_POINTS, seed=1)
    comp = Composite({"state": doc}, core=core)
    comp.run(arm9_abm.T_END)
    rows = comp.state["emitter"]["instance"].history
    assert rows, "ABM emitted no history"
    pops = np.array([r["population"] for r in rows], dtype=float)
    assert pops.shape[1] == 2
    # Discrete individuals: every emitted count is an integer.
    assert np.array_equal(pops, np.round(pops)), "ABM populations are not integers"
    assert (pops >= 0).all(), "negative population count"


def test_abm_extinction_possible_at_small_population():
    """Predator collapse + total extinction (states the ODE can never reach) occur at small N."""
    res = abm_ensemble(SMALL_N, SMALL_SCALE, seed=0)
    assert res.predator_extinct_fraction > SMALL_PRED_EXTINCT_MIN, (
        f"predator_extinct_fraction {res.predator_extinct_fraction} too low at K={SMALL_SCALE}"
    )
    assert res.total_extinct_fraction > SMALL_TOTAL_EXTINCT_MIN, (
        f"total_extinct_fraction {res.total_extinct_fraction} too low at K={SMALL_SCALE}"
    )
    # total extinction implies prey extinction, which implies predators also die,
    # so total <= predator (prey-driven absorbing path is a subset of collapse).
    assert res.total_extinct_fraction <= res.predator_extinct_fraction


def test_predator_only_extinction_is_not_frozen():
    """When predators die but prey survive, prey keep growing (not frozen) — bounded by the cap."""
    from viva_sbml_decomposition.processes.predator_prey_abm import (
        POP_CAP_DENSITY, simulate_abm)
    # Find a run where predators go extinct but prey do not, and assert the prey
    # grew past their start (pure birth continued) rather than being held frozen.
    found = False
    for s in range(60):
        rng = np.random.default_rng(5000 + s)
        _, counts, status = simulate_abm(SMALL_SCALE, rng=rng)
        if status["predator_extinct"] and not status["prey_extinct"]:
            found = True
            prey = counts[0]
            assert prey[-1] > prey[0], "prey did not grow after predator extinction (frozen?)"
            assert prey.max() <= np.ceil(POP_CAP_DENSITY * SMALL_SCALE)
            break
    assert found, "no predator-only-extinction run found to check (increase search)"


def test_meanfield_ode_runs_finite_and_never_extinct():
    """The mean-field SBML reduction integrates finite in libroadrunner/CVODE and persists."""
    traj, t = meanfield_trajectory(scale=LARGE_SCALE)
    assert set(traj) == set(MEANFIELD_SPECIES)
    for name in MEANFIELD_SPECIES:
        y = np.asarray(traj[name], dtype=float)
        assert np.all(np.isfinite(y)), f"{name} not finite"
        # Neutrally-stable LV orbit: strictly positive, never hits the absorbing 0.
        assert y.min() > 0.0, f"{name} reached extinction — the ODE should orbit forever"
    assert len(t) == arm9_abm.N_POINTS + 1


def test_large_population_abm_mean_matches_ode():
    """Reduction VALID at large N: SSA ensemble mean tracks the ODE within tolerance, no extinction."""
    otraj, _ = meanfield_trajectory(scale=LARGE_SCALE)
    ode_prey = np.asarray(otraj["Prey"]) / LARGE_SCALE
    ode_pred = np.asarray(otraj["Pred"]) / LARGE_SCALE
    res = abm_ensemble(LARGE_N, LARGE_SCALE, seed=0)
    assert res.predator_extinct_fraction == 0.0, "unexpected predator extinction at large population"
    assert res.total_extinct_fraction == 0.0, "unexpected total extinction at large population"
    max_rel = _rel_err_mean_vs_ode(res, ode_prey, ode_pred)
    assert max_rel < T_LARGE, f"large-N max rel err {max_rel:.3f} >= T_LARGE {T_LARGE}"


def test_small_population_breaks_the_reduction():
    """Reduction INVALID at small N: ABM goes extinct while the ODE persists, mean diverges far more."""
    otraj, _ = meanfield_trajectory(scale=SMALL_SCALE)
    ode_prey = np.asarray(otraj["Prey"]) / SMALL_SCALE
    ode_pred = np.asarray(otraj["Pred"]) / SMALL_SCALE
    # The ODE itself never goes extinct, at any scale.
    assert float(np.asarray(otraj["Prey"]).min()) > 0.0
    assert float(np.asarray(otraj["Pred"]).min()) > 0.0

    small = abm_ensemble(SMALL_N, SMALL_SCALE, seed=0)
    assert small.predator_extinct_fraction > SMALL_PRED_EXTINCT_MIN
    assert small.total_extinct_fraction > SMALL_TOTAL_EXTINCT_MIN
    small_rel = _rel_err_mean_vs_ode(small, ode_prey, ode_pred)

    # And the small-N divergence is dramatically worse than the large-N agreement.
    otraj_l, _ = meanfield_trajectory(scale=LARGE_SCALE)
    large = abm_ensemble(LARGE_N, LARGE_SCALE, seed=0)
    large_rel = _rel_err_mean_vs_ode(
        large, np.asarray(otraj_l["Prey"]) / LARGE_SCALE,
        np.asarray(otraj_l["Pred"]) / LARGE_SCALE)
    assert small_rel > 3 * large_rel, (
        f"small-N rel err {small_rel:.3f} not dramatically worse than large-N {large_rel:.3f}"
    )


def test_abm_ensemble_is_seed_controlled():
    """Same seed -> identical ensemble statistics (reproducible, non-flaky)."""
    a = abm_ensemble(40, SMALL_SCALE, seed=7)
    b = abm_ensemble(40, SMALL_SCALE, seed=7)
    assert np.array_equal(a.mean, b.mean)
    assert a.predator_extinct_fraction == b.predator_extinct_fraction
    assert a.total_extinct_fraction == b.total_extinct_fraction


def test_meanfield_antimony_on_model_page():
    """The mean-field generator carries the LV Antimony on config.model (the Model tab)."""
    doc = arm9_abm.predator_prey_meanfield()
    tp = doc["tellurium"]
    model = tp["config"]["model"]
    assert "Prey" in model and "Pred" in model
    assert "predation" in model and "predator_birth" in model


def test_figure_renders_html():
    """The ABMMeanField figure renders to interactive Plotly html."""
    from viva_sbml_decomposition.visualizations.abm_meanfield import ABMMeanField
    inst = ABMMeanField(config={}, core=allocate_core())
    out = inst.update(ABMMeanField.generate_state())
    assert set(out) == {"html"}
    assert "<div" in out["html"] and "Plotly.newPlot" in out["html"]
    assert "viva_ABMMeanField" in out["html"]
