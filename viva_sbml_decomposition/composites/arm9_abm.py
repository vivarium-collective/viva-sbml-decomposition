"""Arm 9: a stochastic ABM + its mean-field SBML/Tellurium reduction.

The investigation's ADVERSARIAL arm. Arms 1-8 take a model that is *already* an
ODE network and ask how to decompose it. Arm 9 asks the prior question the naive
"any model is an SBML model" claim glosses over: **can an agent-based model even
BE a Tellurium/SBML model?** The honest answer, quantified here, is *only in one
limit*.

Two composites run the SAME Lotka-Volterra predator-prey system:

* :func:`predator_prey_abm` — the ABM: a well-mixed population of DISCRETE
  individuals whose births/predation/deaths are STOCHASTIC EVENTS, advanced by
  the exact Gillespie SSA in a native pbg
  :class:`~viva_sbml_decomposition.processes.predator_prey_abm.PredatorPreyABMProcess`.
  Integer-count state, one stochastic trajectory per run. **This is NOT an SBML
  model** — there is no continuous RHS for CVODE to integrate; the dynamics live
  in the jump process and its ensemble.
* :func:`predator_prey_meanfield` — the attempted reduction: the deterministic
  Lotka-Volterra ODE as Antimony, integrated by libroadrunner/CVODE in a
  :class:`TelluriumProcess`. This IS the SBML model — the mean-field,
  large-population, continuous-deterministic limit of the ABM. Its Antimony is
  carried on the generator's ``config.model`` so the Model page shows exactly
  what the reduction is.

The finding (see :func:`abm_ensemble` / the arm-9 study): at LARGE population the
SSA ensemble mean tracks the ODE to a few percent (reduction VALID); at SMALL
population the stochastic system goes EXTINCT in the great majority of runs while
the neutrally-stable ODE orbits forever (reduction INVALID). The ODE
structurally cannot represent variance, discreteness, or extinction probability.
"""
from __future__ import annotations

import numpy as np
from process_bigraph import allocate_core
from process_bigraph.composite_generator import composite_generator
from viva_tellurium.composites import make_tellurium_document, register_tellurium

from viva_sbml_decomposition.processes.predator_prey_abm import (
    A_PREDATION,
    B_PRED_BIRTH,
    LARGE_SCALE,
    M_PRED_DEATH,
    N_POINTS,
    R_PREY_BIRTH,
    SMALL_SCALE,
    T_END,
    X0_DENSITY,
    Y0_DENSITY,
    abm_ensemble,
    register_predator_prey_abm,
)

from ._emit import strip_emitter
from ._run import extract_run

MEANFIELD_SPECIES = ["Prey", "Pred"]


def build_lv_antimony(scale: int) -> str:
    """The deterministic Lotka-Volterra ODE as Antimony at population scale ``K``.

    Quadratic interaction rates carry the ``1/K`` system-size factor so the
    *density* ``X/K`` obeys the scale-free LV ODE at every ``K`` (the mean-field
    limit the ABM converges to as ``K -> inf``). Four reactions, two floating
    species — the libroadrunner/CVODE-integrable reduction of the ABM.
    """
    x0 = int(round(scale * X0_DENSITY))
    y0 = int(round(scale * Y0_DENSITY))
    return f"""model lotka_volterra_meanfield
  # Mean-field (large-population, deterministic) limit of the stochastic ABM.
  # Species are continuous population counts; K = {scale} is the system size.
  Prey = {x0};
  Pred = {y0};
  r = {R_PREY_BIRTH};   # prey birth (per capita)
  a = {A_PREDATION};   # predation
  b = {B_PRED_BIRTH};   # predator birth from predation
  m = {M_PRED_DEATH};   # predator death (per capita)
  K = {scale};

  prey_birth:     => Prey; r*Prey;
  predation:      Prey => ; (a/K)*Prey*Pred;
  predator_birth: => Pred; (b/K)*Prey*Pred;
  predator_death: Pred => ; m*Pred;
end"""


# The canonical Antimony shown on the Model page: the large-scale reduction (the
# regime where it is actually a faithful reduction of the ABM).
LV_ANTIMONY = build_lv_antimony(LARGE_SCALE)


def _interval(t_end: float, n_points: int) -> float:
    return t_end / n_points


def _build_meanfield_document(scale: int, t_end: float, n_points: int) -> dict:
    return make_tellurium_document(
        model=build_lv_antimony(scale),
        model_format="antimony",
        interval=_interval(t_end, n_points),
    )


def _build_abm_document(scale: int, n_points: int, seed: int) -> dict:
    """Native SSA ABM process + RAM emitter over the integer population array."""
    x0 = int(round(scale * X0_DENSITY))
    y0 = int(round(scale * Y0_DENSITY))
    return {
        "population": np.array([x0, y0], dtype=float),
        "abm": {
            "_type": "process",
            "address": "local:PredatorPreyABMProcess",
            "config": {"scale": scale, "seed": seed},
            "interval": _interval(T_END, n_points),
            "inputs": {"population": ["population"]},
            "outputs": {"population": ["population"]},
        },
        "emitter": {
            "_type": "step",
            "address": "local:ram-emitter",
            "config": {"emit": {"population": "array[(2),float]",
                                "time": "float", "global_time": "float"}},
            "inputs": {"population": ["population"],
                       "time": ["global_time"], "global_time": ["global_time"]},
        },
    }


@composite_generator(
    name="predator_prey_abm",
    description=(
        "Arm 9 (ABM): a well-mixed stochastic Lotka-Volterra predator-prey model "
        "as a population of DISCRETE individuals with stochastic birth/predation/"
        "death events, advanced by the exact Gillespie SSA in a native "
        "process-bigraph process. Integer-count state, one stochastic trajectory "
        "per run. This is the model that is NOT an SBML/ODE model — the adversarial "
        "case for the 'any model is an SBML model' claim."
    ),
    default_n_steps=N_POINTS,
    core_extensions=[register_predator_prey_abm],
    fidelity="real",
    methods=["Gillespie exact SSA (direct method)", "native Python/numpy"],
)
def predator_prey_abm(core=None, scale: int = LARGE_SCALE, seed: int = -1):
    """Stochastic LV ABM composite (emitter stripped for the workbench).

    ``seed=-1`` (the default) draws a fresh OS-entropy RNG per instance, so a
    workbench run is a NEW random realisation each time (appropriate for an
    interactive single-trajectory demo); pass a non-negative ``seed`` to make a
    composite run reproducible. The reproducible ensemble statistics that back
    the study finding come from :func:`abm_ensemble`, which is always seeded.
    """
    return strip_emitter(_build_abm_document(scale, N_POINTS, seed))


@composite_generator(
    name="predator_prey_meanfield",
    description=(
        "Arm 9 (mean-field reduction): the deterministic Lotka-Volterra ODE as "
        "Antimony integrated by libroadrunner/CVODE in a TelluriumProcess — the "
        "SBML reduction of the stochastic ABM. Faithful only in the large-"
        "population, well-mixed, deterministic limit; it structurally cannot "
        "represent the ABM's stochasticity, discreteness, or extinction."
    ),
    default_n_steps=N_POINTS,
    core_extensions=[register_tellurium],
    fidelity="real",
    methods=["Lotka-Volterra ODE (Antimony)", "libroadrunner / CVODE"],
)
def predator_prey_meanfield(core=None):
    """Mean-field LV ODE composite (Tellurium; emitter stripped for the workbench)."""
    return strip_emitter(_build_meanfield_document(LARGE_SCALE, T_END, N_POINTS))


def meanfield_trajectory(core=None, scale: int = LARGE_SCALE,
                         t_end: float = T_END, n_points: int = N_POINTS):
    """Run the mean-field LV ODE in libroadrunner/CVODE; return ``(traj, t)``.

    ``traj`` is ``{"Prey": ndarray, "Pred": ndarray}`` in population-count units
    at system size ``scale`` (divide by ``scale`` for densities), ``t`` the
    aligned time grid. Real libroadrunner integration through the pbg run path
    (same RAM-emitter readback as the other arms).
    """
    if core is None:
        core = allocate_core()
    register_tellurium(core)
    doc = _build_meanfield_document(scale, t_end, n_points)
    return extract_run(doc, core, t_end, MEANFIELD_SPECIES)


# Single-name interface for the ensemble helper (re-exported for symmetry with
# the other arms' run_arm / *_speedup entry points).
__all__ = [
    "predator_prey_abm", "predator_prey_meanfield", "meanfield_trajectory",
    "abm_ensemble", "build_lv_antimony", "LV_ANTIMONY", "MEANFIELD_SPECIES",
    "LARGE_SCALE", "SMALL_SCALE", "T_END", "N_POINTS",
]
