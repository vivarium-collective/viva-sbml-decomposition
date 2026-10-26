"""Arm 9: a well-mixed stochastic Lotka-Volterra ABM as a native pbg Process.

This is the investigation's ADVERSARIAL arm — the case built to BREAK the naive
"any model can be written as an SBML/Tellurium model" claim. Arms 1-8 all run a
deterministic continuous ODE network (glycolysis) that libroadrunner integrates
natively. Arm 9 is a different *kind* of model: a population of DISCRETE
INDIVIDUALS whose interactions are STOCHASTIC EVENTS. There is no ODE to hand to
CVODE; the state is integer counts and the dynamics are a continuous-time Markov
jump process advanced by the Gillespie direct method (exact SSA).

The model is the classic Lotka-Volterra predator-prey system expressed as four
elementary reactions over integer counts (prey ``X``, predator ``Y``):

    prey birth        : X -> X + 1     propensity  r * X
    predation         : X -> X - 1     propensity  (a/K) * X * Y
    predator birth    : Y -> Y + 1     propensity  (b/K) * X * Y
    predator death    : Y -> Y - 1     propensity  m * Y

``K`` is the **population scale** (system size). Writing the quadratic
interaction propensities with the ``1/K`` factor is the standard system-size
scaling: the *density* ``x = X/K``, ``y = Y/K`` obeys the SAME scale-free
Lotka-Volterra ODE at every ``K``

    dx/dt = r*x - a*x*y        dy/dt = b*x*y - m*y

so the deterministic mean-field limit is fixed while ``K`` sets how many discrete
individuals realise it. That is exactly the knob that makes the reduction valid
or invalid: as ``K -> inf`` the SSA ensemble mean converges to the ODE (the van
Kampen / Kurtz large-population limit); at small ``K`` demographic stochasticity
and discreteness dominate and the stochastic system can go EXTINCT — an absorbing
state the neutrally-stable ODE orbits can never reach.

The SSA core (:func:`simulate_abm`, :func:`abm_ensemble`) is a plain, seeded
numpy loop so the ensemble statistics are reproducible and fast; the pbg
:class:`PredatorPreyABMProcess` wraps the identical per-interval stepper so the
ABM also runs as a genuine process-bigraph process (integer-count leaves, one
stochastic trajectory per run).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from process_bigraph import Process, allocate_core
from process_bigraph.emitter import RAMEmitter

# --- Lotka-Volterra base rates (scale-free density dynamics) -----------------
# Fixed point of the density ODE is (x*, y*) = (m/b, r/a) = (1, 1); the natural
# period near it is 2*pi/sqrt(r*m) ~ 6.28. Chosen so the dynamics are a clean,
# recognisable oscillation and the fixed point sits at K individuals of each.
R_PREY_BIRTH = 1.0
A_PREDATION = 1.0
B_PRED_BIRTH = 1.0
M_PRED_DEATH = 1.0

# Initial densities (offset from the fixed point so the system oscillates).
X0_DENSITY = 1.8
Y0_DENSITY = 1.0

# Calibrated population scales + horizon (see tests / arm9 report). "Large" is
# the regime where the mean-field reduction is valid; "small" is where it breaks.
LARGE_SCALE = 400
SMALL_SCALE = 8
T_END = 12.0          # ~2 natural periods
N_POINTS = 240        # emitted grid resolution for the trajectory


@dataclass
class EnsembleResult:
    """Ensemble summary over ``n_runs`` SSA trajectories at one population scale.

    ``mean``/``variance`` are (2, n_points) arrays over [prey, predator] on the
    shared time grid ``t`` (NaN rows, i.e. extinct runs, excluded pointwise).
    ``extinct_fraction`` is the fraction of runs that hit an absorbing state
    (prey or predator count 0) before ``t_end`` — a quantity the ODE can NEVER
    exhibit. ``scale`` is K; ``mean_density`` = mean / K for ODE comparison.
    """

    t: np.ndarray
    mean: np.ndarray
    variance: np.ndarray
    extinct_fraction: float
    scale: int
    n_runs: int

    @property
    def mean_density(self) -> np.ndarray:
        return self.mean / self.scale


def simulate_abm(scale, t_end=T_END, n_points=N_POINTS, rng=None,
                 x0_density=X0_DENSITY, y0_density=Y0_DENSITY):
    """One exact-SSA (Gillespie direct method) Lotka-Volterra trajectory.

    Returns ``(t, counts, extinct)``: ``t`` the uniform output grid
    (``n_points + 1`` points, 0..t_end), ``counts`` a (2, n_points+1) integer
    array of [prey, predator] sampled onto that grid (last value held across
    gaps, as a jump process is piecewise-constant), and ``extinct`` True if
    either population reached 0 before ``t_end`` (the run is then held at its
    absorbing state for the remainder of the grid).
    """
    if rng is None:
        rng = np.random.default_rng()
    r, a, b, m = R_PREY_BIRTH, A_PREDATION, B_PRED_BIRTH, M_PRED_DEATH
    x = int(round(scale * x0_density))
    y = int(round(scale * y0_density))

    grid = np.linspace(0.0, t_end, n_points + 1)
    counts = np.empty((2, n_points + 1), dtype=float)
    t = 0.0
    gi = 0
    extinct = False
    while gi <= n_points:
        # Sample every grid point we have now passed (piecewise-constant hold).
        while gi <= n_points and grid[gi] <= t:
            counts[0, gi] = x
            counts[1, gi] = y
            gi += 1
        if gi > n_points:
            break
        if x <= 0 or y <= 0:
            # Absorbing state: hold the remaining grid at the current counts.
            extinct = True
            counts[0, gi:] = x
            counts[1, gi:] = y
            break
        p_birth_prey = r * x
        p_predation = (a / scale) * x * y
        p_birth_pred = (b / scale) * x * y
        p_death_pred = m * y
        a0 = p_birth_prey + p_predation + p_birth_pred + p_death_pred
        if a0 <= 0.0:
            extinct = True
            counts[0, gi:] = x
            counts[1, gi:] = y
            break
        tau = rng.exponential(1.0 / a0)
        t += tau
        u = rng.random() * a0
        if u < p_birth_prey:
            x += 1
        elif u < p_birth_prey + p_predation:
            x -= 1
        elif u < p_birth_prey + p_predation + p_birth_pred:
            y += 1
        else:
            y -= 1
    return grid, counts, extinct


def abm_ensemble(n_runs, scale, t_end=T_END, n_points=N_POINTS, seed=0,
                 x0_density=X0_DENSITY, y0_density=Y0_DENSITY) -> EnsembleResult:
    """Run ``n_runs`` independent SSA trajectories; return an :class:`EnsembleResult`.

    Seed-controlled (one child RNG per run from a ``SeedSequence``) so the
    ensemble statistics are reproducible. The ensemble mean/variance are taken
    over ALL runs (extinct runs included — they are held at their absorbing
    state, which is the honest ensemble behaviour); ``extinct_fraction`` reports
    the share that went extinct so the discreteness/absorption effect is
    quantified rather than hidden.
    """
    children = np.random.SeedSequence(seed).spawn(n_runs)
    t = None
    stacks = []
    n_extinct = 0
    for child in children:
        rng = np.random.default_rng(child)
        grid, counts, extinct = simulate_abm(
            scale, t_end=t_end, n_points=n_points, rng=rng,
            x0_density=x0_density, y0_density=y0_density)
        t = grid
        stacks.append(counts)
        n_extinct += int(extinct)
    arr = np.stack(stacks, axis=0)  # (n_runs, 2, n_points+1)
    mean = arr.mean(axis=0)
    variance = arr.var(axis=0)
    return EnsembleResult(
        t=t, mean=mean, variance=variance,
        extinct_fraction=n_extinct / n_runs, scale=int(scale), n_runs=int(n_runs))


class PredatorPreyABMProcess(Process):
    """Stochastic Lotka-Volterra ABM as one process-bigraph process.

    Owns the exact-SSA stepper and advances the integer population state over
    each coupling ``interval`` with the Gillespie direct method. The state leaf
    is a 2-element integer-valued ``array`` ``population`` = [prey, predator];
    it is overwritten each tick (a jump process is not an additive delta). No
    ODE / libroadrunner is involved — this model has no continuous RHS. Each run
    is ONE stochastic realisation; an ensemble (:func:`abm_ensemble`) is needed
    for statistics.
    """

    config_schema = {
        "scale": {"_type": "integer", "_default": LARGE_SCALE},
        "seed": {"_type": "integer", "_default": -1},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self._scale = int(self.config["scale"])
        seed = int(self.config["seed"])
        self._rng = np.random.default_rng(None if seed < 0 else seed)

    def inputs(self):
        return {"population": "array[(2),float]"}

    def outputs(self):
        return {"population": "overwrite[array[(2),float]]"}

    def _step_interval(self, x, y, interval):
        r, a, b, m = R_PREY_BIRTH, A_PREDATION, B_PRED_BIRTH, M_PRED_DEATH
        scale = self._scale
        t = 0.0
        while True:
            if x <= 0 or y <= 0:
                break
            p_birth_prey = r * x
            p_predation = (a / scale) * x * y
            p_birth_pred = (b / scale) * x * y
            p_death_pred = m * y
            a0 = p_birth_prey + p_predation + p_birth_pred + p_death_pred
            if a0 <= 0.0:
                break
            tau = self._rng.exponential(1.0 / a0)
            if t + tau > interval:
                break
            t += tau
            u = self._rng.random() * a0
            if u < p_birth_prey:
                x += 1
            elif u < p_birth_prey + p_predation:
                x -= 1
            elif u < p_birth_prey + p_predation + p_birth_pred:
                y += 1
            else:
                y -= 1
        return x, y

    def initial_state(self):
        x = int(round(self._scale * X0_DENSITY))
        y = int(round(self._scale * Y0_DENSITY))
        return {"population": np.array([x, y], dtype=float)}

    def update(self, state, interval):
        pop = state.get("population") if state else None
        if pop is None:
            raise ValueError("PredatorPreyABMProcess requires a 'population' input")
        x = int(round(float(pop[0])))
        y = int(round(float(pop[1])))
        x, y = self._step_interval(x, y, interval)
        return {"population": np.array([x, y], dtype=float)}


def register_predator_prey_abm(core=None):
    """Register PredatorPreyABMProcess + the RAM emitter on ``core``."""
    if core is None:
        core = allocate_core()
    core.register_link("PredatorPreyABMProcess", PredatorPreyABMProcess)
    core.register_link("ram-emitter", RAMEmitter)
    return core
