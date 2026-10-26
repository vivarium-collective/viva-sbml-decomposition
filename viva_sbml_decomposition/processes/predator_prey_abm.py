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

# Numerical safety cap on prey count, as a multiple of the scale K. It bounds
# ONE unbounded regime: once predators go extinct (Y=0) the prey become a pure
# birth process (rate r*X) with no predation to check them, so prey grow without
# limit. The cap stops that runaway (and the ensuing event explosion) at a prey
# density of POP_CAP_DENSITY; it is well above the deterministic LV orbit's peak
# (~1.8 in density units), so it NEVER triggers while both species coexist — it
# only marks "predators extinct, prey escaped". A capped run is held at the cap
# for the remainder of the grid.
POP_CAP_DENSITY = 25.0


@dataclass
class EnsembleResult:
    """Ensemble summary over ``n_runs`` SSA trajectories at one population scale.

    ``mean``/``variance`` are (2, n_points+1) arrays over [prey, predator] on the
    shared time grid ``t``, taken over ALL ``n_runs`` (including runs that went
    extinct or hit the prey cap — those are held at their terminal state, which
    is their honest contribution to the ensemble; nothing is dropped).

    Extinction is reported as TWO distinct fractions, because only prey
    extinction is truly absorbing while predator extinction leaves prey to grow:

    * ``predator_extinct_fraction`` — share of runs in which the PREDATORS died
      out (``Pred`` reached 0) at any point. This is the Lotka-Volterra collapse
      the neutrally-stable ODE can NEVER show (the ODE orbits forever), so it is
      the honest "the ODE misses this" metric.
    * ``total_extinct_fraction`` — share in which BOTH species reached 0 (prey
      extinction is absorbing and the predators then starve to 0).
    * ``prey_extinct_fraction`` — share in which the PREY reached 0 (reported for
      completeness; a prey-extinct run becomes total-extinct).

    ``scale`` is K; ``mean_density`` = mean / K for comparison to the ODE density.
    """

    t: np.ndarray
    mean: np.ndarray
    variance: np.ndarray
    predator_extinct_fraction: float
    total_extinct_fraction: float
    prey_extinct_fraction: float
    scale: int
    n_runs: int

    @property
    def mean_density(self) -> np.ndarray:
        return self.mean / self.scale


def simulate_abm(scale, t_end=T_END, n_points=N_POINTS, rng=None,
                 x0_density=X0_DENSITY, y0_density=Y0_DENSITY):
    """One exact-SSA (Gillespie direct method) Lotka-Volterra trajectory.

    Returns ``(t, counts, status)``. ``t`` is the uniform output grid
    (``n_points + 1`` points, 0..t_end); ``counts`` a (2, n_points+1) integer
    array of [prey, predator] sampled onto that grid. A jump process is constant
    between events, so each grid point records the state that HELD over the
    waiting interval containing it (the PRE-event state) — not the post-event
    state. ``status`` is a dict of booleans: ``prey_extinct``, ``predator_extinct``,
    ``total_extinct``, ``capped``.

    No state is frozen on predator-only extinction: when ``Pred`` hits 0 the prey
    keep reproducing (pure birth ``r*X``), exactly as the real process does, until
    the prey cap (:data:`POP_CAP_DENSITY` * scale) bounds the runaway. Only prey
    extinction (``Prey`` = 0) is genuinely absorbing — the predators then starve
    to 0 — and the SSA runs that decay out naturally (``a0`` -> 0).
    """
    if rng is None:
        rng = np.random.default_rng()
    r, a, b, m = R_PREY_BIRTH, A_PREDATION, B_PRED_BIRTH, M_PRED_DEATH
    x = int(round(scale * x0_density))
    y = int(round(scale * y0_density))
    cap = int(np.ceil(POP_CAP_DENSITY * scale))

    grid = np.linspace(0.0, t_end, n_points + 1)
    counts = np.empty((2, n_points + 1), dtype=float)
    t = 0.0
    gi = 0
    prey_extinct = x <= 0
    predator_extinct = y <= 0
    capped = False
    while gi <= n_points:
        p_birth_prey = r * x if x < cap else 0.0  # cap halts the pure-birth runaway
        p_predation = (a / scale) * x * y
        p_birth_pred = (b / scale) * x * y
        p_death_pred = m * y
        a0 = p_birth_prey + p_predation + p_birth_pred + p_death_pred
        if a0 <= 0.0:
            # No event can fire (both extinct, or prey pinned at the cap with no
            # predators): hold the current state across the rest of the grid.
            counts[0, gi:] = x
            counts[1, gi:] = y
            break
        tau = rng.exponential(1.0 / a0)
        t_next = t + tau
        # Record every grid point inside [t, t_next) with the current PRE-event
        # state (the state that held over this waiting interval).
        while gi <= n_points and grid[gi] < t_next:
            counts[0, gi] = x
            counts[1, gi] = y
            gi += 1
        if gi > n_points:
            break
        # Apply the event at t_next.
        u = rng.random() * a0
        if u < p_birth_prey:
            x += 1
            if x >= cap:
                capped = True
        elif u < p_birth_prey + p_predation:
            x -= 1
        elif u < p_birth_prey + p_predation + p_birth_pred:
            y += 1
        else:
            y -= 1
        t = t_next
        if x <= 0:
            prey_extinct = True
        if y <= 0:
            predator_extinct = True
    status = {
        "prey_extinct": bool(prey_extinct),
        "predator_extinct": bool(predator_extinct),
        "total_extinct": bool(prey_extinct and predator_extinct),
        "capped": bool(capped),
    }
    return grid, counts, status


def abm_ensemble(n_runs, scale, t_end=T_END, n_points=N_POINTS, seed=0,
                 x0_density=X0_DENSITY, y0_density=Y0_DENSITY) -> EnsembleResult:
    """Run ``n_runs`` independent SSA trajectories; return an :class:`EnsembleResult`.

    Seed-controlled (one child RNG per run from a ``SeedSequence``) so the
    ensemble statistics are reproducible. The ensemble mean/variance are taken
    over ALL runs (extinct and capped runs included — held at their terminal
    state, the honest ensemble behaviour); the predator / total / prey extinction
    fractions quantify the discreteness/absorption effects the ODE cannot show.
    """
    children = np.random.SeedSequence(seed).spawn(n_runs)
    t = None
    stacks = []
    n_pred = n_total = n_prey = 0
    for child in children:
        rng = np.random.default_rng(child)
        grid, counts, status = simulate_abm(
            scale, t_end=t_end, n_points=n_points, rng=rng,
            x0_density=x0_density, y0_density=y0_density)
        t = grid
        stacks.append(counts)
        n_pred += int(status["predator_extinct"])
        n_total += int(status["total_extinct"])
        n_prey += int(status["prey_extinct"])
    arr = np.stack(stacks, axis=0)  # (n_runs, 2, n_points+1)
    return EnsembleResult(
        t=t, mean=arr.mean(axis=0), variance=arr.var(axis=0),
        predator_extinct_fraction=n_pred / n_runs,
        total_extinct_fraction=n_total / n_runs,
        prey_extinct_fraction=n_prey / n_runs,
        scale=int(scale), n_runs=int(n_runs))


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
        cap = int(np.ceil(POP_CAP_DENSITY * scale))
        t = 0.0
        while True:
            # Predator-only extinction (y=0) is NOT absorbing: prey keep breeding
            # (pure birth r*X) until the cap. Only a0=0 (both extinct, or prey
            # pinned at the cap with no predators) ends the stepping.
            p_birth_prey = r * x if x < cap else 0.0
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
