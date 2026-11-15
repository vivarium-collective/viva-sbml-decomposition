"""Grid-posterior Bayesian calibration of the mass-action rate constants.

Given noisy synthetic data from the known mass-action model (k1, k2, k2r), infer
the posterior by evaluating the Gaussian likelihood on a grid over the three rate
constants under a uniform prior — the EXACT posterior up to grid resolution, with
no MCMC convergence to validate and full determinism given the data seed (the
right tool for three well-identified parameters; mass-action is fully recoverable).

Returns marginal posteriors, the MAP, 95% credible intervals, the prior widths,
and a posterior-predictive band in observation space over the data. Raw magnitudes
vary in their last digits across BLAS/platforms — grade coverage + informativeness,
not a raw posterior value.
"""
from __future__ import annotations

import itertools
import platform

import numpy as np

from viva_sbml_decomposition._tellurium import load
from viva_sbml_decomposition.models.recovery_targets import MASS_ACTION_ANTIMONY, MASS_ACTION_META

_TRUE = {"k1": 0.5, "k2": 0.3, "k2r": 0.1}
_Z95 = 1.959963984540054  # 97.5th percentile of the standard normal


def _simulate(k: dict, t_end: float, n_points: int) -> dict:
    """Forward-simulate the mass-action model at rate constants `k`."""
    rr = load(MASS_ACTION_ANTIMONY)
    for name, val in k.items():
        rr[name] = float(val)
    rr.reset()
    species = MASS_ACTION_META["species"]
    out = rr.simulate(0, t_end, n_points, selections=["time"] + species)
    return {sp: np.asarray(out[:, i + 1], dtype=float) for i, sp in enumerate(species)}


def grid_calibration(*, params=("k1", "k2", "k2r"), grid_points: int = 11, span: float = 0.5,
                     sigma: float = 0.05, n_points: int = 40, seed: int = 0) -> dict:
    """Exact grid posterior over `params` from noisy mass-action data. See module docstring."""
    from viva_sbml_decomposition.core import build_core
    from viva_sbml_decomposition.processes.recovery import SyntheticTargetGenerator

    species = MASS_ACTION_META["species"]
    t_end = MASS_ACTION_META["t_end"]
    data = SyntheticTargetGenerator(config={
        "antimony": MASS_ACTION_ANTIMONY, "t_end": t_end, "n_points": n_points,
        "observed": list(species), "noise_sigma": sigma, "seed": seed,
    }, core=build_core()).update({})
    times = np.asarray(data["time"], dtype=float)
    data_traj = {sp: np.asarray(data["trajectories"][sp], dtype=float) for sp in species}
    # Relative-noise likelihood: per-point std ~ |data| * sigma (how the data was generated).
    std = {sp: sigma * (np.abs(data_traj[sp]) + 1e-3) for sp in species}

    axes = {p: np.linspace(_TRUE[p] * (1 - span), _TRUE[p] * (1 + span), grid_points) for p in params}
    shape = tuple(grid_points for _ in params)
    logpost = np.empty(shape, dtype=float)
    sims: dict = {}
    for idx in itertools.product(*(range(grid_points) for _ in params)):
        k = {p: float(axes[p][idx[i]]) for i, p in enumerate(params)}
        model = _simulate(k, t_end, len(times))
        ll = 0.0
        for sp in species:
            ll += -0.5 * float(np.sum(((model[sp] - data_traj[sp]) / std[sp]) ** 2))
        logpost[idx] = ll
        sims[idx] = model

    post = np.exp(logpost - logpost.max())
    post /= post.sum()                                   # normalized joint posterior (uniform prior)

    map_idx = np.unravel_index(int(np.argmax(post)), shape)
    map_k = {p: float(axes[p][map_idx[i]]) for i, p in enumerate(params)}

    marginals, ci95, prior_width = {}, {}, {}
    for i, p in enumerate(params):
        other = tuple(j for j in range(len(params)) if j != i)
        marg = post.sum(axis=other)
        marg = marg / marg.sum()
        marginals[p] = [float(x) for x in marg]
        cdf = np.cumsum(marg)
        lo_i = min(int(np.searchsorted(cdf, 0.025)), grid_points - 1)
        hi_i = min(int(np.searchsorted(cdf, 0.975)), grid_points - 1)
        # Grid points are cell centers; the credible interval covers whole cells, so
        # extend to the outer cell boundaries (±half a grid step).
        step = float(axes[p][1] - axes[p][0])
        ci95[p] = [float(axes[p][lo_i]) - step / 2.0, float(axes[p][hi_i]) + step / 2.0]
        prior_width[p] = float(axes[p][-1] - axes[p][0])

    # Posterior-predictive band in OBSERVATION space: the distribution of a NEW
    # observation, so it combines parameter uncertainty with the observation noise.
    # Parameter spread alone collapses to ~zero width for a peaked posterior and
    # would exclude the noisy data; the predictive adds the relative-noise variance
    # the data was drawn with. Gaussian approximation per time:
    # mean ± 1.96·sqrt(param_var + E[(sigma·|model|)^2]), over the full posterior.
    w = post.ravel()
    all_idx = [np.unravel_index(int(c), shape) for c in range(w.size)]
    predictive = {"time": [float(t) for t in times], "observed": {}}
    for sp in species:
        stacked = np.array([sims[idx][sp] for idx in all_idx])     # (ncells, ntime)
        mean = np.average(stacked, axis=0, weights=w)
        param_var = np.average((stacked - mean) ** 2, axis=0, weights=w)
        obs_var = np.average((sigma * np.abs(stacked)) ** 2, axis=0, weights=w)
        sd = np.sqrt(param_var + obs_var)
        predictive["observed"][sp] = {
            "data": [float(x) for x in data_traj[sp]],
            "lo": [float(x) for x in (mean - _Z95 * sd)],
            "hi": [float(x) for x in (mean + _Z95 * sd)]}

    return {"params": list(params), "true": {p: _TRUE[p] for p in params},
            "grid": {p: [float(x) for x in axes[p]] for p in params},
            "marginals": marginals, "map": map_k, "ci95": ci95, "prior_width": prior_width,
            "predictive": predictive, "sigma": sigma, "machine": platform.platform()}
