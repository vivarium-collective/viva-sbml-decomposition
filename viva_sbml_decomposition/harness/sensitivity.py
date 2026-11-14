"""Global sensitivity harness: which parameters drive the glycolysis model?

Seeded Morris elementary-effects screening over the enzyme-capacity (Vmax)
parameters, evaluated in parallel on the Ray ensemble (serial fallback). Morris
is the cheap, robust global-screening method: `r` trajectories each step one
parameter at a time on a normalized grid, giving per-parameter elementary
effects. We report `mu_star` (mean |effect| = importance) and `sigma` (std of
effects = interaction / nonlinearity), and the importance ranking.

Hand-rolled (no SALib dependency). Deterministic given `seed`: the sample and
thus the indices reproduce exactly on a machine; only raw last digits vary across
BLAS/platforms (grade the ranking, not magnitudes). Parallelism never changes the
result — the indices come from the per-sample QoI, which is backend-independent.
"""
from __future__ import annotations

import platform

import numpy as np
import tellurium as te

from viva_sbml_decomposition.composites.ensemble import run_ensemble
from viva_sbml_decomposition.models.glycolysis import GLYCOLYSIS_ANTIMONY


def _vmax_nominal() -> dict:
    """The Vmax (enzyme-capacity) parameters and their nominal values."""
    rr = te.loada(GLYCOLYSIS_ANTIMONY)
    return {p: float(rr[p]) for p in rr.getGlobalParameterIds() if "Vm" in p}


def morris_sensitivity(params=None, *, r: int = 4, levels: int = 4, span: float = 0.2,
                       seed: int = 0, qoi: str = "ATP") -> dict:
    """Morris elementary-effects screening of `qoi` over `params` (default: Vmax).

    `r` trajectories on a `levels`-point grid, each parameter ranged ±`span` around
    its nominal value. Returns params, mu_star, sigma, ranking (importance desc),
    qoi, backend, machine.
    """
    nominal = _vmax_nominal()
    if params is None:
        params = list(nominal)
    D = len(params)
    lo = np.array([nominal[p] * (1.0 - span) for p in params])
    hi = np.array([nominal[p] * (1.0 + span) for p in params])

    rng = np.random.default_rng(seed)
    grid = np.linspace(0.0, 1.0, levels)            # normalized grid points
    delta = levels / (2.0 * (levels - 1))           # standard Morris step (grid units)
    base_choices = grid[grid <= 1.0 - delta + 1e-12]  # bases that leave room for +delta

    # Build r trajectories in normalized [0,1] space: each is base + D one-at-a-time steps.
    norm_samples = []           # normalized D-vectors, r*(D+1) of them
    stepped_param = []          # the param index stepped to reach each sample (None for a base)
    for _ in range(r):
        x = rng.choice(base_choices, size=D)
        order = rng.permutation(D)
        norm_samples.append(x.copy())
        stepped_param.append(None)
        for pi in order:
            x = x.copy()
            x[pi] = min(1.0, x[pi] + delta)
            norm_samples.append(x.copy())
            stepped_param.append(int(pi))

    # Denormalize to parameter-override draws and evaluate the QoI in parallel.
    draws = [{p: float(v) for p, v in zip(params, lo + x * (hi - lo))} for x in norm_samples]
    results = run_ensemble(model=GLYCOLYSIS_ANTIMONY, engine="tellurium",
                           n_instances=len(draws), param_draws=draws, full=False)
    outs = [float(d[qoi]) for d in results]

    # Elementary effects per parameter: (QoI after the step - QoI before) / delta.
    effects = {p: [] for p in params}
    per_traj = D + 1
    for ti in range(r):
        start = ti * per_traj
        prev = outs[start]
        for k in range(1, per_traj):
            idx = start + k
            pi = stepped_param[idx]
            effects[params[pi]].append((outs[idx] - prev) / delta)
            prev = outs[idx]

    mu_star = [float(np.mean(np.abs(effects[p]))) for p in params]
    sigma = [float(np.std(effects[p])) for p in params]
    ranking = [p for _, p in sorted(zip(mu_star, params), key=lambda t: t[0], reverse=True)]

    return {"params": list(params), "mu_star": mu_star, "sigma": sigma,
            "ranking": ranking, "qoi": qoi,
            "backend": getattr(results, "backend", "serial"),
            "machine": platform.platform()}
