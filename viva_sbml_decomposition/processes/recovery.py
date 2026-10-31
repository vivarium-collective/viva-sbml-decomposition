"""Capabilities for the discovery-engine SBML-recovery loop.

These are the first modular, vivarium-style processes the recovery investigation
(``discovery-sbml-recovery``) needs to close the gap between "run the discovery
cycle" and "actually build good flat SBML models from data". Each is a plain
process-bigraph ``Step`` (pure dataflow — no time advance):

* :class:`SyntheticTargetGenerator` — simulate a ground-truth Antimony/SBML model
  with tellurium (libroadrunner) and emit the observed-species trajectories as the
  recovery TARGET (with optional relative Gaussian observation noise). The oracle
  for a recovery study: only its outputs are exposed to the build loop; the model
  itself is the hidden answer key.
* :class:`TrajectoryComparator` — score a candidate model's trajectories against a
  target (per-species range-normalized RMSE + an overall value + a pass flag at a
  configured tolerance). The loop's "compare" step; its residual is what the
  discovery frontier reads.

Ground-truth benchmark models live in
``viva_sbml_decomposition.models.recovery_targets``.
"""
from __future__ import annotations

import numpy as np
from process_bigraph import Step, allocate_core
from process_bigraph.emitter import RAMEmitter


def _load(antimony: str):
    import tellurium as te

    return te.loada(antimony)


class SyntheticTargetGenerator(Step):
    """Ground-truth model -> synthetic target trajectories.

    Simulates ``antimony`` (config) over ``[0, t_end]`` at ``n_points`` samples and
    returns the floating-species trajectories (optionally restricted to
    ``observed``), with optional relative Gaussian noise ``value*(1+N(0, sigma))``.
    The model is passed as config (the hidden oracle), mirroring how
    :class:`SubmodelProcess` takes its model source as config.
    """

    config_schema = {
        "antimony": {"_type": "string", "_default": ""},
        "t_end": {"_type": "float", "_default": 10.0},
        "n_points": {"_type": "integer", "_default": 100},
        "observed": {"_type": "list[string]", "_default": []},
        "noise_sigma": {"_type": "float", "_default": 0.0},
        "seed": {"_type": "integer", "_default": 0},
    }

    def inputs(self):
        return {}

    def outputs(self):
        return {
            "time": "list[float]",
            "trajectories": "map[list[float]]",
            "species_ids": "list[string]",
        }

    def update(self, state):
        cfg = self.config
        if not cfg["antimony"]:
            raise ValueError("SyntheticTargetGenerator requires an 'antimony' model source.")
        rr = _load(cfg["antimony"])
        rr.reset()
        species = list(rr.getFloatingSpeciesIds())
        observed = [s for s in (cfg["observed"] or species) if s in species]
        if not observed:
            raise ValueError(
                f"none of observed={cfg['observed']} are floating species {species}"
            )
        n = max(2, int(cfg["n_points"]))
        result = rr.simulate(0.0, float(cfg["t_end"]), n, selections=["time"] + observed)
        time = [float(x) for x in result[:, 0]]
        traj = {s: [float(x) for x in result[:, i + 1]] for i, s in enumerate(observed)}
        sigma = float(cfg["noise_sigma"])
        if sigma > 0.0:
            rng = np.random.default_rng(int(cfg["seed"]))
            for s in observed:
                vals = np.asarray(traj[s], dtype=float)
                traj[s] = [float(v) for v in vals * (1.0 + rng.normal(0.0, sigma, size=vals.shape))]
        return {"time": time, "trajectories": traj, "species_ids": observed}


def _nrmse(target: list[float], candidate: list[float]) -> float:
    """Range-normalized RMSE of ``candidate`` vs ``target`` on their shared length."""
    t = np.asarray(target, dtype=float)
    c = np.asarray(candidate, dtype=float)
    n = min(len(t), len(c))
    if n == 0:
        return float("inf")
    t, c = t[:n], c[:n]
    rmse = float(np.sqrt(np.mean((c - t) ** 2)))
    rng = float(np.max(t) - np.min(t))
    denom = rng if rng > 1e-12 else (float(np.max(np.abs(t))) or 1.0)
    return rmse / denom


class TrajectoryComparator(Step):
    """Score candidate trajectories against a target.

    Reads ``target`` and ``candidate`` (both ``species -> [values]``) and reports
    per-species range-normalized RMSE, the overall (mean) value, and a pass flag
    at ``tol``. Only species present in BOTH (or in ``observed`` when set) are
    scored; a species missing from the candidate counts as a non-finite residual
    so the comparison can't silently pass by omission.
    """

    config_schema = {
        "tol": {"_type": "float", "_default": 0.05},
        "observed": {"_type": "list[string]", "_default": []},
    }

    def inputs(self):
        return {
            "time": "list[float]",
            "target": "map[list[float]]",
            "candidate": "map[list[float]]",
        }

    def outputs(self):
        return {
            "nrmse": "float",
            "per_species_nrmse": "map[float]",
            "passed": "boolean",
        }

    def update(self, state):
        target = (state or {}).get("target") or {}
        candidate = (state or {}).get("candidate") or {}
        observed = self.config["observed"] or sorted(target.keys())
        per_species: dict[str, float] = {}
        for s in observed:
            if s not in target:
                continue
            if s not in candidate:
                per_species[s] = float("inf")  # missing species is not a free pass
                continue
            per_species[s] = _nrmse(target[s], candidate[s])
        finite = [v for v in per_species.values() if np.isfinite(v)]
        if not per_species:
            overall = float("inf")
        elif len(finite) < len(per_species):
            overall = float("inf")  # any missing/non-finite species -> overall fails
        else:
            overall = float(np.mean(finite))
        passed = bool(np.isfinite(overall) and overall < float(self.config["tol"]))
        return {"nrmse": overall, "per_species_nrmse": per_species, "passed": passed}


def register_recovery(core):
    """Register the recovery capabilities (idempotent)."""
    for name, cls in (
        ("SyntheticTargetGenerator", SyntheticTargetGenerator),
        ("TrajectoryComparator", TrajectoryComparator),
    ):
        if name not in core.link_registry:
            core.register_link(name, cls)
    if "ram-emitter" not in core.link_registry:
        core.register_link("ram-emitter", RAMEmitter)
    return core
