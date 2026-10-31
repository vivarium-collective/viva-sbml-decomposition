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

import re

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


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[A-Za-z_]\w*", s or ""))


def _reactions_from_antimony(antimony: str) -> dict:
    """Extract ``{reaction_id: {"sig": (reactants, products), "params": {name: value}}}``.

    ``sig`` is a stoichiometry signature (frozensets of (species, coefficient)) used
    to MATCH reactions across models regardless of id; ``params`` are the numeric
    parameters the reaction's kinetic law references (local parameters + any global
    parameter named in the rate expression).
    """
    import libsbml
    import tellurium as te

    sbml = te.loada(antimony).getCurrentSBML()
    model = libsbml.readSBMLFromString(sbml).getModel()
    globals_ = {
        model.getParameter(i).getId(): float(model.getParameter(i).getValue())
        for i in range(model.getNumParameters())
    }
    out: dict = {}
    for i in range(model.getNumReactions()):
        rxn = model.getReaction(i)
        reactants = frozenset(
            (rxn.getReactant(j).getSpecies(), float(rxn.getReactant(j).getStoichiometry()))
            for j in range(rxn.getNumReactants())
        )
        products = frozenset(
            (rxn.getProduct(j).getSpecies(), float(rxn.getProduct(j).getStoichiometry()))
            for j in range(rxn.getNumProducts())
        )
        params: dict = {}
        kl = rxn.getKineticLaw()
        if kl is not None:
            for k in range(kl.getNumLocalParameters()):
                lp = kl.getLocalParameter(k)
                params[lp.getId()] = float(lp.getValue())
            formula = libsbml.formulaToString(kl.getMath()) if kl.getMath() else ""
            for name in _tokens(formula) & set(globals_):
                params[name] = globals_[name]
        out[rxn.getId()] = {"sig": (reactants, products), "params": params}
    return out


def _param_set_rel_err(truth_vals: list[float], cand_vals: list[float]) -> float:
    """Max over truth params of the min relative error to any candidate param value."""
    if not truth_vals:
        return 0.0
    if not cand_vals:
        return float("inf")
    errs = []
    for tv in truth_vals:
        denom = abs(tv) if abs(tv) > 1e-12 else 1.0
        errs.append(min(abs(cv - tv) / denom for cv in cand_vals))
    return max(errs)


class StructuralScorer(Step):
    """Grade a recovered flat-SBML model against the ground-truth oracle.

    Parses both ``candidate_antimony`` and ``truth_antimony`` (the hidden answer
    key), matches reactions by stoichiometry signature, and reports: the fraction of
    ground-truth reactions recovered, the count of spurious (unmatched) candidate
    reactions, the worst matched-reaction parameter relative error, and two gate
    flags — ``structure_recovered`` (every truth reaction matched, parameters within
    ``param_tol``, and — when ``require_no_spurious`` — no extra reactions) and
    ``parsimonious`` (no more reactions than the truth).
    """

    config_schema = {
        "param_tol": {"_type": "float", "_default": 0.1},
        "require_no_spurious": {"_type": "boolean", "_default": True},
    }

    def inputs(self):
        return {"candidate_antimony": "string", "truth_antimony": "string"}

    def outputs(self):
        return {
            "reactions_recovered": "float",
            "spurious_count": "integer",
            "param_max_rel_err": "float",
            "structure_recovered": "boolean",
            "parsimonious": "boolean",
        }

    def update(self, state):
        cand = _reactions_from_antimony((state or {})["candidate_antimony"])
        truth = _reactions_from_antimony((state or {})["truth_antimony"])
        truth_by_sig: dict = {}
        for r in truth.values():
            truth_by_sig.setdefault(r["sig"], []).append(r["params"])
        cand_by_sig: dict = {}
        for r in cand.values():
            cand_by_sig.setdefault(r["sig"], []).append(r["params"])

        matched = 0
        param_errs: list[float] = []
        for sig, truth_list in truth_by_sig.items():
            cand_list = cand_by_sig.get(sig, [])
            n = min(len(truth_list), len(cand_list))
            matched += n
            for k in range(n):
                param_errs.append(
                    _param_set_rel_err(list(truth_list[k].values()), list(cand_list[k].values()))
                )
        total_truth = sum(len(v) for v in truth_by_sig.values())
        cand_count = sum(len(v) for v in cand_by_sig.values())
        spurious = sum(
            max(0, len(cl) - len(truth_by_sig.get(sig, [])))
            for sig, cl in cand_by_sig.items()
        )
        reactions_recovered = matched / total_truth if total_truth else 1.0
        param_max = max(param_errs) if param_errs else 0.0
        no_spurious = spurious == 0 or not self.config["require_no_spurious"]
        structure_recovered = bool(
            matched == total_truth
            and param_max <= float(self.config["param_tol"])
            and no_spurious
        )
        return {
            "reactions_recovered": float(reactions_recovered),
            "spurious_count": int(spurious),
            "param_max_rel_err": float(param_max),
            "structure_recovered": structure_recovered,
            "parsimonious": bool(cand_count <= total_truth),
        }


def register_recovery(core):
    """Register the recovery capabilities (idempotent)."""
    for name, cls in (
        ("SyntheticTargetGenerator", SyntheticTargetGenerator),
        ("TrajectoryComparator", TrajectoryComparator),
        ("StructuralScorer", StructuralScorer),
    ):
        if name not in core.link_registry:
            core.register_link(name, cls)
    if "ram-emitter" not in core.link_registry:
        core.register_link("ram-emitter", RAMEmitter)
    return core
