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


class RateLawFitter(Step):
    """Fit a candidate flat-SBML model's rate constants to the target (the "fit" step).

    Given a candidate model with the RIGHT structure but wrong/unknown parameters,
    minimizes the mean range-normalized trajectory error against the oracle target
    (Nelder-Mead) over ``param_names`` (default: all global parameters). This is the
    build loop's parameter-recovery step — it turns a proposed structure into a
    fitted candidate the comparator + structural scorer then grade. Returns the
    fitted parameters, the fitted Antimony, and the achieved nrmse.
    """

    config_schema = {
        "candidate_antimony": {"_type": "string", "_default": ""},
        "target": {"_type": "string", "_default": ""},
        "param_names": {"_type": "list[string]", "_default": []},
        "observed": {"_type": "list[string]", "_default": []},
        "n_points": {"_type": "integer", "_default": 101},
        "max_iter": {"_type": "integer", "_default": 400},
        # auto | trajectory | gradient | two_stage. Gradient matching fits params so
        # the model's RHS matches derivatives estimated from the data (phase-free —
        # the only way to recover OSCILLATORS, where trajectory LSQ is ill-
        # conditioned); two_stage gradient-seeds a trajectory refine. `auto` (default)
        # runs trajectory-only AND two_stage and keeps whichever fits better, so it
        # recovers every ladder rung: exact params where identifiable, the oscillator
        # via gradient-matching.
        "method": {"_type": "string", "_default": "auto"},
    }

    def inputs(self):
        return {}

    def outputs(self):
        return {"fitted_antimony": "string", "fitted_params": "map[float]", "nrmse": "float"}

    def update(self, state):
        import tellurium as te
        from scipy.optimize import least_squares, minimize

        from ..models.recovery_targets import RECOVERY_TARGETS

        cfg = self.config
        if not cfg["candidate_antimony"]:
            raise ValueError("RateLawFitter requires a 'candidate_antimony'.")
        entry = RECOVERY_TARGETS[cfg["target"]]
        truth, meta = entry["antimony"], entry["meta"]
        t_end, n = meta["t_end"], int(cfg["n_points"])
        observed = cfg["observed"] or meta["species"]
        method = cfg["method"] or "two_stage"

        rr_t = _load(truth)
        rr_t.reset()
        tt = np.asarray(rr_t.simulate(0.0, t_end, n, selections=["time"] + observed))
        times = tt[:, 0]
        target = {s: [float(v) for v in tt[:, i + 1]] for i, s in enumerate(observed)}

        rr = _load(cfg["candidate_antimony"])
        pnames = list(cfg["param_names"]) or list(rr.getGlobalParameterIds())
        x0 = np.array([float(rr[p]) for p in pnames], dtype=float)

        def _grad_match(start):
            # Fit params so the model RHS matches data-estimated derivatives at each
            # sample (no integration -> phase-free, well-conditioned).
            data = np.asarray([target[s] for s in observed]).T
            dxdt = np.gradient(data, times[1] - times[0], axis=0)
            roc_order = list(rr.getFloatingSpeciesIds())
            idx = [roc_order.index(s) for s in observed]

            def _resid(theta):
                for p, v in zip(pnames, theta):
                    rr[p] = max(float(v), 1e-12)
                out = []
                for k in range(0, len(times), 3):
                    for i, s in enumerate(observed):
                        rr[s] = float(data[k, i])
                    roc = rr.getRatesOfChange()
                    for i in range(len(observed)):
                        out.append(dxdt[k, i] - float(roc[idx[i]]))
                return np.array(out)

            res = least_squares(_resid, start, method="lm", max_nfev=2000)
            return np.maximum(res.x, 1e-12)

        def _traj_cost(x):
            try:
                rr.reset()
                for p, v in zip(pnames, x):
                    rr[p] = max(float(v), 1e-9)
                res = rr.simulate(0.0, t_end, n, selections=["time"] + observed)
                errs = [_nrmse(target[s], [float(v) for v in res[:, i + 1]]) for i, s in enumerate(observed)]
                return float(np.mean(errs))
            except Exception:
                return 1e6

        def _traj_refine(start):
            r = minimize(_traj_cost, start, method="Nelder-Mead",
                         options={"maxiter": int(cfg["max_iter"]), "xatol": 1e-7, "fatol": 1e-9})
            return np.maximum(r.x, 1e-9)

        if method == "gradient":
            xf = _grad_match(x0)
        elif method == "trajectory":
            xf = _traj_refine(x0)
        elif method == "two_stage":
            xf = _traj_refine(_grad_match(x0))
        else:  # auto: best of trajectory-only and gradient-seeded two-stage
            cands = []
            xt = _traj_refine(x0)
            cands.append((_traj_cost(xt), xt))
            xts = _traj_refine(_grad_match(x0))
            cands.append((_traj_cost(xts), xts))
            xf = min(cands, key=lambda c: c[0])[1]

        fitted = {p: float(v) for p, v in zip(pnames, xf)}
        rr.reset()
        for p, v in fitted.items():
            rr[p] = v
        try:
            fitted_antimony = te.sbmlToAntimony(rr.getCurrentSBML())
        except Exception:
            fitted_antimony = cfg["candidate_antimony"]
        return {"fitted_antimony": fitted_antimony, "fitted_params": fitted, "nrmse": float(_traj_cost(xf))}


class ReactionProposer(Step):
    """Propose a candidate flat-SBML reaction network.

    Two modes:

    * ``enumerate`` (default) — a blind pool: every unimolecular conversion X -> Y
      and (optionally) the bimolecular associations X + Y -> Z, each with a small
      unit rate constant, as a candidate the RateLawFitter then fits/prunes.
    * ``from_data`` — **residual/data-driven structure discovery (SINDy)**: fit the
      data-estimated derivatives against a reaction-pool library with sequentially-
      thresholded least-squares, keeping only the reactions whose rate constants
      survive thresholding. Recovers the actual network (structure + parameters)
      from the target data alone — no prior structure — e.g. recovering A -> B,
      A + B <-> C from mass-action trajectories.
    """

    config_schema = {
        "species": {"_type": "list[string]", "_default": []},
        "initial": {"_type": "map[float]", "_default": {}},
        "include_bimolecular": {"_type": "boolean", "_default": False},
        "k0": {"_type": "float", "_default": 0.1},
        "mode": {"_type": "string", "_default": "enumerate"},
        "target": {"_type": "string", "_default": ""},
        "observed": {"_type": "list[string]", "_default": []},
        "threshold": {"_type": "float", "_default": 0.02},
        "n_points": {"_type": "integer", "_default": 201},
    }

    def inputs(self):
        return {}

    def outputs(self):
        return {"candidate_antimony": "string", "n_reactions": "integer"}

    def update(self, state):
        if (self.config.get("mode") or "enumerate") == "from_data":
            return self._from_data()
        sp = list(self.config["species"])
        if len(sp) < 2:
            raise ValueError("ReactionProposer needs >= 2 species")
        init = self.config["initial"] or {}
        k0 = float(self.config["k0"])
        rxns: list[tuple[str, str]] = [(f"{a} -> {b}", a) for a in sp for b in sp if a != b]
        if self.config["include_bimolecular"]:
            for i, a in enumerate(sp):
                for b in sp[i + 1:]:
                    for c in sp:
                        if c not in (a, b):
                            rxns.append((f"{a} + {b} -> {c}", f"{a}*{b}"))
        lines = ["model proposed()", "  species " + ", ".join(sp) + ";"]
        lines += [f"  {s} = {float(init.get(s, 0.0))};" for s in sp]
        lines += [f"  k{i} = {k0};" for i in range(len(rxns))]
        lines += [f"  R{i}: {rx}; k{i}*{rate};" for i, (rx, rate) in enumerate(rxns)]
        lines.append("end")
        return {"candidate_antimony": "\n".join(lines), "n_reactions": len(rxns)}

    def _from_data(self):
        from ..models.recovery_targets import RECOVERY_TARGETS
        cfg = self.config
        entry = RECOVERY_TARGETS[cfg["target"]]
        truth, meta = entry["antimony"], entry["meta"]
        observed = cfg["observed"] or meta["species"]
        rr = _load(truth)
        rr.reset()
        res = np.asarray(rr.simulate(0.0, meta["t_end"], int(cfg["n_points"]), selections=["time"] + observed))
        data, times = res[:, 1:], res[:, 0]
        reactions = _sindy_reactions(data, times, list(observed), float(cfg["threshold"]))
        init = {s: float(data[0, i]) for i, s in enumerate(observed)}
        return {
            "candidate_antimony": _reactions_to_antimony("recovered", list(observed), init, reactions),
            "n_reactions": len(reactions),
        }


def _reaction_pool(species: list[str]) -> list[tuple[dict, dict]]:
    """Candidate mass-action reactions over ``species``: unimolecular X -> Y,
    dissociation X -> Y + Z, and association X + Y -> Z."""
    from itertools import combinations

    rx: list[tuple[dict, dict]] = []
    for a in species:
        for b in species:
            if a != b:
                rx.append(({a: 1}, {b: 1}))
    for a in species:
        for b, c in combinations([s for s in species if s != a], 2):
            rx.append(({a: 1}, {b: 1, c: 1}))
    for a, b in combinations(species, 2):
        for c in species:
            if c not in (a, b):
                rx.append(({a: 1, b: 1}, {c: 1}))
    return rx


def _sindy_reactions(data, times, species: list[str], threshold: float) -> list[tuple[dict, dict, float]]:
    """Sparse identification of a mass-action network from trajectories: regress the
    data-estimated derivatives on the reaction-pool library with sequentially-
    thresholded least-squares. Returns the surviving ``(reactants, products, k)``."""
    dxdt = np.gradient(data, times[1] - times[0], axis=0)
    pool = _reaction_pool(species)

    def _mono(reactants):
        m = np.ones(data.shape[0])
        for s, c in reactants.items():
            m = m * data[:, species.index(s)] ** c
        return m

    cols = []
    for reac, prod in pool:
        S = np.zeros(len(species))
        for s, c in reac.items():
            S[species.index(s)] -= c
        for s, c in prod.items():
            S[species.index(s)] += c
        m = _mono(reac)
        cols.append(np.concatenate([S[i] * m for i in range(len(species))]))
    A = np.column_stack(cols)
    y = np.concatenate([dxdt[:, i] for i in range(len(species))])
    k, *_ = np.linalg.lstsq(A, y, rcond=None)
    for _ in range(10):  # sequentially thresholded least-squares
        small = np.abs(k) < threshold
        k[small] = 0.0
        big = ~small
        if big.any():
            k[big], *_ = np.linalg.lstsq(A[:, big], y, rcond=None)
    return [(reac, prod, float(k[j])) for j, (reac, prod) in enumerate(pool) if abs(k[j]) > 1e-6]


def _reactions_to_antimony(name: str, species: list[str], initial: dict, reactions: list) -> str:
    def _side(d):
        return " + ".join((f"{c} {s}" if c > 1 else s) for s, c in d.items())

    lines = [f"model {name}()", "  species " + ", ".join(species) + ";"]
    lines += [f"  {s} = {float(initial.get(s, 0.0))};" for s in species]
    lines += [f"  k{j} = {k};" for j, (_, _, k) in enumerate(reactions)]
    for j, (reac, prod, _) in enumerate(reactions):
        rate = f"k{j}" + "".join(f"*{s}" + (f"^{c}" if c > 1 else "") for s, c in reac.items())
        lines.append(f"  R{j}: {_side(reac)} -> {_side(prod)}; {rate};")
    lines.append("end")
    return "\n".join(lines)


def _poly_rhs_min_r2(data, times, degree: int) -> float:
    """Min over species of the R^2 of a best-fit polynomial RHS (monomials up to
    ``degree``) for the data-estimated derivatives. ~1 => the observed dynamics are a
    polynomial (mass-action) reaction network; low => they are not (e.g. a single
    stochastic trajectory, whose derivatives no smooth RHS can fit)."""
    from itertools import combinations_with_replacement

    n, nsp = data.shape
    cols = [np.ones(n)]
    for d in range(1, degree + 1):
        for combo in combinations_with_replacement(range(nsp), d):
            col = np.ones(n)
            for j in combo:
                col = col * data[:, j]
            cols.append(col)
    M = np.column_stack(cols)
    dxdt = np.gradient(data, times[1] - times[0], axis=0)
    r2s = []
    for i in range(nsp):
        y = dxdt[:, i]
        coef, *_ = np.linalg.lstsq(M, y, rcond=None)
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2s.append(1.0 if ss_tot < 1e-12 else 1.0 - float(np.sum((y - M @ coef) ** 2)) / ss_tot)
    return min(r2s) if r2s else 0.0


class NonRecoverabilityDetector(Step):
    """Decide when a target is NOT recoverable as a flat SBML reaction network.

    Fits the best polynomial (mass-action) RHS to the data-estimated derivatives and
    reports the worst-species R^2: ~1 means the observed dynamics ARE a flat reaction
    network (recoverable); a low value means they are not — the honest-failure signal
    that stops the loop overfitting. The canonical non-recoverable case is a single
    stochastic (Gillespie) trajectory at low copy numbers: it is flat-SBML only in the
    large-population mean-field limit, so its noisy derivatives fit no smooth RHS.

    Scope: this detects recoverability of the DETERMINISTIC flat-SBML form from the
    given trajectory. A fully-observed deterministic network reads recoverable; a
    stochastic/low-copy trajectory reads non-recoverable.
    """

    config_schema = {
        "target": {"_type": "string", "_default": ""},
        "observed": {"_type": "list[string]", "_default": []},
        "max_degree": {"_type": "integer", "_default": 3},
        "r2_threshold": {"_type": "float", "_default": 0.9},
        "n_points": {"_type": "integer", "_default": 201},
        "stochastic": {"_type": "boolean", "_default": False},
        "seed": {"_type": "integer", "_default": 0},
    }

    def inputs(self):
        return {}

    def outputs(self):
        return {"non_recoverable": "boolean", "min_r2": "float", "detail": "string"}

    def update(self, state):
        from ..models.recovery_targets import RECOVERY_TARGETS

        cfg = self.config
        entry = RECOVERY_TARGETS[cfg["target"]]
        truth, meta = entry["antimony"], entry["meta"]
        observed = cfg["observed"] or meta["species"]
        stochastic = bool(cfg["stochastic"] or meta.get("stochastic"))
        rr = _load(truth)
        rr.reset()
        if stochastic:
            rr.setIntegrator("gillespie")
            try:
                rr.integrator.seed = int(cfg["seed"])
            except Exception:
                pass
        res = np.asarray(rr.simulate(0.0, meta["t_end"], int(cfg["n_points"]), selections=["time"] + observed))
        min_r2 = _poly_rhs_min_r2(res[:, 1:], res[:, 0], int(cfg["max_degree"]))
        thr = float(cfg["r2_threshold"])
        return {
            "non_recoverable": bool(min_r2 < thr),
            "min_r2": float(min_r2),
            "detail": f"best polynomial-RHS min R^2 {min_r2:.3g} vs {thr}" + (" (stochastic)" if stochastic else ""),
        }


def register_recovery(core):
    """Register the recovery capabilities (idempotent)."""
    for name, cls in (
        ("SyntheticTargetGenerator", SyntheticTargetGenerator),
        ("TrajectoryComparator", TrajectoryComparator),
        ("StructuralScorer", StructuralScorer),
        ("RateLawFitter", RateLawFitter),
        ("ReactionProposer", ReactionProposer),
        ("NonRecoverabilityDetector", NonRecoverabilityDetector),
    ):
        if name not in core.link_registry:
            core.register_link(name, cls)
    if "ram-emitter" not in core.link_registry:
        core.register_link("ram-emitter", RAMEmitter)
    return core
