"""Workspace evaluator hook — grade recovery behavior_tests with the capabilities.

``discovery_engine.study_evaluator`` calls ``register_evaluators(registry)`` and,
for any ``behavior_test`` whose ``measure.kind`` is one of ours, invokes the
registered ``evaluator(test, reader, ws_root) -> code outcome``. This is how the
recovery investigation closes the loop: each acceptance behavior is graded by the
recovery capabilities against the hidden oracle in
``viva_sbml_decomposition.models.recovery_targets``.

The candidate model under test comes from ``measure.candidate_model`` (a study slug
in RECOVERY_TARGETS, or a raw Antimony string); it defaults to the oracle itself —
the identity baseline, which recovers exactly. The build loop will later set
``candidate_model`` to the proposed/fitted model, and the same evaluators grade it.

A code outcome is ``{"result": "PASS"|"FAIL", "measured_value": ..,
"evaluated_by": "code", "operator": "<kind>/<op>", "detail": ".."}`` — the shape
``discovery_engine.study_evaluator`` writes into ``computed_outcomes``.
"""
from __future__ import annotations

# Mark our recovery evaluators STORE-LESS: they re-derive their result internally
# (re-run SINDy + fit against the hidden oracle; run the non-recoverability detector)
# and read no run data. compute_outcomes grades store-less kinds even when the run
# store is unresolved (the recover_candidate composite strips the emitter) — so the
# pipeline now owns the honest grading natively (no manual re-grade).
try:
    from discovery_engine.study_evaluator import storeless
except ImportError:  # engine predates the store-less decorator (discovery-engine #22)
    def storeless(fn):  # type: ignore[misc]
        # No-op fallback: the committed computed_outcomes already carry the grades,
        # so an older engine still renders honestly; the decorator only matters for
        # a live re-grade once the engine has store-less support.
        return fn


def _code_outcome(result: str, measured_value, operator: str, detail: str) -> dict:
    return {
        "result": result,
        "measured_value": measured_value,
        "evaluated_by": "code",
        "operator": operator,
        "detail": detail,
    }


_RECOVERED_CACHE: dict = {}


def _recovered_candidate(target: str, observed):
    """Run the ACTUAL recovery search (SINDy structure discovery + rate-law fit)
    and return the recovered flat-SBML Antimony — the candidate the build loop is
    meant to produce. Cached per target (a study's trajectory/structure/parsimony
    tests share one recovery).

    This is what makes the gate grade REAL recovery instead of the identity
    baseline (candidate = truth): a system the pipeline cannot yet recover then
    fails honestly (nrmse > tol → refuted/investigating) instead of passing
    trivially. If the search raises or yields nothing, a reaction-free model is
    returned so the gate fails honestly rather than crashing."""
    if target in _RECOVERED_CACHE:
        return _RECOVERED_CACHE[target]
    from .core import build_core
    from .models.recovery_targets import RECOVERY_TARGETS
    from .processes.recovery import RateLawFitter, ReactionProposer

    meta = RECOVERY_TARGETS[target]["meta"]
    sp = list(observed) if observed else list(meta["species"])
    cand = None
    try:
        core = build_core()
        prop = ReactionProposer(
            config={"mode": "from_data", "target": target, "observed": sp, "species": sp},
            core=core,
        ).update({})
        fit = RateLawFitter(
            config={"candidate_antimony": prop["candidate_antimony"], "target": target,
                    "observed": sp, "method": "auto"},
            core=core,
        ).update({})
        cand = fit.get("fitted_antimony") or prop.get("candidate_antimony")
    except Exception:
        cand = None
    if not cand:
        cand = "model recovered_empty()\n  species " + ", ".join(sp) + ";\nend"
    _RECOVERED_CACHE[target] = cand
    return cand


def _resolve_models(measure: dict):
    """(truth_antimony, candidate_antimony, meta) from a measure block.

    ``candidate_model: "@recovered"`` grades the REAL recovery-search output (what
    the build loop produces); absent/empty is the identity baseline (candidate =
    truth), retained for back-compat; a RECOVERY_TARGETS slug or raw Antimony is
    used verbatim."""
    from .models.recovery_targets import RECOVERY_TARGETS

    entry = RECOVERY_TARGETS[measure["target"]]
    truth, meta = entry["antimony"], entry["meta"]
    cand = measure.get("candidate_model")
    if cand == "@recovered":
        candidate = _recovered_candidate(measure["target"], measure.get("observed"))
    elif not cand:
        candidate = truth  # identity baseline (back-compat)
    elif cand in RECOVERY_TARGETS:
        candidate = RECOVERY_TARGETS[cand]["antimony"]
    else:
        candidate = cand  # raw Antimony
    return truth, candidate, meta


@storeless
def _trajectory(test, reader, ws_root):
    from .core import build_core
    from .processes.recovery import SyntheticTargetGenerator, TrajectoryComparator

    m = test.get("measure") or {}
    truth, candidate, meta = _resolve_models(m)
    tol = float(m.get("tol", 0.05))
    observed = m.get("observed") or meta["species"]
    t_end, n = meta["t_end"], int(m.get("n_points", 101))
    core = build_core()
    tgt = SyntheticTargetGenerator(
        config={"antimony": truth, "t_end": t_end, "n_points": n, "observed": observed}, core=core
    ).update({})
    cnd = SyntheticTargetGenerator(
        config={"antimony": candidate, "t_end": t_end, "n_points": n, "observed": observed}, core=core
    ).update({})
    res = TrajectoryComparator(config={"tol": tol, "observed": observed}, core=core).update(
        {"time": tgt["time"], "target": tgt["trajectories"], "candidate": cnd["trajectories"]}
    )
    return _code_outcome(
        "PASS" if res["passed"] else "FAIL", round(res["nrmse"], 6),
        "recovery_trajectory/threshold", f"nrmse {res['nrmse']:.4g} vs tol {tol}",
    )


@storeless
def _structure(test, reader, ws_root):
    from .core import build_core
    from .processes.recovery import StructuralScorer

    m = test.get("measure") or {}
    truth, candidate, _ = _resolve_models(m)
    res = StructuralScorer(config={"param_tol": float(m.get("param_tol", 0.1))}, core=build_core()).update(
        {"candidate_antimony": candidate, "truth_antimony": truth}
    )
    return _code_outcome(
        "PASS" if res["structure_recovered"] else "FAIL", round(res["reactions_recovered"], 4),
        "recovery_structure/threshold",
        f"reactions_recovered {res['reactions_recovered']:.2f}, "
        f"param_max_rel_err {res['param_max_rel_err']:.4g}, spurious {res['spurious_count']}",
    )


@storeless
def _parsimony(test, reader, ws_root):
    from .core import build_core
    from .processes.recovery import StructuralScorer

    m = test.get("measure") or {}
    truth, candidate, _ = _resolve_models(m)
    res = StructuralScorer(config={}, core=build_core()).update(
        {"candidate_antimony": candidate, "truth_antimony": truth}
    )
    return _code_outcome(
        "PASS" if res["parsimonious"] else "FAIL", float(res["spurious_count"]),
        "recovery_parsimony/threshold",
        f"parsimonious={res['parsimonious']}, spurious {res['spurious_count']}",
    )


@storeless
def _bistability(test, reader, ws_root):
    """Does the candidate reproduce the ground-truth switch — two stable states?"""
    import numpy as np
    import tellurium as te

    m = test.get("measure") or {}
    _, candidate, meta = _resolve_models(m)
    sp = meta["species"]
    if len(sp) < 2:
        return _code_outcome("FAIL", 0.0, "recovery_bistability/threshold", "need >= 2 species")
    t_end = max(meta["t_end"], 40.0)
    rr = te.loada(candidate)
    states = []
    for hi in sp[:2]:  # seed each of the two species dominant -> its basin
        rr.reset()
        for s in sp:
            rr[s] = 3.0 if s == hi else 0.05
        res = rr.simulate(0.0, t_end, 200, selections=sp)
        states.append(np.array([float(res[-1, i]) for i in range(len(sp))]))
    dist = float(np.linalg.norm(states[0] - states[1]))
    thr = float(m.get("min_separation", 0.5))
    return _code_outcome(
        "PASS" if dist > thr else "FAIL", round(dist, 4),
        "recovery_bistability/threshold", f"steady-state separation {dist:.3g} vs {thr}",
    )


@storeless
def _limit_cycle(test, reader, ws_root):
    """Does the candidate sustain an oscillation (non-decaying late-window amplitude)?"""
    import numpy as np
    import tellurium as te

    m = test.get("measure") or {}
    _, candidate, meta = _resolve_models(m)
    sp = meta["species"]
    t_end = max(meta["t_end"], 40.0)
    rr = te.loada(candidate)
    rr.reset()
    res = np.asarray(rr.simulate(0.0, t_end, 400, selections=sp))
    half = res[200:, :]
    amp = max((float(np.std(half[:, i])) for i in range(half.shape[1])), default=0.0)
    thr = float(m.get("min_amplitude", 0.2))
    return _code_outcome(
        "PASS" if amp > thr else "FAIL", round(amp, 4),
        "recovery_limit_cycle/threshold", f"late-window amplitude {amp:.3g} vs {thr}",
    )



@storeless
def _non_recoverable(test, reader, ws_root):
    from .core import build_core
    from .processes.recovery import NonRecoverabilityDetector

    m = test.get("measure") or {}
    res = NonRecoverabilityDetector(
        config={"target": m["target"], "r2_threshold": float(m.get("r2_threshold", 0.9))},
        core=build_core(),
    ).update({})
    # Adversarial: PASS when the system is CORRECTLY detected as non-recoverable.
    return _code_outcome(
        "PASS" if res["non_recoverable"] else "FAIL", round(res["min_r2"], 4),
        "recovery_non_recoverable/threshold", res["detail"],
    )


def register_evaluators(registry: dict) -> dict:
    """discovery_engine hook: map our custom measure kinds to their evaluators."""
    registry["recovery_trajectory"] = _trajectory
    registry["recovery_structure"] = _structure
    registry["recovery_parsimony"] = _parsimony
    registry["recovery_bistability"] = _bistability
    registry["recovery_limit_cycle"] = _limit_cycle
    registry["recovery_non_recoverable"] = _non_recoverable
    return registry
