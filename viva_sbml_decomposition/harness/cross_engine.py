"""Cross-engine agreement harness: do the engines agree on the same model?

Runs a model monolithically in each requested engine and measures, per engine,
the max relative error of its trajectory against the **tellurium** reference
(the numerical reference for the whole investigation). This is the agreement
half of the cross-engine validation study; the decomposition splitting-error is
reported separately (via the ``reaction_split`` / mechanical-split method).

Engine notes:
- ``tellurium`` (libroadrunner) is the reference; its self-agreement is 0.
- ``copasi`` runs inside the retrying subprocess that ``run_engine_monolithic``
  owns (COPASI has a documented ~1-in-6 nondeterministic loader abort and is not
  bit-reproducible); a run that still cannot complete is tolerated, not raised.
- ``native`` is a hand-ported Teusink RHS — glycolysis only; any other model
  raises, which the harness records as an abort.

An engine that cannot produce a trajectory lands in ``aborted`` with
``agreement[engine] = None``; the harness never propagates the failure, so one
flaky/unsupported engine cannot sink the whole study.
"""
from __future__ import annotations

from viva_sbml_decomposition.composites.engine_monolithic import run_engine_monolithic
from viva_sbml_decomposition.harness.accuracy import compare_to_reference

DEFAULT_ENGINES = ("tellurium", "copasi", "native")
REFERENCE = "tellurium"


def cross_engine_agreement(antimony: str, engines=DEFAULT_ENGINES, *,
                           t_end: float, n_points: int = 101, seed: int = 0) -> dict:
    """Max-rel-error of each engine's monolithic run vs the tellurium reference.

    ``seed`` is accepted for a stable, explicit call signature; the monolithic
    ODE integrations are deterministic, so it does not perturb the result.
    Returns ``{"reference", "species", "agreement": {engine: max_rel_err | None},
    "aborted": [engine, ...]}``.
    """
    n_steps = max(1, int(n_points) - 1)
    ref_traj, ref_t = run_engine_monolithic(
        model=antimony, engine=REFERENCE, n_steps=n_steps, t_end=t_end)

    species = sorted(ref_traj)
    agreement: dict[str, float | None] = {}
    aborted: list[str] = []
    for engine in engines:
        if engine == REFERENCE:
            agreement[REFERENCE] = 0.0
            continue
        try:
            traj, t = run_engine_monolithic(
                model=antimony, engine=engine, n_steps=n_steps, t_end=t_end)
            agreement[engine] = compare_to_reference(ref_traj, ref_t, traj, t).max_rel_err
        except Exception:
            # COPASI permanent abort, or native on a non-glycolysis model, etc.
            aborted.append(engine)
            agreement[engine] = None
    return {"reference": REFERENCE, "species": species,
            "agreement": agreement, "aborted": aborted}
