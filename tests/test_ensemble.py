"""A/B: the generic ``ensemble`` reproduces arm 5 (same-rate) and arm 7 (multi-rate)."""
from __future__ import annotations

import numpy as np
import pytest
from process_bigraph import Composite, allocate_core
from viva_tellurium.composites import register_tellurium

from viva_sbml_decomposition.composites import abm, abm_meanfield, arm5_ensemble, arm7_multirate, arm9_abm
from viva_sbml_decomposition.composites import ensemble as ens
from viva_sbml_decomposition.composites._engine import tellurium_species_order
from viva_sbml_decomposition.models.glycolysis import GLYCOLYSIS_ANTIMONY, GLYCOLYSIS_META

TOL = 1e-9
N = 3


def _draws(n):
    # identical, seeded draws: scale two rate constants per instance
    rng = np.random.default_rng(0)
    params = ens_params()
    return [{p: v * float(rng.uniform(0.9, 1.1)) for p, v in params.items()} for _ in range(n)]


def ens_params():
    import tellurium as te
    r = te.loada(GLYCOLYSIS_ANTIMONY)
    ids = list(r.getGlobalParameterIds())[:2]
    return {p: float(r[p]) for p in ids}


def test_registered_and_carries_model():
    doc = ens.ensemble(n_instances=2)
    assert doc["inst_0"]["config"]["model"].strip() == GLYCOLYSIS_ANTIMONY.strip()
    assert not any(k.startswith("emitter") for k in doc)
    assert abm is arm9_abm.predator_prey_abm and abm_meanfield is arm9_abm.predator_prey_meanfield


def test_same_rate_matches_arm5():
    draws = _draws(N)
    a = ens.run_ensemble("glycolysis", "tellurium", N, param_draws=draws)
    b = arm5_ensemble.glycolysis_ensemble(n_instances=N, param_draws=draws)
    d = 0.0
    for x, y in zip(a, b):
        assert set(x) == set(y)
        d = max(d, max(abs(x[s] - y[s]) for s in x))
    print("A/B arm5 final-state max|diff| =", d, "backend", a.backend, b.backend)
    assert d < TOL


def test_same_rate_full_trajectories_match_arm5():
    from viva_sbml_decomposition.composites.arm1_monolithic import run_arm
    draws = _draws(N)
    a = ens.run_ensemble("glycolysis", "tellurium", N, param_draws=draws, full=True)
    d = 0.0
    for (traj, t), dr in zip(a, draws):
        ref, rt = run_arm(parameter_overrides=dr)
        d = max(d, float(np.max(np.abs(t - rt))),
                max(float(np.max(np.abs(traj[s] - ref[s]))) for s in ref))
    print("A/B arm5 full-trajectory max|diff| =", d)
    assert d < TOL


@pytest.mark.slow
def test_speedup_measurement_runs():
    r = ens.ensemble_speedup("glycolysis", "tellurium", 4, repeats=1)
    print("speedup:", r)
    assert r.serial_s > 0 and r.n_instances == 4
    assert r.backend in ("ray", "serial")
    if r.speedup_claimed:
        assert r.speedup > 0 and r.parallel_s > 0


def _arm7_with_emitters(n, horizon):
    doc = arm7_multirate.build_document(n, horizon)
    for i in range(n):
        doc[f"emitter_{i}"] = {
            "_type": "step", "address": "local:ram-emitter",
            "config": {"emit": {"species": "array[(17),float]", "time": "float"}},
            "inputs": {"species": ["stores", f"s{i}"], "time": ["global_time"]}}
    return doc


def test_multirate_matches_arm7_heap():
    if "scheduler" not in Composite.config_schema:
        pytest.skip("process-bigraph without #270 (no 'scheduler' config field)")
    n, horizon = 4, 10.0
    core = allocate_core()
    register_tellurium(core)
    comp = Composite({"state": _arm7_with_emitters(n, horizon), "scheduler": "heap"}, core=core)
    comp.run(horizon)
    order = list(tellurium_species_order(GLYCOLYSIS_ANTIMONY))
    got = ens.run_multirate("glycolysis", "tellurium", n, "geometric", horizon=horizon,
                            scheduler="heap")
    assert ens._resolve_intervals("geometric", n, horizon) == arm7_multirate.geometric_intervals(n, horizon)
    d = 0.0
    for i, (traj, t) in enumerate(got):
        rows = comp.state[f"emitter_{i}"]["instance"].history
        rt = np.array([r["time"] for r in rows], dtype=float)
        assert t.shape == rt.shape
        d = max(d, float(np.max(np.abs(t - rt))))
        for k, s in enumerate(order):
            ref = np.array([np.asarray(r["species"])[k] for r in rows], dtype=float)
            d = max(d, float(np.max(np.abs(traj[s] - ref))))
    print("A/B arm7 (heap) full-trajectory max|diff| =", d)
    assert d < TOL


def test_multirate_list_intervals_and_validation():
    doc = ens.build_document(n_instances=2, intervals=[0.5, 2.0])
    assert [doc[f"inst_{i}"]["interval"] for i in range(2)] == [0.5, 2.0]
    with pytest.raises(ValueError):
        ens.build_document(n_instances=2, intervals=[0.5])
    with pytest.raises(ValueError):
        ens.build_document(model="glycolysis", engine="copasi", n_instances=2,
                           param_draws=[{"a": 1.0}, None])
