"""Global sensitivity / UQ harness (Phase 4).

Seeded Morris elementary-effects screening over the glycolysis Vmax parameters,
run through the Ray ensemble (serial fallback), ranking parameters by importance.
Morris on a deterministic ODE is reproducible given the seed, so these tests
assert structure + reproducibility (same seed -> same indices) and that the
ranking is well-formed — never a raw magnitude to many digits.
"""


def test_morris_shape_and_determinism():
    from viva_sbml_decomposition.harness.sensitivity import morris_sensitivity

    a = morris_sensitivity(r=3, levels=4, span=0.2, seed=0)
    b = morris_sensitivity(r=3, levels=4, span=0.2, seed=0)
    assert a["params"], "no parameters"
    assert len(a["mu_star"]) == len(a["params"]) == len(a["sigma"])
    assert a["ranking"][0] in a["params"]
    assert set(a["ranking"]) == set(a["params"])
    assert a["mu_star"] == b["mu_star"]            # seeded -> reproducible
    assert a["backend"] in ("ray", "serial")
    assert all(m >= 0 for m in a["mu_star"])       # mu* = mean|EE| >= 0


def test_morris_serial_matches_on_indices(monkeypatch):
    # Parallelism is a speed detail — Ray forced off must give the SAME indices.
    from viva_sbml_decomposition.harness.sensitivity import morris_sensitivity
    import viva_sbml_decomposition.composites.ensemble as ens

    ray_run = morris_sensitivity(r=2, levels=4, span=0.2, seed=1)
    monkeypatch.setattr(ens, "_try_ray", lambda: None)
    serial_run = morris_sensitivity(r=2, levels=4, span=0.2, seed=1)
    assert serial_run["backend"] == "serial"
    assert serial_run["mu_star"] == ray_run["mu_star"]
