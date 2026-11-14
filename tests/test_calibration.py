"""Grid-posterior Bayesian calibration harness (Phase 5).

Infers the posterior over the mass-action rate constants (k1, k2, k2r) from noisy
synthetic data by evaluating the Gaussian likelihood on a grid (exact, no MCMC).
The posterior is deterministic given the data seed, so these tests assert
coverage (the 95% CI contains the truth), a near-truth MAP, normalization, and
reproducibility — never a raw posterior value to many digits.
"""


def test_grid_posterior_recovers_and_is_reproducible():
    from viva_sbml_decomposition.harness.calibration import grid_calibration

    a = grid_calibration(grid_points=9, sigma=0.03, seed=0)
    b = grid_calibration(grid_points=9, sigma=0.03, seed=0)
    for p in a["params"]:
        lo, hi = a["ci95"][p]
        assert lo <= a["true"][p] <= hi                          # 95% CI covers truth
        assert abs(a["map"][p] - a["true"][p]) < 0.5 * a["true"][p]   # MAP near truth
        assert a["marginals"][p] == b["marginals"][p]            # seeded -> reproducible
    first = a["params"][0]
    assert abs(sum(a["marginals"][first]) - 1.0) < 1e-9          # normalized


def test_predictive_band_is_in_observation_space():
    from viva_sbml_decomposition.harness.calibration import grid_calibration

    r = grid_calibration(grid_points=7, sigma=0.05, seed=0)
    pc = r["predictive"]
    assert pc["time"] and all(k in pc["observed"] for k in ("A", "B", "C"))
    one = pc["observed"]["A"]
    assert len(one["lo"]) == len(one["hi"]) == len(pc["time"])
    assert all(lo <= hi for lo, hi in zip(one["lo"], one["hi"]))


def test_calibration_study_is_code_graded_and_passes():
    import pathlib
    import yaml
    sd = pathlib.Path(
        "workspace/investigations/discovery-sbml-recovery/studies/bayesian-calibration/study.yaml")
    d = yaml.safe_load(sd.read_text())
    assert d["investigation"] == "discovery-sbml-recovery"
    canonical = next((r for r in (d.get("runs") or []) if r.get("canonical")), None)
    assert canonical is not None, "study has no canonical run"
    grade = (canonical.get("computed_outcomes") or {}).get("posterior-covers-and-is-informative") or {}
    assert grade.get("result") == "PASS"
    assert grade.get("evaluated_by") == "code"
    assert grade.get("reconcile") == "no_authored"
    assert grade.get("measured_value") is not None and grade["measured_value"] < 0.8
