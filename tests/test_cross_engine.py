"""Cross-engine agreement harness (Phase 2).

The harness runs the same model monolithically in several engines and reports,
per engine, the max relative error of its trajectory against the tellurium
reference. COPASI runs go through the retrying subprocess that
``run_engine_monolithic`` already owns; an engine that cannot run a model (COPASI
permanently aborting, or the glycolysis-specific native process on a different
model) is recorded in ``aborted`` with ``agreement = None``, never raised.
"""
from viva_sbml_decomposition.models.glycolysis import GLYCOLYSIS_ANTIMONY
from viva_sbml_decomposition.models.recovery_targets import MASS_ACTION_ANTIMONY


def test_reference_self_zero_and_native_agreement_on_glycolysis():
    from viva_sbml_decomposition.harness.cross_engine import cross_engine_agreement

    res = cross_engine_agreement(
        GLYCOLYSIS_ANTIMONY, engines=("tellurium", "native"),
        t_end=30.0, n_points=101, seed=0,
    )
    assert res["reference"] == "tellurium"
    assert res["agreement"]["tellurium"] == 0.0          # reference vs itself
    assert res["aborted"] == []
    native = res["agreement"]["native"]
    assert native is not None and native >= 0.0          # native solves glycolysis


def test_unsupported_engine_model_aborts_not_raises():
    from viva_sbml_decomposition.harness.cross_engine import cross_engine_agreement

    # native is glycolysis-specific and raises on any other model; the harness
    # must tolerate that — record the abort, never propagate it.
    res = cross_engine_agreement(
        MASS_ACTION_ANTIMONY, engines=("tellurium", "native"),
        t_end=10.0, n_points=51, seed=0,
    )
    assert res["agreement"]["tellurium"] == 0.0
    assert "native" in res["aborted"]
    assert res["agreement"]["native"] is None


def test_cross_engine_validation_study_is_code_graded_and_passes():
    import pathlib
    import yaml
    sd = pathlib.Path(
        "workspace/investigations/sbml-decomposition/studies/cross-engine-validation/study.yaml")
    d = yaml.safe_load(sd.read_text())
    assert d["investigation"] == "sbml-decomposition"
    canonical = next((r for r in (d.get("runs") or []) if r.get("canonical")), None)
    assert canonical is not None, "study has no canonical run"
    co = canonical.get("computed_outcomes") or {}
    agree = co.get("engines-agree-on-the-same-model") or {}
    # Code-graded (no emitter): computed_outcomes carry the grade, reconcile no_authored.
    assert agree.get("result") == "PASS"
    assert agree.get("evaluated_by") == "code"
    assert agree.get("reconcile") == "no_authored"
    assert agree.get("measured_value") is not None and agree["measured_value"] < 1e-3
