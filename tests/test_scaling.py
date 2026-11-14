"""Scaling-benchmarks harness (Phase 3).

Quantifies decomposition scaling along two axes — model size (monolithic build
strain vs species count) and #instances (ensemble wall-clock speedup) — reusing
`multiorgan_scaling` and `ensemble_speedup`. Wall-clock is not cross-machine
reproducible, so these tests assert only structure and robust qualitative
properties (a finite positive build exponent, a known backend), never absolute
seconds or a fixed speedup factor.
"""


def test_scaling_harness_shape_and_exponent():
    from viva_sbml_decomposition.harness.scaling import scaling_benchmarks

    r = scaling_benchmarks(model_sizes=(1, 2, 4), instance_counts=(1, 2), repeats=1, seed=0)
    assert [p["species"] for p in r["model_size"]] == [4, 7, 13]   # 3n + 1
    assert r["build_exponent"] > 0                                 # finite positive fit
    assert r["ensemble"]["backend"] in ("ray", "serial")
    assert len(r["ensemble"]["speedup"]) == 2
    assert r["machine"]                                            # machine tag present


def test_scaling_harness_tolerates_serial(monkeypatch):
    # Ray-unavailable must yield backend="serial" with finite speedups, never raise.
    import viva_sbml_decomposition.composites.ensemble as ens
    monkeypatch.setattr(ens, "_try_ray", lambda: None)
    from viva_sbml_decomposition.harness.scaling import scaling_benchmarks

    r = scaling_benchmarks(model_sizes=(1, 2), instance_counts=(1, 2), repeats=1, seed=0)
    assert r["ensemble"]["backend"] == "serial"
    assert all(s is not None for s in r["ensemble"]["speedup"])


def test_scaling_study_is_code_graded_and_passes():
    import pathlib
    import yaml
    sd = pathlib.Path(
        "workspace/investigations/sbml-decomposition/studies/scaling-benchmarks/study.yaml")
    d = yaml.safe_load(sd.read_text())
    assert d["investigation"] == "sbml-decomposition"
    canonical = next((r for r in (d.get("runs") or []) if r.get("canonical")), None)
    assert canonical is not None, "study has no canonical run"
    co = canonical.get("computed_outcomes") or {}
    for name in ("ensemble-parallel-speedup", "decomposition-cost-characterized"):
        grade = co.get(name) or {}
        assert grade.get("result") == "PASS", f"{name}: {grade.get('result')}"
        assert grade.get("evaluated_by") == "code"
        assert grade.get("reconcile") == "no_authored"
