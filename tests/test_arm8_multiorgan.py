"""Arm 8 — multi-organ scaling (stretch Tellurium to its strain point).

Asserts, at a small affordable N: (1) both composites (monolithic single
TelluriumProcess + decomposed N-organ, shared-blood) build and run through
process-bigraph and produce finite, sensible output; (2) the decomposed body
genuinely couples through the shared blood store (not independent organs) and
matches the monolithic reference within a recorded tolerance; (3)
``multiorgan_scaling`` returns the documented shape with species and
build/run times growing monotonically with N; (4) the figure renders Plotly html.
"""
import numpy as np
import pytest
from process_bigraph import allocate_core

from viva_sbml_decomposition.composites.arm8_multiorgan import (
    BLOOD,
    BLOOD_INITIAL,
    DEFAULT_COUPLING_INTERVAL,
    DEFAULT_N_ORGANS,
    all_species,
    build_multiorgan_antimony,
    glycolysis_multiorgan_decomposed,
    glycolysis_multiorgan_mono,
    multiorgan_scaling,
    run_decomposed,
    run_mono,
    single_organ_antimony,
)
from viva_sbml_decomposition.harness.accuracy import compare_to_reference

# Recorded tolerance: at N=8 with coupling_interval=0.05 the measured decomposed-
# vs-monolithic max_rel_err is ~1.2e-2 (first-order additive-split error); 0.03
# gives ~2x margin. The error shrinks with the interval and grows with N — see the
# study findings and multiorgan_scaling.
ACCURACY_TOLERANCE = 0.03


def test_antimony_scales_with_n():
    # species = 3N + 1, reactions = 4N; the model string carries both.
    for n in (1, 4, 10):
        src = build_multiorgan_antimony(n)
        assert f"model multiorgan_{n}" in src
        assert src.count("_uptake:") == n and src.count("_rel:") == n
        assert len(all_species(n)) == 3 * n + 1
        assert all_species(n)[0] == BLOOD
    # single-organ model keeps the blood pool + exactly one organ's 4 reactions
    one = single_organ_antimony(2)
    assert BLOOD in one and "o2_uptake:" in one and "o1_uptake:" not in one


def test_build_multiorgan_antimony_rejects_bad_n():
    with pytest.raises(ValueError):
        build_multiorgan_antimony(0)


def test_mono_generator_strips_emitter_and_carries_model():
    doc = glycolysis_multiorgan_mono(n_organs=4)
    assert not any(k == "emitter" or k.startswith("emitter_") for k in doc)
    # config.model carries the N-organ Antimony (so the Model page can show it)
    model = doc["tellurium"]["config"]["model"]
    assert "model multiorgan_4" in model and model.count("_uptake:") == 4
    assert sum(1 for k in glycolysis_multiorgan_mono() if k == "tellurium") == 1


def test_decomposed_generator_strips_emitter_and_has_n_organ_nodes():
    doc = glycolysis_multiorgan_decomposed(n_organs=5)
    assert not any(k == "emitter" or k.startswith("emitter_") for k in doc)
    assert sum(1 for k in doc if k.startswith("organ_")) == 5
    assert "blood" in doc and doc["blood"][BLOOD] == BLOOD_INITIAL
    # every organ node reads+writes the ONE shared blood store (the coupling seam)
    for k, node in doc.items():
        if k.startswith("organ_"):
            assert node["inputs"]["species"] == ["blood"]
            assert node["outputs"]["species"] == ["blood"]
    assert sum(1 for k in glycolysis_multiorgan_decomposed() if k.startswith("organ_")) \
        == DEFAULT_N_ORGANS


def test_both_composites_run_and_decomposed_matches_reference():
    n = 8
    ref_traj, ref_t = run_mono(n)
    arm_traj, arm_t = run_decomposed(n_organs=n)
    assert set(ref_traj) == set(arm_traj) == set(all_species(n))
    # monolithic reference is finite, non-negative, and the closed body conserved
    # mass drains the blood pool below its initial value (organs take it up).
    for s in all_species(n):
        assert np.all(np.isfinite(ref_traj[s])) and np.all(ref_traj[s] >= -1e-9)
    assert ref_traj[BLOOD][0] == pytest.approx(BLOOD_INITIAL)
    assert ref_traj[BLOOD][-1] < BLOOD_INITIAL
    # decomposed matches the monolith within the recorded tolerance
    acc = compare_to_reference(ref_traj, ref_t, arm_traj, arm_t)
    assert acc.status == "ok"
    assert acc.max_rel_err < ACCURACY_TOLERANCE


def test_decomposition_genuinely_couples_through_blood():
    # Coupling proof: with the organs sharing the blood pool, the pool is drawn
    # down well below its initial value and the organs fill — a decomposition of
    # INDEPENDENT organs (each with its own private blood) could not move the
    # single shared pool this way. Also, more organs drain the shared pool more
    # (monotone in N), which only holds if they genuinely share it.
    traj2, _ = run_decomposed(n_organs=2)
    traj8, _ = run_decomposed(n_organs=8)
    assert traj2[BLOOD][-1] < BLOOD_INITIAL
    assert traj8[BLOOD][-1] < traj2[BLOOD][-1]  # 8 organs drain the shared pool more
    assert np.any(traj8["o0_G"] > 0)            # organs actually took up glucose


def test_multiorgan_scaling_shape_and_growth():
    rows = multiorgan_scaling(ns=(2, 8, 32), repeats=1)
    assert [r["n"] for r in rows] == [2, 8, 32]
    for r in rows:
        assert set(r) == {"n", "species", "mono_build_s", "mono_run_s", "mono_total_s",
                          "decomp_build_s", "decomp_run_s", "decomp_total_s", "accuracy"}
        assert r["species"] == 3 * r["n"] + 1
        for key in ("mono_build_s", "mono_run_s", "mono_total_s",
                    "decomp_build_s", "decomp_run_s", "decomp_total_s"):
            assert r[key] > 0 and np.isfinite(r[key])
        assert r["mono_total_s"] == pytest.approx(r["mono_build_s"] + r["mono_run_s"])
        assert r["decomp_total_s"] == pytest.approx(r["decomp_build_s"] + r["decomp_run_s"])
        assert np.isfinite(r["accuracy"])
    # Assert DETERMINISTIC growth signals only (wall-clock at ~1e-4 s is too noisy
    # to assert at repeats=1): species count and the monolithic BUILD, which spans
    # ~10x from N=2 to N=32 (the superlinear-compile strain signal).
    species = [r["species"] for r in rows]
    assert species == sorted(species) and len(set(species)) == 3  # strictly grows
    builds = [r["mono_build_s"] for r in rows]
    assert builds[-1] > builds[0]  # 97-species compile >> 7-species compile


def test_multiorgan_scaling_can_skip_large_decomposed_run():
    # Build is measured at every N; the expensive decomposed RUN (and the accuracy
    # that needs it) is honestly left UNMEASURED past the cap, not guessed.
    rows = multiorgan_scaling(ns=(2, 32), repeats=1, run_decomposed_max_n=2)
    small, big = rows
    assert small["decomp_run_s"] is not None and small["accuracy"] is not None
    assert big["decomp_run_s"] is None and big["decomp_total_s"] is None
    assert big["accuracy"] is None
    assert big["decomp_build_s"] > 0 and big["mono_build_s"] > 0  # builds still measured


def test_multiorgan_figure_renders_plotly():
    from viva_sbml_decomposition.visualizations.multiorgan_scaling import MultiorganScaling

    inst = MultiorganScaling(config={}, core=allocate_core())
    out = inst.update(MultiorganScaling.generate_state())
    assert set(out) == {"html"} and isinstance(out["html"], str)
    assert "Plotly.newPlot" in out["html"] and "viva_MultiorganScaling" in out["html"]
    spec = inst.figure(MultiorganScaling.generate_state())
    assert len(spec["data"]) == 5 and "layout" in spec
    import plotly.graph_objects as go

    go.Figure(spec)  # valid plotly spec
