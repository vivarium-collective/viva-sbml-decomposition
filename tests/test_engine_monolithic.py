"""A/B: the generic ``engine_monolithic`` / ``native_composite`` reproduce the arms.

Each test runs the generic composite configured to match an arm and asserts the
emitted trajectory equals the arm's. Same underlying process + same config + same
interval, so the comparison is exact (< 1e-9; measured 0.0).
"""
from __future__ import annotations

import numpy as np
import pytest
from process_bigraph import allocate_core
from viva_tellurium.composites import make_tellurium_document, register_tellurium

from viva_sbml_decomposition.composites import _engine, arm1_monolithic, arm4_native, arm8_multiorgan, arm10_copasi
from viva_sbml_decomposition.composites._run import extract_run
from viva_sbml_decomposition.composites.engine_monolithic import (
    build_document, engine_monolithic, native_composite, run_engine_monolithic,
    run_native_composite)
from viva_sbml_decomposition.models.glycolysis import GLYCOLYSIS_ANTIMONY
from viva_sbml_decomposition.models.registry import get_model

TOL = 1e-9


def max_abs_diff(a, b):
    (traj_a, t_a), (traj_b, t_b) = a, b
    assert set(traj_a) == set(traj_b)
    assert t_a.shape == t_b.shape
    d = float(np.max(np.abs(t_a - t_b)))
    for s in traj_a:
        assert traj_a[s].shape == traj_b[s].shape
        d = max(d, float(np.max(np.abs(traj_a[s] - traj_b[s]))))
    return d


def test_tellurium_glycolysis_matches_arm1():
    d = max_abs_diff(run_engine_monolithic("glycolysis", "tellurium"), arm1_monolithic.run_arm())
    print("A/B arm1 max|diff| =", d)
    assert d < TOL


def test_tellurium_multiorgan_matches_arm8_mono():
    # Arm 8's mono generator document (make_tellurium_document, map[float] species,
    # same tolerances/interval) run through the framework. The generic composite uses
    # the bulk species port, which arm 1 documents as bit-identical to the map path.
    model = get_model("multiorgan_scaled")["antimony"]
    species = arm8_multiorgan.all_species(4)
    assert model == arm8_multiorgan.build_multiorgan_antimony(4)
    doc = make_tellurium_document(
        model=model, model_format="antimony",
        absolute_tolerance=arm8_multiorgan.ABSOLUTE_TOLERANCE,
        relative_tolerance=arm8_multiorgan.RELATIVE_TOLERANCE,
        interval=arm8_multiorgan.T_END / arm8_multiorgan.DEFAULT_N_STEPS)
    core = allocate_core()
    register_tellurium(core)
    arm = extract_run(doc, core, arm8_multiorgan.T_END, species)
    gen = run_engine_monolithic("multiorgan_scaled", "tellurium",
                                n_steps=arm8_multiorgan.DEFAULT_N_STEPS)
    gen = ({s: gen[0][s] for s in species}, gen[1])
    d = max_abs_diff(gen, arm)
    print("A/B arm8 max|diff| =", d)
    assert d < TOL


def test_native_matches_arm4_scalar():
    d = max_abs_diff(run_native_composite("glycolysis"), arm4_native.run_arm(leaf="scalar"))
    print("A/B arm4 max|diff| =", d)
    assert d < TOL


def test_copasi_glycolysis_matches_arm10_mono():
    # Both go through an abort-safe retrying subprocess (COPASI never loads here).
    # Tolerance 1e-6, not 1e-9: the COPASI arm is not bit-reproducible against ITSELF
    # across subprocess runs (measured arm10-vs-arm10 ~5e-8, ~1e-7 against the
    # generic; two generic runs, or a repeat of the same arm, are often exactly 0).
    # Same process/config/interval, so any residual is COPASI's run-to-run noise,
    # three orders below the 1e-3 engine-accuracy gate in test_arm10_copasi.
    d = max_abs_diff(run_engine_monolithic("glycolysis", "copasi"), arm10_copasi.run_copasi_mono())
    print("A/B arm10 max|diff| =", d)
    assert d < 1e-6


# --- structure / helper contracts (no engine run) ---------------------------------

@pytest.mark.parametrize("engine", _engine.ENGINES)
def test_config_model_carries_antimony_for_every_engine(engine):
    doc = engine_monolithic(model="glycolysis", engine=engine)
    (node,) = [v for v in doc.values() if isinstance(v, dict) and v.get("_type") == "process"]
    assert node["config"]["model"] == GLYCOLYSIS_ANTIMONY
    assert "emitter" not in doc


def test_raw_antimony_and_unknown_model():
    doc = build_document(GLYCOLYSIS_ANTIMONY, "tellurium")
    assert doc["tellurium"]["config"]["model"] == GLYCOLYSIS_ANTIMONY
    with pytest.raises(KeyError):
        build_document("nope", "tellurium")


def test_native_rejects_non_glycolysis():
    with pytest.raises(ValueError, match="glycolysis-specific"):
        native_composite(model="multiorgan_scaled")
    with pytest.raises(ValueError, match="glycolysis-specific"):
        _engine.engine_node("native", get_model("lotka_volterra")["antimony"],
                            interval=0.1, reactions=["x"])


def test_subset_role_nodes():
    from viva_sbml_decomposition.processes.native_glycolysis import REACTIONS
    rx = list(REACTIONS[:2])
    for eng, addr in (("tellurium", "SubmodelProcess"), ("copasi", "CopasiProcess"),
                      ("native", "NativeSubmodelProcess")):
        en = _engine.engine_node(eng, GLYCOLYSIS_ANTIMONY, interval=0.01, reactions=rx)
        assert en.node["address"] == f"local:{addr}"
        assert en.node["config"]["reactions"] == rx
        assert en.node["config"]["model"] == GLYCOLYSIS_ANTIMONY
    with pytest.raises(ValueError):
        _engine.engine_node("native", GLYCOLYSIS_ANTIMONY, interval=0.01, reactions=["bogus"])
