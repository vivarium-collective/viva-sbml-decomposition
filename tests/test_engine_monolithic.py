"""Golden regression tests for the generic ``engine_monolithic`` / ``native_composite``.

The generic composite was A/B-verified against the retired per-engine arms (max|diff|
0.0 for tellurium/native, ~1e-7 for COPASI). Those arm comparisons are replaced by
frozen GOLDEN references captured from that proven output (see ``_golden.py``), plus
the properties the arm test files used to carry: agreement with a direct
libroadrunner run, native-vs-tellurium accuracy, scalar-vs-array leaf equivalence.
"""
from __future__ import annotations

import numpy as np
import pytest
import tellurium as te
from process_bigraph import allocate_core

from _golden import assert_golden
from viva_sbml_decomposition.composites import _engine
from viva_sbml_decomposition.composites._run import extract_run
from viva_sbml_decomposition.composites.engine_monolithic import (
    build_document, engine_monolithic, native_composite, run_engine_monolithic,
    run_native_composite)
from viva_sbml_decomposition.harness.accuracy import compare_to_reference
from viva_sbml_decomposition.models.glycolysis import GLYCOLYSIS_ANTIMONY, GLYCOLYSIS_META
from viva_sbml_decomposition.models.registry import get_model

SPECIES = GLYCOLYSIS_META["species"]

GOLD_TELLURIUM = {
    "final": {"ATP": 2.50839062495427, "F16P": 0.6019076395836988, "PEP": 0.07361684247063688},
    "checksum": 7206.955454419397, "n": 301}
GOLD_NATIVE = {
    "final": {"ATP": 2.508390624954269, "F16P": 0.6019076395836988, "PEP": 0.07361684247063686},
    "checksum": 7206.955462638626, "n": 301}
GOLD_COPASI = {
    "final": {"ATP": 2.508390624954269, "F16P": 0.6019076395836989, "PEP": 0.07361684247063684},
    "checksum": 7206.955463694953, "n": 301}
GOLD_MULTIORGAN = {
    "final": {"Glc_blood": 28.34269793440732, "o0_G": 2.8062041540803966,
              "o3_P": 9.609644347806858},
    "checksum": 20099.999999998625, "n": 201}


def test_tellurium_glycolysis_golden():
    assert_golden(run_engine_monolithic("glycolysis", "tellurium"), GOLD_TELLURIUM)


def test_tellurium_matches_direct_libroadrunner():
    traj, t = run_engine_monolithic("glycolysis", "tellurium", n_steps=300)
    t_end = GLYCOLYSIS_META["t_end"]
    np.testing.assert_allclose(t, np.linspace(0, t_end, 301), rtol=0, atol=1e-9)
    r = te.loada(GLYCOLYSIS_ANTIMONY)
    r.integrator.absolute_tolerance = 1e-12
    r.integrator.relative_tolerance = 1e-8
    r.timeCourseSelections = ["time"] + [f"[{s}]" for s in SPECIES]
    res = np.asarray(r.simulate(0, t_end, 301), dtype=float)
    direct = {s: res[:, i + 1] for i, s in enumerate(SPECIES)}
    assert set(traj) == set(SPECIES) and len(SPECIES) == 17
    for s in SPECIES:
        assert traj[s].shape == t.shape and np.all(np.isfinite(traj[s]))
        assert np.allclose(traj[s], direct[s], rtol=1e-6, atol=1e-9), s
    assert compare_to_reference(direct, t, traj, t).status == "ok"


def test_tellurium_multiorgan_golden():
    # Registry model "multiorgan_scaled" (n=4 organs, 13 species), 200 steps.
    assert_golden(run_engine_monolithic("multiorgan_scaled", "tellurium", n_steps=200),
                  GOLD_MULTIORGAN)


def test_native_glycolysis_golden():
    assert_golden(run_native_composite("glycolysis"), GOLD_NATIVE)


# Native vs the libroadrunner reference: observed ~1e-7 max relative error; 1e-5 is a
# modest margin that would still fail a wrong RHS.
NATIVE_VS_TELLURIUM_T = 1e-5


@pytest.mark.parametrize("leaf", ["scalar", "array"])
def test_native_matches_tellurium_within_tolerance(leaf):
    ref_traj, ref_t = run_engine_monolithic("glycolysis", "tellurium")
    nat_traj, nat_t = _run_native_leaf(leaf)
    assert set(nat_traj) == set(SPECIES) == set(ref_traj)
    res = compare_to_reference(ref_traj, ref_t, nat_traj, nat_t)
    assert not res.diverged
    assert res.max_rel_err < NATIVE_VS_TELLURIUM_T, res.max_rel_err


def _run_native_leaf(leaf, n_steps=300):
    from viva_sbml_decomposition.composites.engine_monolithic import (
        _build, _document, _species_ids, resolve_model)
    from viva_sbml_decomposition.composites._engine import register_engines
    en, t_end = _build("glycolysis", "native", n_steps, None, native_leaf=leaf)
    core = allocate_core()
    register_engines(core)
    _, antimony = resolve_model("glycolysis")
    return extract_run(_document(en), core, t_end, _species_ids(en, antimony))


def test_native_scalar_and_array_leaves_agree():
    a_traj, a_t = _run_native_leaf("scalar")
    b_traj, b_t = _run_native_leaf("array")
    np.testing.assert_allclose(a_t, b_t)
    for name in a_traj:
        a, b = np.asarray(a_traj[name]), np.asarray(b_traj[name])
        scale = max(float(np.max(np.abs(a))), 1e-12)
        assert float(np.max(np.abs(a - b))) / scale < 1e-9, name


def test_copasi_glycolysis_golden():
    # Runs in the abort-safe retrying subprocess (COPASI never loads in-process).
    # rtol 1e-5: COPASI is not bit-reproducible run to run (~5e-8..1e-7 measured).
    assert_golden(run_engine_monolithic("glycolysis", "copasi"), GOLD_COPASI, rtol=1e-5)


def test_stripped_generator_has_no_ram_emitter():
    doc = engine_monolithic()
    assert "tellurium" in doc
    assert not any(k == "emitter" or k.startswith("emitter_") for k in doc)


def test_extract_run_fails_loudly_without_emitter():
    from viva_tellurium.composites import register_tellurium
    core = allocate_core()
    register_tellurium(core)
    with pytest.raises(KeyError, match="emitter"):
        extract_run(engine_monolithic(), core, 1.0, ["ATP"])


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
