from pathlib import Path

import pytest
import yaml

from viva_sbml_decomposition.models.registry import MODELS, all_models, get_model

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "viva_sbml_decomposition/data/vg_data/structured_resources.yaml"
IDS = {"glycolysis", "multiorgan_scaled", "lotka_volterra"}


def test_registry_ids_and_fields():
    assert {m["id"] for m in all_models()} == IDS
    for m in all_models():
        assert m["antimony"].strip()
        assert m["n_species"] > 0 and m["n_reactions"] > 0
        assert get_model(m["id"]) is MODELS[m["id"]]


def test_counts_match_roadrunner():
    te = pytest.importorskip("tellurium")
    for m in all_models():
        r = te.loada(m["antimony"])
        assert len(r.getFloatingSpeciesIds()) == m["n_species"]
        assert r.getNumReactions() == m["n_reactions"]


def test_manifest_yaml_matches_registry():
    raw = yaml.safe_load(MANIFEST.read_text())
    entries = [r for r in raw["resources"] if r["type"] == "sbml-model"]
    assert len(entries) == 3
    for e in entries:
        assert e["antimony"].strip() == MODELS[e["id"]]["antimony"].strip()
        assert e["used_by"]


def test_workbench_loader():
    sr = pytest.importorskip("vivarium_workbench.lib.structured_resources")
    payload = sr.load_structured_resources(ROOT)
    assert payload is not None
    group = payload["groups"]["sbml-model"]
    assert len(group) == 3
    for e in group:
        assert e["antimony"].strip() == MODELS[e["id"]]["antimony"].strip()
