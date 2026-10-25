"""Arm 6: the COUPLING DEMONSTRATION — a composition the monolith can't express.

Arm 6 takes arm 3's modular decomposition and swaps ONE submodel node (lower
glycolysis) for a fully NATIVE process-bigraph process computing that submodel's
reactions, while the OTHER submodel (upper glycolysis) stays a
Tellurium/libroadrunner :class:`SubmodelProcess`. Both share the same ``species``
store and reconcile additively through the bigraph apply path. You cannot mix a
hand-written native process and an SBML submodel inside one libroadrunner model —
so this is a real composition the monolithic arm 1 structurally cannot represent.

Because mixing engines changes the numerics, arm 6 has NO monolithic reference
(that is the POINT). The tests therefore assert only that it BUILDS, RUNS, and
EMITS finite, sensible output for the full species set — and that its record is
marked accuracy-exempt so the Task-12 figure keeps it off the accuracy axis.
"""
from __future__ import annotations

import numpy as np

from viva_sbml_decomposition.composites import arm6_coupling
from viva_sbml_decomposition.harness.composability import score_arm
from viva_sbml_decomposition.models.glycolysis import GLYCOLYSIS_META


def test_arm6_runs_a_coupling_the_monolith_cannot():
    names = GLYCOLYSIS_META["species"]
    traj, t = arm6_coupling.run_arm()

    # The full 17-species set round-trips through the shared store by name.
    assert set(traj) == set(names) and len(names) == 17
    assert t[0] == 0.0 and np.isclose(t[-1], GLYCOLYSIS_META["t_end"])

    # Finite + sensible: every species is finite, bounded, and non-negative to a
    # small tolerance over the whole run (NOT matched against a reference — arm 6
    # has none by construction).
    for s in names:
        v = traj[s]
        assert v.shape == t.shape
        assert np.all(np.isfinite(v)), f"{s} produced non-finite values"
        assert np.all(v > -1e-6), f"{s} went substantially negative"
        assert np.all(v < 1e4), f"{s} blew up (unbounded)"

    # Arm 6 carries NO accuracy (no monolithic reference) — accuracy-exempt.
    s6 = score_arm("arm6")
    assert s6.accuracy is None and s6.accuracy_exempt is True


def test_arm6_is_a_genuine_cross_engine_composition():
    # Prove it is NOT arm 3 relabeled: one node is a Tellurium SubmodelProcess
    # (SBML/libroadrunner) and a DIFFERENT node is a native pbg process, both
    # wired to the ONE shared species store.
    doc = arm6_coupling.glycolysis_coupled()

    addresses = {
        k: v["address"]
        for k, v in doc.items()
        if isinstance(v, dict) and "address" in v
    }
    assert any(a.endswith("SubmodelProcess") for a in addresses.values()), (
        f"no Tellurium SubmodelProcess node present: {addresses}"
    )
    assert any("NativeSubmodel" in a for a in addresses.values()), (
        f"no native pbg submodel node present: {addresses}"
    )

    # Both submodel nodes share the single species store (couple only through it).
    proc_nodes = [
        v for v in doc.values()
        if isinstance(v, dict) and v.get("_type") == "process"
    ]
    assert len(proc_nodes) >= 2
    for node in proc_nodes:
        assert node["inputs"]["species"] == ["species"]
        assert node["outputs"]["species"] == ["species"]

    # A single shared species store seeded with the full species set.
    assert "species" in doc and isinstance(doc["species"], dict)
    assert set(doc["species"]) == set(GLYCOLYSIS_META["species"])

    # Emitter stripped for the workbench generator.
    assert not any(k == "emitter" or k.startswith("emitter_") for k in doc)
