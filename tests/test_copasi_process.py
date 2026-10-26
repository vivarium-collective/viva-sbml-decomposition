"""CopasiProcess (``processes/copasi_glycolysis.py``) as a SECOND SBML engine.

Three things are under test, all through the generic composites:

1. **COPASI monolith** - ``engine_monolithic(engine='copasi')``: the whole Teusink
   model run by COPASI/basico (LSODA) through process-bigraph, validated against the
   libroadrunner/CVODE monolith. Two different integrators agree to a small non-zero
   residual (measured max-relative-error ~1.29e-7), so ``MONO_TOL`` sits well above
   it and tracks a genuine engine swap, not integrator noise.

2. **Cross-engine coupling** - ``reaction_split(engine_a='tellurium',
   engine_b='copasi')``: the aldolase-seam decomposition with upper glycolysis on a
   tellurium ``SubmodelProcess`` and lower glycolysis on COPASI, both reconciling
   additive deltas through one ``species`` store.

3. **Bulk-SET bit-identity** - the raw-handle fast path ``CopasiProcess`` uses to
   inject state equals basico's per-name ``set_species`` (the guarantee the COPASI
   runtime story rests on).

Tolerances:

* ``MONO_TOL = 1e-3`` - COPASI monolith vs libroadrunner; ~4 orders of headroom.
* ``ENGINE_SWAP_TOL = 1e-3`` - tellurium+COPASI vs tellurium+tellurium at the SAME
  coupling interval; isolates the engine substitution from the (shared) splitting
  error (measured ~3.2e-7).
* ``CROSS_TOL = 0.5`` - cross-engine vs the monolith at the fine interval. The
  aldolase seam cuts the strong phosphate/energy coupling, so additive splitting
  carries a large first-order error that shrinks with the interval: a property of
  the SEAM, not a COPASI defect. Measured: coarse(0.005) ~1.41, fine(0.001) ~0.21.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from viva_sbml_decomposition.composites.engine_monolithic import (
    engine_monolithic, run_engine_monolithic)
from viva_sbml_decomposition.composites.reaction_split import (
    build_document, run_reaction_split)
from viva_sbml_decomposition.harness.accuracy import compare_to_reference
from viva_sbml_decomposition.harness.timing import time_run
from viva_sbml_decomposition.models.glycolysis import GLYCOLYSIS_META

MONO_TOL = 1e-3
ENGINE_SWAP_TOL = 1e-3
CROSS_TOL = 0.5

COARSE_INTERVAL = 0.005   # ~1.41 max rel err (outside CROSS_TOL): domain-safe, fast
FINE_INTERVAL = 0.001     # ~0.21 max rel err (within CROSS_TOL): the convergence point


def _cross(dt):
    return run_reaction_split("glycolysis", "aldolase", "tellurium", "copasi",
                              coupling_interval=dt)


@pytest.fixture(scope="module")
def runs():
    """Compute the runs (except the slow fine interval) once per module."""
    return SimpleNamespace(
        ref=run_engine_monolithic("glycolysis", "tellurium", n_steps=300),
        mono=run_engine_monolithic("glycolysis", "copasi", n_steps=300),
        cross_coarse=_cross(COARSE_INTERVAL),
        tell_coarse=run_reaction_split("glycolysis", "aldolase", "tellurium", "tellurium",
                                       coupling_interval=COARSE_INTERVAL),
    )


# --------------------------------------------------------------------------- #
# COPASI monolith: second engine on the whole model, validated vs. libroadrunner.
# --------------------------------------------------------------------------- #

def test_copasi_mono_runs_through_process_bigraph(runs):
    traj, t = runs.mono
    assert set(traj) == set(GLYCOLYSIS_META["species"])
    assert t[0] == pytest.approx(0.0)
    assert t[-1] == pytest.approx(GLYCOLYSIS_META["t_end"], rel=1e-6)
    for s, arr in traj.items():
        assert arr.shape == t.shape
        assert np.all(np.isfinite(arr)), f"{s} non-finite"


def test_copasi_mono_matches_libroadrunner_reference(runs):
    res = compare_to_reference(*runs.ref, *runs.mono)
    assert not res.diverged
    assert res.max_rel_err < MONO_TOL, (
        f"COPASI-mono max_rel_err={res.max_rel_err:.3e} exceeds {MONO_TOL:.0e}")


def test_copasi_mono_generator_builds_stripped_doc():
    doc = engine_monolithic(model="glycolysis", engine="copasi")
    assert "emitter" not in doc
    assert any(node.get("address") == "local:CopasiProcess"
               for node in doc.values() if isinstance(node, dict))


# --------------------------------------------------------------------------- #
# Cross-engine coupling: Tellurium + COPASI sharing one store.
# --------------------------------------------------------------------------- #

def test_cross_engine_builds_and_runs_through_process_bigraph(runs):
    traj, t = runs.cross_coarse
    assert set(traj) == set(GLYCOLYSIS_META["species"])
    assert np.isclose(t[-1], GLYCOLYSIS_META["t_end"])
    for s, arr in traj.items():
        assert arr.shape == t.shape
        assert np.all(np.isfinite(arr)), f"{s} non-finite"
        assert np.all(arr > -1e-6), f"{s} went substantially negative"


def test_cross_engine_reproduces_tellurium_split_at_same_interval(runs):
    """COPASI-lower reproduces Tellurium-lower at the same interval (clean swap).

    The two composites differ in ONE thing - the lower-half engine - so at a matched
    coupling interval they must agree to machine-noise levels. Fails hard on a
    mis-wired COPASI node.
    """
    res = compare_to_reference(*runs.tell_coarse, *runs.cross_coarse)
    assert not res.diverged
    assert res.max_rel_err < ENGINE_SWAP_TOL, (
        f"COPASI-lower vs Tellurium-lower (same interval) max_rel_err="
        f"{res.max_rel_err:.3e} exceeds {ENGINE_SWAP_TOL:.0e}")


@pytest.mark.slow
def test_cross_engine_converges_to_reference(runs):
    """Two engines sharing one store converge to the monolith as the interval shrinks.

    Monotone trend (fine beats coarse) AND the fine run is within CROSS_TOL while the
    coarse run is not - a gate that asserts convergence rather than passing
    vacuously. Slow: the fine run is 0.001 = 30k steps (~85 s).
    """
    ref_traj, ref_t = runs.ref
    coarse = compare_to_reference(ref_traj, ref_t, *runs.cross_coarse)
    fine = compare_to_reference(ref_traj, ref_t, *_cross(FINE_INTERVAL))
    assert not fine.diverged and not coarse.diverged
    assert fine.max_rel_err < coarse.max_rel_err
    assert fine.max_rel_err < CROSS_TOL
    assert coarse.max_rel_err > CROSS_TOL, "the convergence gate is vacuous"


def test_cross_engine_uses_two_distinct_engines():
    doc = build_document("glycolysis", "aldolase", "tellurium", "copasi")
    addresses = {k: v["address"] for k, v in doc.items()
                 if isinstance(v, dict) and "address" in v and v.get("_type") == "process"}
    tell = [k for k, a in addresses.items() if a == "local:SubmodelProcess"]
    cop = [k for k, a in addresses.items() if a == "local:CopasiProcess"]
    assert len(tell) == 1, f"expected one Tellurium node: {addresses}"
    assert len(cop) == 1, f"expected one COPASI node: {addresses}"
    assert tell[0] != cop[0]
    # Both share the single species store seeded with the full species set.
    assert set(doc["species"]) == set(GLYCOLYSIS_META["species"])
    for node in (doc[tell[0]], doc[cop[0]]):
        assert node["inputs"]["species"] == ["species"]
        assert node["outputs"]["species"] == ["species"]


# --------------------------------------------------------------------------- #
# Bulk-SET bit-identity (the guarantee the COPASI runtime story rests on).
# --------------------------------------------------------------------------- #

def test_bulk_set_is_bit_identical_to_basico_set_species():
    """The raw-handle bulk SET == basico's per-name set_species, and actually writes.

    Injecting one perturbed state via the hot-loop raw path and via
    ``basico.set_species(exact=True, initial_concentration=...)`` must give equal
    post-step trajectories - the pinned fast-path guarantee - AND both must differ
    clearly from the unset baseline, so a silent no-op SET cannot masquerade as a
    pass. Run through the retrying subprocess (loads COPASI).
    """
    from viva_sbml_decomposition.processes.copasi_checks import copasi_set_bit_identity

    r = copasi_set_bit_identity()
    # SET-level: the two paths write the IDENTICAL initial concentration (no
    # integration), the pure bit-identity guarantee.
    assert r["set_level_max_abs_diff"] < 1e-12, (
        f"raw-handle SET writes a different initial concentration than "
        f"basico.set_species (max |d| = {r['set_level_max_abs_diff']:.3e}); "
        f"the fast path is NOT bit-identical")
    # Integration-level: same post-step trajectory to within CVODE's own rtol (1e-8);
    # the machine-eps SET difference amplifies to ~1e-10 over one interval.
    assert r["post_step_max_abs_diff"] < 1e-8, (
        f"post-step divergence {r['post_step_max_abs_diff']:.3e} exceeds the "
        f"integrator tolerance; the SET paths are not equivalent")
    # Not a silent no-op: the perturbed SET visibly moved the result.
    assert r["raw_vs_baseline"] > 1e-3, (
        f"perturbed SET barely moved the result ({r['raw_vs_baseline']:.3e}); "
        f"a silent no-op SET could be passing")


# --------------------------------------------------------------------------- #
# Measurement.
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_measurement_reports_accuracy_and_timing():
    """COPASI-mono accuracy + wall-clock and a finite cross-engine accuracy number."""
    ref = run_engine_monolithic("glycolysis", "tellurium", n_steps=300)
    mono = compare_to_reference(*ref, *run_engine_monolithic("glycolysis", "copasi", n_steps=300))
    timing = time_run(lambda: run_engine_monolithic("glycolysis", "copasi", n_steps=300),
                      repeats=1, warmup=True)
    cross = compare_to_reference(*ref, *_cross(COARSE_INTERVAL))
    assert not mono.diverged and mono.max_rel_err < MONO_TOL
    assert timing.median_s > 0.0
    assert not cross.diverged and np.isfinite(cross.max_rel_err)
