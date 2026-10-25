import numpy as np
import pytest
import tellurium as te
from process_bigraph import allocate_core

from viva_sbml_decomposition.composites._run import extract_run
from viva_sbml_decomposition.composites.arm1_monolithic import glycolysis_monolithic, run_arm
from viva_sbml_decomposition.harness.accuracy import compare_to_reference
from viva_sbml_decomposition.models.glycolysis import GLYCOLYSIS_ANTIMONY, GLYCOLYSIS_META

N = 300


def test_arm1_emits_reference_trajectory():
    names = GLYCOLYSIS_META["species"]
    t_end = GLYCOLYSIS_META["t_end"]
    traj, t = run_arm(n_steps=N)
    assert set(traj) == set(names) and len(names) == 17
    np.testing.assert_allclose(t, np.linspace(0, t_end, N + 1), rtol=0, atol=1e-9)
    r = te.loada(GLYCOLYSIS_ANTIMONY)
    r.integrator.absolute_tolerance = 1e-12
    r.integrator.relative_tolerance = 1e-8
    r.timeCourseSelections = ["time"] + [f"[{s}]" for s in names]
    res = np.asarray(r.simulate(0, t_end, N + 1), dtype=float)
    np.testing.assert_allclose(res[:, 0], t, atol=1e-9)
    direct = {s: res[:, i + 1] for i, s in enumerate(names)}
    for s in names:
        assert traj[s].shape == t.shape and np.all(np.isfinite(traj[s]))
        assert np.allclose(traj[s], direct[s], rtol=1e-6, atol=1e-9), s
    assert compare_to_reference(direct, t, traj, t).status == "ok"


def test_stripped_generator_has_no_ram_emitter():
    doc = glycolysis_monolithic()
    assert "tellurium" in doc
    assert not any(k == "emitter" or k.startswith("emitter_") for k in doc)


def test_extract_run_fails_loudly():
    doc = glycolysis_monolithic()  # emitter stripped
    with pytest.raises(KeyError, match="emitter"):
        extract_run(doc, _core(), 1.0, ["ATP"])


def _core():
    from viva_tellurium.composites import register_tellurium
    c = allocate_core()
    register_tellurium(c)
    return c
