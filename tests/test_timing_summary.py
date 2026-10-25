"""Task 10b: process-vs-framework timing captured through the shared run path."""
from __future__ import annotations

import math

import numpy as np

import pytest

from viva_sbml_decomposition.composites import arm3_modular, arm4_native
from viva_sbml_decomposition.composites._run import extract_run_timed
from viva_sbml_decomposition.composites.arm1_monolithic import (
    DEFAULT_N_STEPS, GLYCOLYSIS_ANTIMONY, _interval)
from viva_sbml_decomposition.models.glycolysis import GLYCOLYSIS_META

from viva_tellurium import make_tellurium_document  # noqa: E402
from viva_tellurium.composites import register_tellurium  # noqa: E402
from process_bigraph import allocate_core  # noqa: E402

def _assert_converged(traj):
    """Honesty guard: a diverged run must fail, never yield a timing."""
    assert all(np.all(np.isfinite(v)) for v in traj.values()), "run diverged (non-finite)"


SP, T_END = GLYCOLYSIS_META["species"], GLYCOLYSIS_META["t_end"]


def _arm1(n_steps=20, detail=True):
    core = allocate_core()
    register_tellurium(core)
    doc = make_tellurium_document(model=GLYCOLYSIS_ANTIMONY, model_format="antimony",
                                  interval=_interval(n_steps))
    return extract_run_timed(doc, core, T_END, SP, detail=detail)


def _arm4(leaf="scalar", n_steps=200):
    core = allocate_core()
    arm4_native.register_native_glycolysis(core)
    return extract_run_timed(arm4_native._build_document(leaf, n_steps), core, T_END, SP)


def test_timing_summary_captured():
    traj, t, rt = _arm1()
    _assert_converged(traj)
    assert set(traj) == set(SP)
    assert 0.0 <= rt.overhead_fraction <= 1.0
    assert math.isclose(rt.process_s + rt.framework_s, rt.total_s, rel_tol=1e-6, abs_tol=1e-6)
    assert rt.per_process and all(v >= 0 for v in rt.per_process.values())
    _, _, off = _arm1(detail=False)
    assert off.per_process == {}


def test_overhead_fraction_ordering():
    """Metric discriminates arms. Measured finding (contradicts the naive expectation
    that decomposed arms carry a higher fraction): ``overhead_fraction`` is
    framework/total, so the monolithic arm (a very cheap CVODE call per cycle) has
    the HIGHEST fraction (~0.3); arm 4 (python RHS + LSODA, process-heavy) ~0.02.
    Arm 3 is excluded: it only converges at coupling_interval ~0.0005 (slow), and
    timing a diverged run is meaningless.
    """
    n = 200
    traj1, _, mono = _arm1(n_steps=n)
    traj4, _, nat = _arm4(n_steps=n)
    _assert_converged(traj1)
    _assert_converged(traj4)
    assert mono.overhead_fraction > nat.overhead_fraction
    assert mono.overhead_fraction / nat.overhead_fraction > 2.0


def test_arm4_scalar_vs_array_capturable():
    ts, _, s = _arm4("scalar")
    ta, _, a = _arm4("array")
    _assert_converged(ts)
    _assert_converged(ta)
    for rt in (s, a):
        assert 0.0 <= rt.overhead_fraction <= 1.0 and rt.per_process
    print(f"arm4 overhead_fraction scalar={s.overhead_fraction:.3f} array={a.overhead_fraction:.3f}")
