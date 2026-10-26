"""Process-vs-framework timing captured through the shared run path (generic composites)."""
from __future__ import annotations

import numpy as np
import pytest

from viva_sbml_decomposition.composites.engine_monolithic import time_engine_monolithic


def test_timed_run_reports_process_vs_framework_split():
    traj, t, rt = time_engine_monolithic("glycolysis", "tellurium", 20)
    assert all(np.all(np.isfinite(v)) for v in traj.values())
    assert 0.0 <= rt.overhead_fraction <= 1.0
    assert rt.process_s + rt.framework_s == pytest.approx(rt.total_s, rel=1e-6, abs=1e-6)
    assert rt.per_process and all(v >= 0 for v in rt.per_process.values())
    _, _, off = time_engine_monolithic("glycolysis", "tellurium", 20, detail=False)
    assert off.per_process == {}


def test_overhead_fraction_ordering():
    """The metric discriminates engines: the cheap tellurium monolith carries the
    HIGHEST framework share (~0.3); the process-heavy native port ~0.02."""
    _, _, mono = time_engine_monolithic("glycolysis", "tellurium", 200)
    _, _, nat = time_engine_monolithic("glycolysis", "native", 200)
    assert mono.overhead_fraction > nat.overhead_fraction
    assert mono.overhead_fraction / nat.overhead_fraction > 2.0


@pytest.mark.parametrize("leaf", ["scalar", "array"])
def test_native_leaf_variants_are_timeable(leaf):
    traj, _, rt = time_engine_monolithic("glycolysis", "native", 50, native_leaf=leaf)
    assert all(np.all(np.isfinite(v)) for v in traj.values())
    assert 0.0 <= rt.overhead_fraction <= 1.0 and rt.per_process
