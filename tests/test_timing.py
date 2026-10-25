import itertools

import pytest

from viva_sbml_decomposition.harness import timing
from viva_sbml_decomposition.harness.timing import overhead_ratio, time_run


def _fake_clock(monkeypatch, ticks):
    it = iter(ticks)
    monkeypatch.setattr(timing.time, "perf_counter", lambda: next(it))


def test_overhead_ratio_is_arm_over_monolithic():
    assert overhead_ratio(2.0, 1.0) == 2.0
    with pytest.raises(ValueError):
        overhead_ratio(1.0, 0.0)


def test_time_run_separates_setup_from_run(monkeypatch):
    # setup: 0 -> 5 (5s); run: 10 -> 11 (1s)
    _fake_clock(monkeypatch, [0.0, 5.0, 10.0, 11.0])
    calls = []

    def work():
        calls.append("run")

    def setup():
        calls.append("setup")

    res = time_run(work, repeats=1, warmup=False, setup=setup)
    assert res.setup_s == 5.0
    assert res.all_s == [1.0]
    assert calls == ["setup", "run"]


def test_repeats_take_median(monkeypatch):
    # durations 3, 1, 2 -> median 2; warmup call is untimed and discarded
    ticks = itertools.chain.from_iterable([(0.0, 3.0), (10.0, 11.0), (20.0, 22.0)])
    _fake_clock(monkeypatch, ticks)
    n = []
    res = time_run(lambda: n.append(1), repeats=3, warmup=True)
    assert len(n) == 4
    assert res.all_s == [3.0, 1.0, 2.0]
    assert res.median_s == 2.0
    assert res.setup_s == 0.0
