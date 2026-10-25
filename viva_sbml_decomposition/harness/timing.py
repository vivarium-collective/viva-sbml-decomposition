"""Timing harness: report an arm's overhead as a ratio to the monolithic arm."""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional


@dataclass
class TimingResult:
    median_s: float
    all_s: List[float] = field(default_factory=list)
    setup_s: float = 0.0


def time_run(
    callable: Callable[[], object],
    *,
    repeats: int,
    warmup: bool,
    setup: Optional[Callable[[], object]] = None,
) -> TimingResult:
    """Time `callable` `repeats` times; median over repeats.

    `setup` (optional) runs once, timed separately as `setup_s`, before the
    runs. If `warmup`, one extra untimed call is made and discarded.
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    setup_s = 0.0
    if setup is not None:
        t0 = time.perf_counter()
        setup()
        setup_s = time.perf_counter() - t0
    if warmup:
        callable()
    all_s: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        callable()
        all_s.append(time.perf_counter() - t0)
    return TimingResult(median_s=statistics.median(all_s), all_s=all_s, setup_s=setup_s)


def overhead_ratio(arm_median: float, monolithic_median: float) -> float:
    if monolithic_median <= 0:
        raise ValueError("monolithic_median must be > 0")
    return arm_median / monolithic_median


@dataclass
class RunTiming:
    """Process-vs-framework split of one composite run (process-bigraph timing_summary).

    ``framework_s`` is ONE lumped aggregate (view/project/apply/scheduling are not
    sub-bucketed). ``per_process`` maps process path -> seconds; empty unless
    per-process detail was on. Under Ray it collapses to a synthetic
    ``("<batched>", "<RuntimeClass>")`` key (recorded as-is, not faked).
    """

    total_s: float
    process_s: float
    framework_s: float
    overhead_fraction: float
    per_process: dict = field(default_factory=dict)

    def top(self, n: int = 5):
        return sorted(self.per_process.items(), key=lambda kv: kv[1], reverse=True)[:n]


def run_timing_from_composite(comp) -> RunTiming:
    """Read ``comp.timing_summary()`` (call AFTER ``comp.run``)."""
    s = comp.timing_summary()
    total = float(s.total)
    framework = float(s.framework_time)
    return RunTiming(
        total_s=total,
        process_s=float(s.process_time),
        framework_s=framework,
        overhead_fraction=(framework / total) if total > 0 else 0.0,
        per_process=dict(s.per_process),
    )
