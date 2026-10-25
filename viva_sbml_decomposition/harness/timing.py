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
