"""Golden-reference helpers for the generic-composite regression tests.

Each golden is a small inline dict captured from the generic composite's own output
(the output the A/B tests proved identical to the retired per-arm composites):
the final value of a few representative species, the number of emitted time
points, and a checksum (the sum of every species value over every time point).
A change that perturbs any species at any point moves the checksum.
"""
from __future__ import annotations

import numpy as np


def summarize(traj, keys):
    """The ``(final, checksum, n)`` summary of a ``(traj, t)``-style trajectory dict."""
    return {"final": {k: float(traj[k][-1]) for k in keys},
            "checksum": float(sum(float(np.sum(v)) for v in traj.values()))}


def assert_golden(result, golden, *, rtol=1e-7):
    """Assert ``result == (traj, t)`` matches ``golden``.

    ``golden`` = ``{"final": {species: value}, "checksum": float, "n": n_points}``.
    """
    traj, t = result
    assert len(t) == golden["n"], f"{len(t)} time points, golden has {golden['n']}"
    for k, v in golden["final"].items():
        np.testing.assert_allclose(traj[k][-1], v, rtol=rtol, atol=1e-12, err_msg=k)
    got = sum(float(np.sum(v)) for v in traj.values())
    np.testing.assert_allclose(got, golden["checksum"], rtol=rtol, err_msg="checksum")
