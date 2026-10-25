"""Shared run/extract helper: run a built document and read its RAM-emitter history."""
from __future__ import annotations

import numpy as np
from process_bigraph import Composite

from ..harness.timing import run_timing_from_composite


def _read_history(comp, species, emitter_node):
    node = comp.state.get(emitter_node)
    if not isinstance(node, dict) or "instance" not in node:
        raise KeyError(f"emitter node {emitter_node!r} not found in composite state "
                       f"(nodes: {sorted(comp.state)}); was it stripped?")
    rows = getattr(node["instance"], "history", None)
    if not rows:
        raise ValueError(f"emitter {emitter_node!r} recorded no history")
    t = np.array([r["time"] for r in rows], dtype=float)
    traj = {}
    for s in species:
        missing = [i for i, r in enumerate(rows) if s not in r.get("species", {})]
        if missing:
            raise KeyError(f"species {s!r} missing from emitted rows (e.g. row {missing[0]})")
        traj[s] = np.array([r["species"][s] for r in rows], dtype=float)
    return traj, t


def extract_run(doc, core, t_end, species, *, emitter_node="emitter"):
    """Run ``doc`` (must contain a RAM emitter node) for ``t_end``; return ``(traj, t)``.

    ``traj`` is ``{species: ndarray}`` for ``species``; ``t`` the emitted time array.
    Raises ``KeyError``/``ValueError`` loudly if the emitter node, its history,
    or any species is missing.
    """
    comp = Composite({"state": doc}, core=core)
    comp.run(t_end)
    return _read_history(comp, species, emitter_node)


def extract_run_timed(doc, core, t_end, species, *, emitter_node="emitter",
                      detail=True, clean_headline=False):
    """Like :func:`extract_run` but also return a :class:`RunTiming`: ``(traj, t, timing)``.

    ``detail=True`` sets ``composite._profile_per_process = True`` before ``run`` so
    ``timing.per_process`` is populated (the in-app equivalent is the emitter
    ``detail='timing'`` config). Detail perturbs ``total`` by ~1.1%, so a single
    detail-ON run is acceptable for the headline.

    ``clean_headline=True`` (with ``detail=True``) does one extra detail-OFF run on a
    deep copy of ``doc``: the headline total/process/framework/overhead_fraction come
    from that clean run, ``per_process`` from the detail-ON run. Costs 2x runtime.
    """
    import copy

    clean = None
    if detail and clean_headline:
        c0 = Composite({"state": copy.deepcopy(doc)}, core=core)
        c0._profile_per_process = False
        c0.run(t_end)
        clean = run_timing_from_composite(c0)
    comp = Composite({"state": doc}, core=core)
    comp._profile_per_process = bool(detail)
    comp.run(t_end)
    timing = run_timing_from_composite(comp)
    if clean is not None:
        clean.per_process = timing.per_process
        timing = clean
    traj, t = _read_history(comp, species, emitter_node)
    return traj, t, timing
