"""Shared run/extract helper: run a built document and read its RAM-emitter history."""
from __future__ import annotations

import copy

import numpy as np
from process_bigraph import Composite

from ..harness.timing import run_timing_from_composite


def _read_history(comp, species, emitter_node, species_order=None):
    node = comp.state.get(emitter_node)
    if not isinstance(node, dict) or "instance" not in node:
        raise KeyError(f"emitter node {emitter_node!r} not found in composite state "
                       f"(nodes: {sorted(comp.state)}); was it stripped?")
    rows = getattr(node["instance"], "history", None)
    if not rows:
        raise ValueError(f"emitter {emitter_node!r} recorded no history")
    t = np.array([r["time"] for r in rows], dtype=float)
    traj = {}
    if species_order is not None:
        # Bulk-port arms emit ``species`` as a single ordered numpy Array leaf
        # (the roadrunner getFloatingSpeciesConcentrations order) rather than a
        # string-keyed ``map[float]``. Map each requested species back to its
        # index in ``species_order`` and slice the array column out of each row.
        n = len(species_order)
        index = {s: i for i, s in enumerate(species_order)}
        for s in species:
            if s not in index:
                raise KeyError(f"species {s!r} not in emitted species_order {species_order!r}")
        bad = [i for i, r in enumerate(rows)
               if np.asarray(r.get("species")).shape != (n,)]
        if bad:
            raise ValueError(
                f"emitted species array has wrong shape in row {bad[0]} "
                f"(expected ({n},)); was the bulk species port wired correctly?")
        for s in species:
            i = index[s]
            traj[s] = np.array([np.asarray(r["species"])[i] for r in rows], dtype=float)
        return traj, t
    for s in species:
        missing = [i for i, r in enumerate(rows) if s not in r.get("species", {})]
        if missing:
            raise KeyError(f"species {s!r} missing from emitted rows (e.g. row {missing[0]})")
        traj[s] = np.array([r["species"][s] for r in rows], dtype=float)
    return traj, t


def extract_run(doc, core, t_end, species, *, emitter_node="emitter", species_order=None):
    """Run ``doc`` (must contain a RAM emitter node) for ``t_end``; return ``(traj, t)``.

    ``traj`` is ``{species: ndarray}`` for ``species``; ``t`` the emitted time array.
    Raises ``KeyError``/``ValueError`` loudly if the emitter node, its history,
    or any species is missing.

    ``species_order`` (optional) names the index order of the emitted ``species``
    Array leaf for bulk-port arms (arm 1 with ``bulk_species=True``); when given,
    the emitted ``species`` is read as an ``array[(n),float]`` and sliced per
    species. Left ``None`` (the default) it is read as the string-keyed
    ``map[float]`` every other arm emits.
    """
    comp = Composite({"state": doc}, core=core)
    comp.run(t_end)
    return _read_history(comp, species, emitter_node, species_order=species_order)


def extract_run_timed(doc, core, t_end, species, *, emitter_node="emitter",
                      detail=True, clean_headline=False, species_order=None):
    """Like :func:`extract_run` but also return a :class:`RunTiming`: ``(traj, t, timing)``.

    ``detail=True`` sets ``composite._profile_per_process = True`` before ``run`` so
    ``timing.per_process`` is populated (the in-app equivalent is the emitter
    ``detail='timing'`` config). Detail perturbs ``total`` by ~1.1%, so a single
    detail-ON run is acceptable for the headline.

    ``clean_headline=True`` (with ``detail=True``) does one extra detail-OFF run on a
    deep copy of ``doc``: the headline total/process/framework/overhead_fraction come
    from that clean run, ``per_process`` from the detail-ON run. Costs 2x runtime.
    """
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
    traj, t = _read_history(comp, species, emitter_node, species_order=species_order)
    return traj, t, timing
