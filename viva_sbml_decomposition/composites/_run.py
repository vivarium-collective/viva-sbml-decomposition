"""Shared run/extract helper: run a built document and read its RAM-emitter history."""
from __future__ import annotations

import numpy as np
from process_bigraph import Composite


def extract_run(doc, core, t_end, species, *, emitter_node="emitter"):
    """Run ``doc`` (must contain a RAM emitter node) for ``t_end``; return ``(traj, t)``.

    ``traj`` is ``{species: ndarray}`` for ``species``; ``t`` the emitted time array.
    Raises ``KeyError``/``ValueError`` loudly if the emitter node, its history,
    or any species is missing.
    """
    comp = Composite({"state": doc}, core=core)
    comp.run(t_end)
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
