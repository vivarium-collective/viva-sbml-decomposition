"""Emitter policy: drop the library builder's hardcoded RAM emitter node.

``viva_tellurium.make_tellurium_document`` bakes in a ``local:ram-emitter`` node
so the document runs standalone. Inside the workbench that pins the sink to RAM
(not the workspace's xarray sink) and its address is not registered on the run
core. ``strip_emitter`` removes it so the run path injects the configured sink.
"""
from __future__ import annotations

from typing import Any


def strip_emitter(doc: dict) -> dict:
    """Drop the ``emitter`` / ``emitter_<i>`` nodes from ``doc`` in place.

    Handles flat documents and ``{"state": {...}}``-wrapped ones.
    """
    state: Any = doc.get("state") if isinstance(doc.get("state"), dict) else doc
    for key in [k for k in list(state) if k == "emitter" or k.startswith("emitter_")]:
        state.pop(key, None)
    return doc
