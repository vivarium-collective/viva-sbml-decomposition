"""Workspace build hook — the executor for the discovery cycle's capability-gap bridge.

Design (coordinated with the discovery-engine session; mirrors the viz-loop seam):

    cycle.model_architect identifies a capability_gap  (a mechanism no registered
    component provides)  ->  engine emits a `needs_build` frontier item  ->  engine
    calls an INJECTED build hook to fill it  ->  cycle re-catalogs.

The engine-side piece (the `capability_gap` / `needs_build` frontier kind, its
accept-gate, and close-on-fresh) lives in discovery_engine, exactly like the
viz-loop's `propose_visualizations` / `render_hook` / `_close_built_viz_proposals`.
THIS module is the workspace-side executor it injects. Pilot behavior, never
fabricates working behavior:

  1. REUSE   — a registered process already provides the mechanism -> return it.
  2. PRESENT — the mechanism maps to a capability we have already built -> return it.
  3. SCAFFOLD — a genuine gap -> record it in the capability ledger and return
     ``needs_implementation`` (a human/agent builds the real process).

The function contract (``build_capability(gap, ws_root) -> result``) will align to
the engine's finalized injected-hook signature; until then it is callable directly.
"""
from __future__ import annotations

# Capabilities this workspace has already built (mechanism -> implementing process).
# Kept in sync with workspace/notes/capability-ledger.md.
BUILT_CAPABILITIES: dict[str, str] = {
    "synthetic-target-generator": "viva_sbml_decomposition.processes.recovery.SyntheticTargetGenerator",
    "trajectory-comparator": "viva_sbml_decomposition.processes.recovery.TrajectoryComparator",
    "structural-scorer": "viva_sbml_decomposition.processes.recovery.StructuralScorer",
    "parsimony-evaluator": "viva_sbml_decomposition.processes.recovery.StructuralScorer",
    "rate-law-fitter": "viva_sbml_decomposition.processes.recovery.RateLawFitter",
    "reaction-proposer": "viva_sbml_decomposition.processes.recovery.ReactionProposer",
}


def _resolve(path: str):
    import importlib

    mod, _, attr = path.rpartition(".")
    return getattr(importlib.import_module(mod), attr, None)


def build_capability(gap: dict, ws_root=None) -> dict:
    """Fill a capability gap (the injected build-hook the cycle calls).

    ``gap`` carries the required mechanism (``capability`` / ``mechanism``) and a
    content-addressed ``id`` (``cg:<hash>``) so the same gap never re-proposes.
    Returns ``{status, process, detail, gap_id}`` where ``status`` is one of the
    engine's four: ``reuse`` (an existing component matches → engine re-catalogs it)
    | ``built`` (a new process was produced) | ``needs_implementation`` (no component,
    gated to a human/builder — the engine never auto-builds) | ``failed``.

    Pilot: matches the mechanism to an already-built capability (``reuse``), else
    ``needs_implementation`` — it never fabricates a working process.
    """
    gap = gap or {}
    cap = gap.get("capability") or gap.get("mechanism") or ""
    gap_id = gap.get("id")
    path = BUILT_CAPABILITIES.get(cap)
    if path and _resolve(path) is not None:
        return {
            "status": "reuse",
            "capability": cap,
            "process": path,
            "detail": f"{cap} is already provided by {path}",
            "gap_id": gap_id,
        }
    # Genuine gap: do not fabricate a working process. Hand back for authoring.
    return {
        "status": "needs_implementation",
        "capability": cap,
        "process": None,
        "detail": (
            f"no registered process provides {cap!r}; scaffold a Step in "
            "viva_sbml_decomposition/processes/ and add it to the capability ledger"
        ),
        "gap_id": gap_id,
    }


def register_build_hook(registry: dict) -> dict:
    """Symmetry with register_evaluators: expose the build executor to the engine.

    The discovery-engine injects this when it emits capability_gap frontier items.
    """
    registry["build_capability"] = build_capability
    return registry
