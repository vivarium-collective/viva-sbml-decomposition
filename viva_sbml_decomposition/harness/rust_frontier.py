"""Best-effort probe of the viva-core Rust engine on the one portable arm.

`viva_core.run(spec, n_ticks)` executes a `.pbg` document in the Rust engine, but
only for processes registered in viva-core's *native* Rust registry. Our composites
are Python processes (`local:` addresses), so this probe's honest expected outcome
is that viva-core cannot run `native_composite` without a Rust-side process port —
and "the engine can't run our composites without a port" is itself the frontier
finding. If the arm *does* run, we time it against the Python native monolith and
record the ODE speedup. Either way the result is a structured dict; the probe never
raises. The cited dispatch and viva-hybrid figures are attributed, not re-measured
here (see the notes); only the ODE/native-composite attempt is measured in-workspace.

Run via `scripts/probe_rust_frontier.py`, which commits the result as
`visualizations/rust_frontier.json`. Not run in CI (no Rust toolchain).
"""
from __future__ import annotations

import os
import platform
import sys
import time

_DEFAULT_VIVA_CORE_PY = os.path.expanduser("~/code/viva-core/python")

# Attributed context (viva-core's own scalar-dispatch benchmark; viva-hybrid's
# published negative result). Cited, not re-measured in this workspace.
_DISPATCH_NOTE = ("viva-core accelerates bigraph *dispatch* ~1.27x on scalar state "
                  "(viva-core's reported benchmark; cited, not re-measured here) and "
                  "regresses on array-heavy state today")
_ODE_NOTE = ("viva-core does not accelerate ODE integration: it runs only processes in "
             "its native Rust registry, so a tellurium/COPASI/Python composite needs a "
             "Rust-side process port before it can run at all")
_VIVA_HYBRID_NOTE = ("viva-hybrid is a rigorous negative result: ~2x over numba, with no "
                     "regime that is both >=5x and faithful to exact SSA")
_CAVEAT = ("Best-effort spike, not CI-gated (no Rust toolchain in the pipeline). The "
           "dispatch (~1.27x) and viva-hybrid (~2x) figures are cited from those "
           "projects' own benchmarks; only the native-composite ODE attempt is measured "
           "in this workspace.")


def _base_result() -> dict:
    return {
        "viva_core_available": False,
        "native_composite_runs": False,
        "ode_speedup": None,
        "dispatch_speedup": 1.27,      # cited (see _DISPATCH_NOTE / _CAVEAT)
        "dispatch_note": _DISPATCH_NOTE,
        "ode_note": _ODE_NOTE,
        "viva_hybrid_note": _VIVA_HYBRID_NOTE,
        "requires_port": True,
        "caveat": _CAVEAT,
        "machine": platform.platform(),
    }


def rust_frontier_probe(*, viva_core_python: str | None = None, n_steps: int = 50) -> dict:
    """Attempt to run `native_composite` through viva-core. Never raises."""
    result = _base_result()
    path = viva_core_python or _DEFAULT_VIVA_CORE_PY
    try:
        if path and os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)
        import viva_core  # noqa: F401
    except Exception as exc:  # import / ABI / missing build — honest "not available"
        result["ode_note"] = f"{_ODE_NOTE} (viva_core import failed: {type(exc).__name__})"
        return result

    result["viva_core_available"] = True
    try:
        import json

        from viva_sbml_decomposition.composites.engine_monolithic import (
            _run_inproc, build_document)

        doc = build_document(model="glycolysis", engine="native", n_steps=n_steps)
        spec = json.dumps(doc)

        t0 = time.perf_counter()
        viva_core.run(spec, n_steps)                      # Rust engine attempt
        rust_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        _run_inproc("glycolysis", "native", n_steps, None)  # Python native monolith
        py_s = time.perf_counter() - t1

        result["native_composite_runs"] = True
        result["requires_port"] = False
        result["ode_speedup"] = float(py_s / rust_s) if rust_s > 0 else None
        result["ode_note"] = ("viva-core ran the native composite; measured ODE speedup "
                              "vs the Python native monolith (see ode_speedup)")
    except Exception as exc:
        # Expected: the native process is not in viva-core's Rust registry.
        result["native_composite_runs"] = False
        result["requires_port"] = True
        result["ode_note"] = (f"{_ODE_NOTE} (viva_core.run rejected the composite: "
                              f"{type(exc).__name__}: {str(exc)[:160]})")
    return result
