#!/usr/bin/env python
"""Run the best-effort viva-core spike and commit its machine-tagged result.

Not run in CI (needs the Rust build). Prepends the viva-core Python bindings,
runs the probe, and writes visualizations/rust_frontier.json — the honest
frontier finding the figure + grade read.

Usage:  PYTHONPATH=. python scripts/probe_rust_frontier.py [--viva-core-python PATH]
"""
from __future__ import annotations

import argparse
import json
import pathlib


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--viva-core-python", default=None)
    ap.add_argument("--n-steps", type=int, default=50)
    args = ap.parse_args()

    from viva_sbml_decomposition.harness.rust_frontier import rust_frontier_probe

    r = rust_frontier_probe(viva_core_python=args.viva_core_python, n_steps=args.n_steps)
    out = pathlib.Path("viva_sbml_decomposition/visualizations/rust_frontier.json")
    out.write_text(json.dumps(r, indent=2) + "\n")
    print(f"rust_frontier: available={r['viva_core_available']} "
          f"native_runs={r['native_composite_runs']} requires_port={r['requires_port']} "
          f"ode_speedup={r['ode_speedup']} -> {out.name}")


if __name__ == "__main__":
    main()
