# Task C2 report: generic reaction_split

New: `composites/reaction_split.py` (`reaction_split`, `build_document`, `run_reaction_split`), registered in `composites/__init__.py`; tests in `tests/test_reaction_split.py`.
Built on `engine_node(engine, model, reactions=..., species_path=("species",))` for both nodes; one shared additive `["species"]` store seeded from the model; `EnergyStep` added only when the model has assignment rules; `config.model` carried on both nodes. Seams: `aldolase`, `mechanical`/`operator`, explicit `{"upper","lower"}`.

## A/B max-abs-diffs (full trajectory, all species x all timepoints)
- aldolase tell+tell vs arm3: 0.0
- aldolase tell+native vs arm6: 0.0
- aldolase tell+copasi vs arm10 cross (subprocess, slow-marked): 0.0 (bound asserted < 1e-6)
- mechanical tell+tell vs arm2: NOT bit-identical. dt=0.05: 0.662, 0.01: 0.186, 0.005: 0.095, 0.001: 0.029.

## Arm 2 (closest-faithful)
Arm 2 is one bespoke `OperatorSplitProcess` (sequential Lie split, overwrite coupling). reaction_split is two nodes with additive (Jacobi) coupling over the same partition (core vs vGLYCO/vTreha). Same first-order error order, different constant, so the difference is O(dt) (first-order convergence). Test asserts <1.0 at 0.05, <0.15 at 0.005 and a >3x shrink. Reproducing Lie exactly would need a sequential operator, outside the delta-node path.

## _engine.py notes (for C3)
Nothing fought me. Minor: `engine_node` subset nodes carry no `stores`/emit info (only monolith does), so the composite must seed the store and build the emitter itself; `tellurium_species_order` instantiates a TelluriumProcess each call; `run_in_subprocess_with_retry` kwargs must be JSON (fine for seam dicts/strings). `resolve_model`/`T_END_BY_MODEL` live in engine_monolithic, imported from there.
