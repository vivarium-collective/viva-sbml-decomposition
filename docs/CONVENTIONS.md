# Sim-artifact conventions

Every study in this workspace commits a **seeded, reproducible artifact** so a
reviewer can re-run and regenerate it. There are two artifact classes, chosen by
**how the study is graded — not which investigation it lives in**, and not two
standards of rigor.

## 1. Emitter-backed studies → committed `runs.db`

A study that runs real simulations through an emitter commits a seeded
`studies/<slug>/runs.db` — the SQLite run store the workbench Simulations DB and
the published galleries read. Re-running with the committed seeds regenerates the
same `runs.db`. Most decomposition studies are of this class.

## 2. Code-graded studies → committed `computed_outcomes`

A study graded by the `viva_sbml_decomposition.evaluators` with no emitter
commits the seeded `computed_outcomes` block on its canonical run (in
`study.yaml`) plus the rendered `report.html`, not a `runs.db`. The whole
**recovery** investigation is of this class, and so is the decomposition
investigation's **`cross-engine-validation`** study (its grade is the
engine-agreement computed by `harness/cross_engine.py`, not an emitted
time-series). Re-grading regenerates the same PASS/FAIL decision; raw floats are
not bit-reproducible across BLAS/platforms, but the thresholded verdict is (Phase 0).

Artifact class follows the grading method, so a code-graded study living in the
decomposition investigation correctly commits `computed_outcomes`, not a `runs.db`.

## Synthesis studies (exempt from both)

Each investigation has one **synthesis** study — `decomposition-synthesis` and
`recoverability-frontier` — that aggregates its sibling studies into a landscape
/ frontier and runs no new simulation of its own. These commit neither a
`runs.db` nor `computed_outcomes`; their artifact is the rendered synthesis
figure/report.

## Enforcement

`tests/test_artifact_convention.py` pins the invariant so a committed artifact
cannot silently vanish: every non-synthesis study with a run commits at least one
of the two artifacts (`runs.db` or `computed_outcomes`), across both
investigations. The synthesis exemption is allowed only for a study that
genuinely has no own graded/emitted run (so a real study cannot hide a missing
artifact by being named a synthesis).
