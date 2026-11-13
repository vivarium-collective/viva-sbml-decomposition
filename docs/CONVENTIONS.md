# Sim-artifact conventions

Every study in this workspace commits a **seeded, reproducible artifact** so a
reviewer can re-run and regenerate it. There are two artifact classes, chosen by
how the study is graded — not two standards of rigor.

## 1. Emitter-backed studies → committed `runs.db`

The **decomposition** investigation (`sbml-decomposition`) runs real simulations
through an emitter. Each such study commits a seeded
`workspace/investigations/sbml-decomposition/studies/<slug>/runs.db` — the
SQLite run store the workbench Simulations DB and the published galleries read.
Re-running the study with the committed seeds regenerates the same `runs.db`.

## 2. Code-graded studies → committed `computed_outcomes`

The **recovery** investigation (`discovery-sbml-recovery`) is graded by the
`viva_sbml_decomposition.evaluators` against a hidden oracle. The recovery
composite strips the emitter (there is no emitted time-series to store), so the
reproducible artifact is the seeded `computed_outcomes` block on each study's
canonical run (committed in `study.yaml`), together with the rendered
`report.html`. Re-running `scripts/recover.py <slug>` then
`scripts/grade_recovery.py <slug>` regenerates the same graded decision
(the raw fit floats are not bit-reproducible across BLAS/platforms, but the
thresholded PASS/FAIL verdict is — see Phase 0).

## Synthesis studies (exempt from both)

Each investigation has one **synthesis** study — `decomposition-synthesis` and
`recoverability-frontier` — that aggregates its sibling studies into a landscape
/ frontier and runs no new simulation of its own. These commit neither a
`runs.db` nor `computed_outcomes`; their artifact is the rendered synthesis
figure/report.

## Enforcement

`tests/test_artifact_convention.py` pins all three cases so a committed artifact
cannot silently vanish: every emitter-backed decomposition study has a `runs.db`,
every recovery study with a run has `computed_outcomes`, and the synthesis
exemption is allowed only for a study that genuinely has no own graded/emitted
run (so a real study cannot hide a missing `runs.db` by being named a synthesis).
