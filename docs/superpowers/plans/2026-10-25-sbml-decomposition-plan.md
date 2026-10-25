# viva-sbml-decomposition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone viva workspace whose discovery-engine-driven investigation runs one SBML metabolic model across six implementation arms (monolithic Tellurium → fully native process-bigraph, plus ensemble and a coupling demonstration) and quantifies the speed × accuracy × composability tradeoff end-to-end.

**Architecture:** A viva-template standalone workspace (`viva-sbml-decomposition`, package `viva_sbml_decomposition`). Arms 1–3 + 5 wrap the real libroadrunner/CVODE integrator through `viva-tellurium` with inline Antimony; arm 4 is native process-bigraph Processes; arm 6 couples a decomposed submodel to something monolithic SBML can't absorb. A small reusable harness (pure Python, TDD) computes the timing ratio, the accuracy divergence vs the monolithic reference, and the composability score. Studies (one per arm) live under one Investigation whose acceptance criteria + per-study verdicts feed the RFC-0005 discovery-engine crank, which names the next study and signals closure. The 3-way tradeoff is a v2 Plotly Visualization rendered against real xarray/zarr runs.

**Tech Stack:** Python ≥3.11, process-bigraph / bigraph-schema, viva-superpowers (workbench + discovery engine), viva-tellurium (libroadrunner), Ray (ensemble backend), numpy, Plotly, pytest.

**Spec:** `docs/superpowers/specs/2026-10-25-sbml-decomposition-design.md` (copied into the repo in Task 1; source of record is the approved design doc).

## Global Constraints

- Package name is `viva_sbml_decomposition` (viva_ prefix); repo `viva-sbml-decomposition`. (spec: How it gets made)
- Standalone workspace scaffolded from viva-template. (spec)
- No AI attribution in commits/PRs; no `Co-Authored-By: Claude` trailer, no "Generated with Claude Code". (user global instruction — overrides any session attribution reminder)
- All commits/pushes happen in a dedicated git worktree, never in a canonical `~/code/<repo>` checkout. (user global instruction)
- Studies set `runtime.default_emitter: xarray`; workspace sets `runtime.max_generations: 1`; library RAM emitters are stripped (mosaic-compose `_emit.strip_emitter` pattern). (spec)
- Every arm's model source is authored as **inline Antimony** via `make_tellurium_document(model=..., model_format="antimony")` so it renders in the study Model tab. (spec: Goals)
- Every arm integrates the **same** model with matched integrator tolerances and output cadence; only the decomposition changes. (spec: Anti-goals)
- The monolithic arm (arm 1) is the single accuracy reference for arms 2–5; arm 6 is exempt (no monolithic equivalent). (spec: Measurement)
- Performance headline metric = **ratio to monolithic**, arms timed together in one session on one quiesced host (Mac mini). (spec)

## Review Focus

- **Mismatched time grids across arms.** Arms emit at different cadences or stop at different final times; accuracy RMSE is then computed on misaligned samples and is meaningless. Expected: the accuracy harness resamples every arm onto one common grid (the reference's) and refuses/raises on non-overlapping spans. → test in Task 3.
- **Non-finite trajectory values.** A stiff arm diverges or an operator-split step produces NaN/Inf; silently feeding these into RMSE yields NaN and a blank chart. Expected: the accuracy harness detects non-finite values and reports the arm as "diverged", not NaN. → test in Task 3.
- **Ensemble that runs serially.** Arm 5 falls back to serial execution (Ray unavailable) and reports a "speedup" that is really overhead. Expected: the ensemble arm asserts concurrent execution (≥1 speedup over serial on N≥ cores) or explicitly records "serial fallback — no speedup claimed". → test in Task 8.
- **Species name/order drift between arms.** Native PBG (arm 4) or modular (arm 3) arms name or order species differently from the reference, so the comparison silently pairs the wrong columns. Expected: accuracy comparison pairs by species name, and raises on a missing/extra species rather than comparing positionally. → test in Task 3.
- **Composability score read as accuracy-equivalent.** The 3-way figure encodes composability on the same footing as a measured number, implying arm 6 "failed accuracy" (it has none). Expected: arm 6 is shown as a capability callout, excluded from the accuracy axis; the decision table marks its accuracy cell "N/A (no monolithic equivalent)". → test in Task 10 (viz data assembly) + enforced in Task 9.

---

## Task 1: Scaffold the workspace + bring up the workbench

**Files:**
- Create: repo `viva-sbml-decomposition` via `/viva-workspace` (standalone); package `viva_sbml_decomposition/` with `build_core()`.
- Create: `docs/superpowers/specs/2026-10-25-sbml-decomposition-design.md` (copy of approved design).
- Create: `docs/superpowers/plans/2026-10-25-sbml-decomposition-plan.md` (this plan).

**Interfaces:**
- Produces: a workspace root with `workspace.yaml`, an importable `viva_sbml_decomposition` package, `build_core()`; the workbench server reachable (URL under `.pbg/server/`).

- [ ] **Step 1:** Invoke `/viva-workspace` in standalone mode to create `viva-sbml-decomposition` (package `viva_sbml_decomposition`). Rename the scaffold default if it prefixes differently.
- [ ] **Step 2:** Move into a dedicated worktree for all subsequent commits (per global constraint). Set `runtime.max_generations: 1` in `workspace.yaml`.
- [ ] **Step 3:** Install `viva-tellurium` into the workspace venv (`/viva-catalog install viva-tellurium` or editable), and `ray` + `plotly`.
- [ ] **Step 4:** Copy the design doc and this plan into `docs/superpowers/`.
- [ ] **Step 5:** Start the workbench (`/viva-workbench start`); confirm it serves the workspace (`/viva-status`).
- [ ] **Step 6: Commit** the scaffold + docs.

**Deliverable test:** `/viva-status` reports a live workbench over this workspace; `python -c "import viva_sbml_decomposition; viva_sbml_decomposition.build_core()"` succeeds.

---

## Task 2: Choose & validate the metabolic model (monolithic reference)

**Files:**
- Create: `viva_sbml_decomposition/models/glycolysis.py` — the Antimony source string + metadata (BioModels id, provenance) for the chosen metabolic model (Teusink glycolysis or equivalent curated entry).
- Test: `tests/test_models.py`

**Interfaces:**
- Produces: `GLYCOLYSIS_ANTIMONY: str`, `GLYCOLYSIS_META: dict` (`{"name", "biomodels_id", "species": [...], "t_end", "provenance"}`), consumed by every arm.

- [ ] **Step 1: Write the failing test** — `test_glycolysis_integrates_monolithic`: load `GLYCOLYSIS_ANTIMONY` through tellurium/libroadrunner, simulate to `GLYCOLYSIS_META["t_end"]`, assert all `GLYCOLYSIS_META["species"]` present in output and the final state is finite and non-negative.
- [ ] **Step 2:** Run it; expect FAIL (module missing).
- [ ] **Step 3:** Author `glycolysis.py`: pick a curated BioModels metabolic model, obtain its Antimony (via tellurium's SBML→Antimony or BioModels), record provenance. Keep it as a clean inline Antimony string.
- [ ] **Step 4:** Run the test; expect PASS. If the model won't integrate cleanly, pick another curated metabolic entry (noted in spec "Left to the builder").
- [ ] **Step 5: Commit.**

---

## Task 3: Accuracy harness (divergence vs reference)

**Files:**
- Create: `viva_sbml_decomposition/harness/accuracy.py`
- Test: `tests/test_accuracy.py`

**Interfaces:**
- Consumes: trajectories as `{species_name: np.ndarray}` + a time array per arm.
- Produces: `compare_to_reference(ref_traj, ref_t, arm_traj, arm_t) -> AccuracyResult` where `AccuracyResult` has `per_species_rmse: dict`, `max_rel_err: float`, `diverged: bool`, and `status: Literal["ok","diverged"]`.

- [ ] **Step 1: Write failing tests** covering the Review-Focus cases:
  - `test_identical_trajectories_zero_error`: ref vs itself → all RMSE 0, `status=="ok"`.
  - `test_resamples_onto_common_grid`: arm on a coarser time grid is resampled to the reference grid before comparison (assert result finite, not raising).
  - `test_nonfinite_marks_diverged`: an arm containing NaN/Inf → `diverged==True`, `status=="diverged"`, no NaN leaks into `max_rel_err` (it's `inf` or flagged).
  - `test_species_paired_by_name_not_position`: arm with species in different order → correct pairing; missing species → raises `ValueError`.
  - `test_nonoverlapping_spans_raise`: arm time span disjoint from reference → raises `ValueError`.
- [ ] **Step 2:** Run; expect FAIL.
- [ ] **Step 3:** Implement `accuracy.py`: resample by interpolation onto the reference grid, pair by species name, compute per-species RMSE + global max relative error, detect non-finite → diverged. Pure numpy.
- [ ] **Step 4:** Run; expect PASS.
- [ ] **Step 5: Commit.**

---

## Task 4: Timing harness (ratio to monolithic)

**Files:**
- Create: `viva_sbml_decomposition/harness/timing.py`
- Test: `tests/test_timing.py`

**Interfaces:**
- Produces: `time_run(callable, *, repeats:int, warmup:bool) -> TimingResult` (`median_s`, `all_s`, `setup_s`); `overhead_ratio(arm_median:float, monolithic_median:float) -> float`.

- [ ] **Step 1: Write failing tests**: `test_overhead_ratio_is_arm_over_monolithic` (2.0 vs 1.0 → 2.0); `test_time_run_separates_setup_from_run` (a callable with a measurable setup sleep records `setup_s` distinct from run); `test_repeats_take_median` (deterministic timings → median).
- [ ] **Step 2:** Run; expect FAIL.
- [ ] **Step 3:** Implement `timing.py` using `time.perf_counter`, a warmup discard, median over repeats. The host-quiescence concern is procedural (spec: Measurement) — the harness only guarantees ratios + repeats.
- [ ] **Step 4:** Run; expect PASS.
- [ ] **Step 5: Commit.**

---

## Task 5: Arm 1 — monolithic Tellurium composite (the reference)

**Files:**
- Create: `viva_sbml_decomposition/composites/arm1_monolithic.py`
- Create: `viva_sbml_decomposition/composites/_emit.py` (copy the strip_emitter pattern from mosaic-compose)
- Test: `tests/test_arm1_monolithic.py`

**Interfaces:**
- Consumes: `GLYCOLYSIS_ANTIMONY`, `GLYCOLYSIS_META` (Task 2).
- Produces: `@composite_generator`-decorated `glycolysis_monolithic(core=None)` returning a stripped tellurium document; a helper `run_arm(core, n_steps) -> (traj_dict, t_array)` used by every arm's study + the harness.

- [ ] **Step 1: Write failing test** `test_arm1_emits_reference_trajectory`: build + run the composite for `GLYCOLYSIS_META`'s span, assert all species emitted, finite, matches Task 2's direct libroadrunner result within tight tolerance (this arm *is* Tellurium, so it should match near-exactly).
- [ ] **Step 2:** Run; expect FAIL.
- [ ] **Step 3:** Implement the composite with `make_tellurium_document(model=GLYCOLYSIS_ANTIMONY, model_format="antimony", interval=...)` wrapped in `strip_emitter`; `core_extensions=[register_tellurium]`.
- [ ] **Step 4:** Run; expect PASS.
- [ ] **Step 5: Commit.**

---

## Task 6: Arm 2 — operator-split Tellurium (same model, N coupled sub-steps)

**Files:**
- Create: `viva_sbml_decomposition/composites/arm2_operator_split.py`
- Test: `tests/test_arm2_operator_split.py`

**Interfaces:**
- Consumes: Task 2 model, Task 5 `run_arm`, Task 3 accuracy harness.
- Produces: `glycolysis_operator_split(core=None, coupling_interval=...)` — the full model integrated in sub-steps that exchange state through shared stores each vivarium step.

- [ ] **Step 1: Write failing test** `test_arm2_converges_to_reference_as_interval_shrinks`: run at a coarse and a fine coupling interval; assert accuracy-vs-reference (Task 3) improves as the interval shrinks, and the fine interval is within a provisional tolerance `T` (spec: Left to the builder — tolerance is provisional, calibrated here).
- [ ] **Step 2:** Run; expect FAIL.
- [ ] **Step 3:** Implement: a composite that advances the libroadrunner model by `coupling_interval`, writes species to a shared store, re-reads before the next sub-step (operator splitting with matched tolerances per Global Constraints).
- [ ] **Step 4:** Run; expect PASS (tune provisional `T`).
- [ ] **Step 5: Commit.**

---

## Task 7: Arm 3 — modular submodels as coupled processes

**Files:**
- Create: `viva_sbml_decomposition/composites/arm3_modular.py`
- Create: `viva_sbml_decomposition/models/glycolysis_submodels.py` — the Antimony partition (seams chosen per spec "Left to the builder").
- Test: `tests/test_arm3_modular.py`

**Interfaces:**
- Consumes: Task 3 accuracy, Task 5 reference.
- Produces: `glycolysis_modular(core=None)` — several `TelluriumProcess`es (one per submodel) coupled through shared species stores; shared species reconcile additively.

- [ ] **Step 1: Write failing test** `test_arm3_matches_reference_within_tolerance`: run the modular composite; compare to reference (Task 3); assert `status=="ok"` and within provisional `T`; assert species paired by name (Review Focus).
- [ ] **Step 2:** Run; expect FAIL.
- [ ] **Step 3:** Partition the Antimony into biologically-meaningful submodels sharing boundary species; build one tellurium process per submodel; wire shared species through a common store.
- [ ] **Step 4:** Run; expect PASS.
- [ ] **Step 5: Commit.**

---

## Task 8: Arm 5 — ensemble on a real parallel backend

(Arm 4 native-PBG is Task 9; arm 5 is placed here because it reuses only arm 1.)

**Files:**
- Create: `viva_sbml_decomposition/composites/arm5_ensemble.py`
- Test: `tests/test_arm5_ensemble.py`

**Interfaces:**
- Consumes: Task 5 arm-1 composite (the instance), Task 4 timing.
- Produces: `glycolysis_ensemble(core=None, n_instances=..., param_draws=...)` running N monolithic instances concurrently via Ray; `ensemble_speedup(n_instances) -> (parallel_s, serial_s, speedup)`.

- [ ] **Step 1: Write failing tests**: `test_ensemble_runs_all_instances` (N instances → N result sets, each finite); `test_ensemble_parallel_or_declares_serial` (Review Focus — on a multi-core host with Ray, `speedup > 1`; if Ray unavailable, result explicitly records `backend=="serial", speedup_claimed=False` and does not report a speedup).
- [ ] **Step 2:** Run; expect FAIL.
- [ ] **Step 3:** Implement with Ray (remote per instance; parameter draws over the model's rate constants); multiprocessing/serial fallback that sets the honesty flag.
- [ ] **Step 4:** Run; expect PASS.
- [ ] **Step 5: Commit.**

---

## Task 9: Arm 4 — fully native process-bigraph network

**Files:**
- Create: `viva_sbml_decomposition/processes/native_glycolysis.py` — native `Process`/`Step` classes for the ODE network.
- Create: `viva_sbml_decomposition/composites/arm4_native.py`
- Test: `tests/test_arm4_native.py`

**Interfaces:**
- Consumes: Task 2 model (the rate law reference), Task 3 accuracy, Task 5 reference.
- Produces: native Processes (auto-discovered per bigraph-schema convention: subclass `Process`, declare `bigraph-schema` dep) and `glycolysis_native(core=None)`.

- [ ] **Step 1: Write failing test** `test_arm4_matches_reference_within_tolerance`: run the native composite; compare to reference (Task 3); assert within provisional `T` and species paired by name.
- [ ] **Step 2:** Run; expect FAIL.
- [ ] **Step 3:** Implement the ODE RHS as native pbg Processes (integration scheme per spec "Left to the builder" — e.g. one Process owning the rate vector + an explicit integrator step; matched tolerance to reference where the scheme allows).
- [ ] **Step 4:** Run; expect PASS.
- [ ] **Step 5: Commit.**

---

## Task 10: Composability scoring + arm 6 coupling demonstration

**Files:**
- Create: `viva_sbml_decomposition/harness/composability.py` — the scoring rubric.
- Create: `viva_sbml_decomposition/composites/arm6_coupling.py`
- Test: `tests/test_composability.py`, `tests/test_arm6_coupling.py`

**Interfaces:**
- Produces: `score_arm(arm_id) -> ComposabilityScore` (`substitutability`, `coupling_readiness`, `multiscale_fit`, each on a fixed scale per spec "Left to the builder"); `glycolysis_coupled(core=None)` — arm 3's decomposition with one submodel swapped for the native-PBG version (Task 9) OR coupled to an external input, demonstrating a composition the monolith can't express.

- [ ] **Step 1: Write failing tests**: `test_composability_scores_monotonic` (monolithic scores lowest on substitutability/coupling-readiness, native/modular highest); `test_arm6_runs_a_coupling_the_monolith_cannot` (the coupled composite builds + runs + emits; it has no monolithic reference → `score_arm` marks accuracy `N/A`); `test_arm6_excluded_from_accuracy_axis` (Review Focus — the viz-data assembler flags arm 6 `accuracy=None`).
- [ ] **Step 2:** Run; expect FAIL.
- [ ] **Step 3:** Implement the rubric (fixed criteria + scale) and the arm-6 composite (reuse Task 9 native submodel inside Task 7's modular wiring).
- [ ] **Step 4:** Run; expect PASS.
- [ ] **Step 5: Commit.**

---

## Task 11: Investigation + studies, wired for the discovery engine

**Files:**
- Workspace data: one investigation + six studies (model × arm) authored via `/viva-investigation` and `/viva-study`.

**Interfaces:**
- Consumes: all arm composites (Tasks 5–10).
- Produces: `investigations/sbml-decomposition/` with acceptance criteria (the tradeoff questions) and per-study canonical verdicts set so the discovery-engine roll-up is real.

- [ ] **Step 1:** `/viva-investigation new sbml-decomposition` with the overview/question from the spec.
- [ ] **Step 2:** For each arm, `/viva-study` a member study: baseline composite = that arm's composite, `runtime.default_emitter: xarray`, the arm's acceptance criteria (accuracy within `T`, overhead ratio bound, composability target; arm 6's accuracy criterion marked N/A).
- [ ] **Step 3:** Run each study (`/viva-run` or the study's Simulate phase); confirm each emits the expected observables.
- [ ] **Step 4:** Set each study's canonical verdict from its run (`/viva-biology-forward` / `viva-study` decide phase) so the roll-up is populated.
- [ ] **Step 5: Verify** via `/viva-navigate decisions sbml-decomposition` that the discovery-engine frontier is non-empty and names a next action (proves the crank is wired).
- [ ] **Step 6: Commit** (dashboard authoring commits to the workspace branch).

---

## Task 12: The 3-way tradeoff visualization

**Files:**
- Create: `viva_sbml_decomposition/visualizations/tradeoff.py` — a v2 decorated-function Visualization.
- Test: `tests/test_tradeoff_viz.py`

**Interfaces:**
- Consumes: the six studies' runs (accuracy results, timing ratios, composability scores).
- Produces: `tradeoff_map(...)` Visualization — accuracy-vs-speed scatter with composability encoded (size/color), arm 6 shown as a capability callout (not on the accuracy axis), + a decision table with a composability column and arm 6 accuracy cell "N/A".

- [ ] **Step 1: Write failing test** `test_tradeoff_data_excludes_arm6_from_accuracy`: the viz-data assembler places arms 1–5 on the accuracy axis and arm 6 only in the callout/table with accuracy `None` (Review Focus).
- [ ] **Step 2:** Run; expect FAIL.
- [ ] **Step 3:** Implement via `/viva-viz` (v2 Visualization) rendering against the real xarray/zarr runs (mosaic-compose foundations pattern). Push for an informative interactive figure, not a default scatter.
- [ ] **Step 4:** Run; expect PASS; confirm the figure renders in the dashboard.
- [ ] **Step 5: Commit.**

---

## Task 13: Turn the crank to closure + report

**Files:**
- Workspace data: updated verdicts, narrative, decision table.

- [ ] **Step 1:** Run `/viva-navigate decisions sbml-decomposition`; resolve each frontier item (set verdicts, adjudicate open challenges, fill narrative gaps) until `closure` reports settled (every acceptance criterion met, no open narrative gap).
- [ ] **Step 2:** `/viva-report` (Pass A reviewer-readiness + Pass B lint) to regenerate the investigation report; fix anything it flags.
- [ ] **Step 3:** Confirm the report states the decision guide (use / avoid / why) grounded in the 3-way figure — the spec's core deliverable (and its anti-goal: no recommendation = failure).
- [ ] **Step 4: Commit.** Optionally publish the read-only workbench to gh-pages (on request).

---

## Next increment (not this plan)

Models 2 (oscillator) and 3 (signaling cascade) reuse Tasks 3/4 harness + the arm patterns (Tasks 5–10). They are authored study-by-study as the discovery-engine frontier calls for them (`/viva-navigate decisions`), each a new model module + its arms + member studies under the same investigation — no new harness. Plan them lightly (or just execute) once Task 13 proves the slice.
