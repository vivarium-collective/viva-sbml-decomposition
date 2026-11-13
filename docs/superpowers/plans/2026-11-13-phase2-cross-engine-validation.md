# Phase 2 — Cross-Engine Validation Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one new study to the **decomposition** investigation that validates the three engines agree on the same model — tellurium vs COPASI vs native run monolithically, graded pairwise against the tellurium reference — and promotes the decomposition **splitting-error** to a first-class, reported metric, with an honest COPASI-nondeterminism caveat. The study's verdict is **derived** by the discovery-engine (Phase 0 foundation), and adding it **reopens** the investigation so its verdict is re-derived.

**Architecture:** Reuse, don't rebuild. `engine_monolithic` already runs any of the three engines; `compare_to_reference` already measures max-relative-error between two trajectories; `reaction_split` + the `mechanical-split-cost` method already produce splitting-error-vs-interval; `run_in_subprocess_with_retry` already survives COPASI's intermittent loader abort; `accuracy_divergence` is an existing visualization. Phase 2 orchestrates these into one agreement harness, one study with derived (not authored) outcomes, and one figure, then wires the study into the investigation.

**Tech Stack:** Python 3.12, process-bigraph composites, tellurium/libroadrunner + COPASI (basico) + the native process, `discovery_engine` (nested `study.*` API), the workspace viz framework, pytest.

**Spec:** `docs/superpowers/specs/2026-11-13-sbml-decomposition-clarity-and-capabilities-design.md` (Phase 2 of §6, capability 1 → decomposition investigation; §6a derive-first verdicts). Phases 0+1 are merged (`main` @ `d8160b1`).

## Global Constraints

- Worktree only (`~/code/viva-sbml-decomposition--cleanup`, branch `cleanup/phase2-cross-engine-validation` off `origin/main` @ `d8160b1`); canonical checkout read-only. Verify `git branch --show-current` + HEAD before each commit.
- No AI attribution; squash-merge; no auto-merge; one PR for the phase.
- Run all tooling with the canonical venv + `PYTHONPATH=.`; CLI tools also with the venv bin on `PATH`. Engine imports use the nested `discovery_engine.study.*` layout.
- **Derive-first (Phase 0 foundation):** the study's verdict and the investigation roll-up are **computed** by the engine from graded outcomes — do NOT hand-author a favorable `verdict_status`. `tests/test_verdict_honesty.py::test_no_verdict_overclaim` must stay green after the study is added (reopening the investigation re-derives, never overclaims).
- **Reproducibility (two-class convention, Phase 1 `docs/CONVENTIONS.md`):** this is an emitter-backed study → commit a seeded `runs.db`. Seed every run; keep COPASI deterministic where possible and flag where it is not.
- **COPASI sharp edge:** COPASI has a documented ~1-in-6 nondeterministic C++ loader abort and is not bit-reproducible. Load/run COPASI only through `run_in_subprocess_with_retry`; never gate a PASS on COPASI bit-identity — grade agreement to a tolerance and state the caveat.
- **Anti-goal A1 (design):** this capability lands **complete** (harness + study + derived verdict + committed runs.db + figure + caveat in the report) or not at all. No scaffolded/deferred half-study.

## Review Focus

- **A COPASI abort fails the whole study instead of being tolerated** (the ~1-in-6 loader abort surfaces during the run) → Task 1's harness routes every COPASI load through `run_in_subprocess_with_retry`, and its test asserts a simulated transient failure is retried, not propagated.
- **The agreement verdict silently overclaims** (COPASI monolithic agreement is actually worse than the threshold, but the study authors a passing verdict) → Task 2 grades the measured agreement and lets the engine derive the verdict; the honesty test fails if an authored verdict exceeds the computed one. The verdict must reflect whatever is measured, including a `passing-with-caveats` or worse if COPASI diverges.
- **The splitting-error metric is presented without its interval dependence** (a single error number implies the split is simply "wrong") → Task 2 reports splitting-error as a function of coupling interval (reusing the `mechanical-split-cost` method), not a scalar, so the first-order convergence is visible.
- **The figure renders empty / the report 404s** (missing run data, or a viz that assumes all three engines produced a trajectory when COPASI aborted) → Task 3's figure degrades gracefully when an engine arm is missing, and Task 4 asserts the rendered report ≥ 20000 bytes and the publish render-check exits 0.
- **Adding the study breaks the README/dashboard or the investigation acceptance** (member list, acceptance criteria, or generated tables go stale) → Task 4 runs `write_investigation_acceptance`, `gen-readme --check`, and asserts the new study appears in both the investigation `computed_acceptance.criteria` and the README table.

---

## File Structure

- `viva_sbml_decomposition/harness/cross_engine.py` — **new**, the 3-engine agreement harness (Task 1).
- `workspace/investigations/sbml-decomposition/studies/cross-engine-validation/study.yaml` — **new** study (Task 2).
- `workspace/investigations/sbml-decomposition/studies/cross-engine-validation/runs.db` — **new**, committed seeded run store (Task 2).
- `viva_sbml_decomposition/visualizations/cross_engine_agreement.py` — **new** figure, or an extension of `accuracy_divergence.py` (Task 3; builder picks, see decision).
- `workspace/investigations/sbml-decomposition/investigation.yaml` — member + `computed_acceptance` refreshed by the engine (Task 4).
- `workspace/investigations/sbml-decomposition/report.html`, `README.md` — regenerated (Task 4).
- `tests/test_cross_engine.py` — **new** (Tasks 1–2); `tests/test_artifact_convention.py`, `tests/test_verdict_honesty.py` — stay green (Task 4).

---

### Task 1: Three-engine agreement harness

**Files:**
- Create: `viva_sbml_decomposition/harness/cross_engine.py`
- Create test: `tests/test_cross_engine.py`

**Interfaces:**
- Consumes: `engine_monolithic`/`build_document` (run a model in one engine), `viva_sbml_decomposition.harness.accuracy.compare_to_reference(ref_traj, ref_t, arm_traj, arm_t) -> AccuracyResult` (has `.max_rel_err`), `viva_sbml_decomposition.composites._engine.run_in_subprocess_with_retry` (COPASI safety), the investigation's canonical Teusink-glycolysis Antimony (reuse `models/` — builder resolves the exact symbol).
- Produces: `cross_engine_agreement(antimony: str, engines=("tellurium","copasi","native"), t_end, n_points, seed) -> dict` returning `{"reference": "tellurium", "species": [...], "agreement": {engine: max_rel_err_vs_reference | None}, "aborted": [engines that could not run]}`. COPASI runs via `run_in_subprocess_with_retry`; an engine that still cannot produce a trajectory is recorded in `aborted` with `agreement[engine] = None`, never raising.

- [ ] **Step 1: Write the failing test** `tests/test_cross_engine.py`:

```python
def test_agreement_tellurium_self_is_zero_and_cross_engine_finite():
    from viva_sbml_decomposition.harness.cross_engine import cross_engine_agreement
    res = cross_engine_agreement(_MASS_ACTION, engines=("tellurium", "native"), t_end=10.0, n_points=101, seed=0)
    assert res["reference"] == "tellurium"
    assert res["agreement"]["tellurium"] == 0.0          # reference vs itself
    assert res["agreement"]["native"] is not None and res["agreement"]["native"] >= 0.0

def test_copasi_transient_abort_is_retried_not_raised(monkeypatch):
    # A COPASI load that fails once then succeeds must be retried, and a
    # permanently-aborting engine lands in `aborted` with agreement None — never raises.
    ...  # drive run_in_subprocess_with_retry with a one-shot failure; assert no exception
```

(Builder: use a small inline mass-action Antimony for `_MASS_ACTION`; for the retry test, inject a transient failure through the retry wrapper's seam rather than invoking real COPASI.)

- [ ] **Step 2: Run it, verify it fails** — `pytest tests/test_cross_engine.py -v` → FAIL (harness missing).
- [ ] **Step 3: Implement `cross_engine_agreement`** in `harness/cross_engine.py`: run the model in each engine (tellurium + native in-process; COPASI through `run_in_subprocess_with_retry`), align trajectories on the shared grid, compute `compare_to_reference(tellurium, engine).max_rel_err` per engine, collect aborts. Deterministic: fixed `seed`, fixed grid.
- [ ] **Step 4: Run it** — `pytest tests/test_cross_engine.py -v` → PASS.
- [ ] **Step 5: Commit** — `git add viva_sbml_decomposition/harness/cross_engine.py tests/test_cross_engine.py && git commit -m "feat(cross-engine): 3-engine monolithic agreement harness (COPASI via retry, abort-tolerant)"`

---

### Task 2: The cross-engine-validation study + derived verdict + runs.db

**Files:**
- Create: `workspace/investigations/sbml-decomposition/studies/cross-engine-validation/study.yaml`
- Create (by running): `.../cross-engine-validation/runs.db`
- Extend test: `tests/test_cross_engine.py`

**Interfaces:**
- Consumes: Task 1's `cross_engine_agreement`; the `kind: accuracy` behavior-test contract (`measure.kind: accuracy`, `pass_if: {op: threshold, detail: "< <tol>"}`, graded by the registered evaluators → `computed_outcomes`); the `mechanical-split-cost` splitting-error method (reused or cited) for the first-class split-error metric; `/viva-study` + `/viva-investigation` skills and `scripts/` to run + grade the study the canonical way.
- Produces: a study whose canonical run commits `computed_outcomes` + a seeded `runs.db`, with behavior tests: (a) **cross-engine-agreement** — tellurium↔native and tellurium↔COPASI monolithic max-rel-err below a stated tolerance (derived PASS/FAIL from the measurement); (b) **splitting-error-reported** — the additive-split error is reported as a function of coupling interval (first-class metric). Verdict is **derived**, not authored.

- [ ] **Step 1: Write the failing test** — add to `tests/test_cross_engine.py` a test asserting the study YAML exists, declares `investigation: sbml-decomposition`, has `behavior_tests` of `kind: accuracy`, and (after the run) its canonical run carries `computed_outcomes` for each behavior test. Initially FAILS (no study).
- [ ] **Step 2: Scaffold the study** — create `cross-engine-validation/study.yaml` via `/viva-study` (so it is canonical): title, question ("do the three engines agree on the same model, and what does the additive split cost?"), baseline composite(s) that run the agreement harness / engine arms, and the two `kind: accuracy` behavior tests with explicit thresholds + the COPASI caveat in each `description`. Do **not** set a favorable authored `verdict_status`.
- [ ] **Step 3: Run + grade deterministically** — run the study (seeded) so the harness produces the agreement numbers; grade via the engine so `computed_outcomes` are written; `write_investigation_acceptance` for the roll-up. The verdict is whatever the measurement supports — if COPASI monolithic agreement is worse than tolerance, the honest outcome is `passing-with-caveats`/`failing`, not a forced pass.
- [ ] **Step 4: Commit the seeded runs.db** — confirm `runs.db` exists (emitter-backed convention); if the harness is code-graded with no emitter, record `computed_outcomes` as the artifact and note it (mirrors Phase 1 `docs/CONVENTIONS.md`; **decision flagged** — see end).
- [ ] **Step 5: Verify** — `pytest tests/test_cross_engine.py tests/test_verdict_honesty.py tests/test_artifact_convention.py -v` → PASS (agreement graded; no overclaim; artifact committed).
- [ ] **Step 6: Commit** — `git add workspace/investigations/sbml-decomposition/studies/cross-engine-validation && git commit -m "feat(study): cross-engine-validation — 3-engine agreement + first-class splitting-error, derived verdict"`

---

### Task 3: The agreement/divergence figure

**Files:**
- Create: `viva_sbml_decomposition/visualizations/cross_engine_agreement.py` (or extend `accuracy_divergence.py` — **decision flagged**)
- Extend test: `tests/test_visualizations.py`

**Interfaces:**
- Consumes: the study's committed run data / `computed_outcomes`; the workspace viz framework (a decorated visualization function, as in the existing `visualizations/*`).
- Produces: one figure showing the cross-engine agreement/divergence map (tellurium vs COPASI vs native) **and** the splitting-error-vs-interval curve, degrading gracefully when a COPASI arm is absent.

- [ ] **Step 1: Write the failing test** — add to `tests/test_visualizations.py` a test that the new viz builds a non-empty figure from the study data and does not raise when the COPASI arm is `None`.
- [ ] **Step 2: Run it, verify it fails.**
- [ ] **Step 3: Implement the viz** following the existing `accuracy_divergence`/`tradeoff_map` patterns.
- [ ] **Step 4: Verify** — `pytest tests/test_visualizations.py -v` → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(viz): cross-engine agreement/divergence + splitting-error figure"`

---

### Task 4: Wire into the investigation; render; regenerate

**Files:**
- Modify (engine-written): `workspace/investigations/sbml-decomposition/investigation.yaml` (member + `computed_acceptance`)
- Create/refresh: `workspace/investigations/sbml-decomposition/report.html`
- Modify: `README.md`

**Interfaces:**
- Consumes: `discovery_engine...write_investigation_acceptance(inv_dir, workspace=".")`; the AC↔study link (`computed_acceptance.criteria` `{study, behavior}` pair for the new study); `scripts/publish_investigation_reports.py` + `vivarium-workbench gen-readme`.
- Produces: the decomposition investigation reopened and its verdict **re-derived** with the new study included; a refreshed `report.html` + README listing the new study.

- [ ] **Step 1: Write the failing test** — add to `tests/test_verdict_honesty.py` (or `test_cross_engine.py`) a check that `investigation.yaml`'s `computed_acceptance.criteria` includes a `{study: cross-engine-validation, ...}` entry. FAILS until wired.
- [ ] **Step 2: Add the AC↔study link** — add the acceptance criterion pair(s) for `cross-engine-validation` to the investigation's authored acceptance inputs, then re-run `write_investigation_acceptance` so the engine re-derives `computed_acceptance` + `computed_verdict_status` with the study included.
- [ ] **Step 3: Render + regenerate** — `scripts/publish_investigation_reports.py --workspace . --only sbml-decomposition --out <tmp>`, copy to `workspace/investigations/sbml-decomposition/report.html`; `vivarium-workbench gen-readme --workspace .`.
- [ ] **Step 4: Verify** — `pytest tests/test_verdict_honesty.py -v` (no overclaim; new criterion present); report.html ≥ 20000 bytes; `gen-readme --check` exit 0; the new study appears in the README table + the investigation criteria.
- [ ] **Step 5: Commit** — `git commit -m "feat(investigation): add cross-engine-validation; re-derive verdict; render report + README"`

---

### Task 5: Full-suite green + PR

- [ ] **Step 1: Run every CI gate locally** (venv + `PYTHONPATH=.`): `pytest tests/ -v`; `python scripts/lint-workspace.py`; `uv lock --check`; `vivarium-workbench gen-readme --workspace . --check`; `python scripts/check-workflow-freshness.py --check`; `vivarium-workbench-publish --workspace . --out "$TMPDIR/render-check"`.
- [ ] **Step 2: Verify branch/HEAD** — `cleanup/phase2-cross-engine-validation`; `git log --oneline origin/main..HEAD` shows only this phase's commits.
- [ ] **Step 3: Push + open PR** — `git push -u origin …`; `gh pr create --base main` titled `feat: Phase 2 — cross-engine validation study (3-engine agreement + first-class splitting-error)`, body summarizing the derived verdict (honest, incl. the COPASI caveat and whatever agreement was actually measured), the reused infra, and any flagged decision. No AI attribution.
- [ ] **Step 4: Do NOT merge** — hand to the owner.

---

## Decisions flagged for your review

1. **Reference model:** the agreement harness runs the investigation's canonical **Teusink glycolysis** (BIOMD0000000064, 17 species — the model every decomposition study already uses) as the primary case, so the cross-engine agreement is on the same model as the rest of the investigation. *Alternative:* run the full oracle ladder (several models) for a broader agreement map — more runtime, more COPASI abort exposure.
2. **Splitting-error source:** reuse the existing `mechanical-split-cost` additive-split error-vs-interval data as the first-class metric (cite + surface it in this study's figure) rather than re-measuring it, to avoid duplicating a run. *Alternative:* re-measure inside this study for a self-contained artifact.
3. **Figure home:** a new `cross_engine_agreement.py` viz vs. extending `accuracy_divergence.py`. Leaning new (keeps the existing figure stable), builder's call.
4. **Artifact class:** if the agreement harness is emitter-backed, commit `runs.db` (decomposition convention); if it is code-graded like recovery, commit `computed_outcomes` and note it per `docs/CONVENTIONS.md`. Resolved at Task 2 Step 4 against how the study actually runs.
5. **Agreement tolerance:** the pass threshold for monolithic cross-engine agreement (the existing seam-level swap measured ~3e-7; monolithic COPASI-vs-tellurium on a stiff glycolysis model may be looser). The threshold is stated explicitly in the behavior test and the **verdict is derived from the measurement** — a looser-than-hoped COPASI agreement yields an honest `passing-with-caveats`, not a failure to hide.
