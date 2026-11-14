# Phase 3 — Scaling / Performance Benchmarks Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one new study to the **decomposition** investigation that quantifies decomposition scaling systematically along two axes — **model size** (monolithic build strain vs species count) and **#instances** (ensemble wall-clock speedup vs instance count) — and commits a **performance baseline** that Phase 6's viva-core spike will be measured against. The verdict is **derived** by the discovery-engine and graded on *robust, machine-independent* properties (a superlinear build exponent, a real parallel speedup), never on absolute wall-clock, which is not cross-machine reproducible.

**Architecture:** Reuse the existing scaling harnesses — `composites/_multiorgan.multiorgan_scaling(ns)` (model-size curve: mono vs decomposed build/run/total/accuracy) and the `ensemble` / heap-scheduler path (instance scaling) — run one bounded systematic sweep offline, commit the numbers as a baseline artifact, grade qualitative properties, and render consolidated scaling curves. No new timing methodology; this makes the scaling evidence first-class and sets the Phase-6 reference.

**Tech Stack:** Python 3.12, process-bigraph composites, tellurium, Ray (ensemble; serial fallback), numpy (fit the scaling exponent), `discovery_engine` (nested `study.*`), the viz framework, pytest.

**Spec:** `docs/superpowers/specs/2026-11-13-sbml-decomposition-clarity-and-capabilities-design.md` (Phase 3 of §6, capability 2; §6a derive-first). Phases 0–2 merged (`main` @ `613d749`).

## Global Constraints

- Worktree only (`~/code/viva-sbml-decomposition--cleanup`, branch `cleanup/phase3-scaling-benchmarks` off `origin/main` @ `613d749`); canonical read-only. Verify branch + HEAD before each commit.
- No AI attribution; squash-merge; no auto-merge; one PR.
- Canonical venv + `PYTHONPATH=.`; nested `discovery_engine.study.*`; derive-first verdicts (no hand-authored favorable `verdict_status`; `test_verdict_honesty` stays green).
- **Wall-clock is NOT cross-machine reproducible.** Grade only machine-independent properties (build-time exponent > 1; ensemble speedup > 1 where Ray is available). The committed baseline numbers are **this machine's**, labelled as such — the honest analogue of Phase 2's COPASI-non-bit-reproducibility caveat. No test asserts an absolute second-count or a fixed speedup factor.
- **Code-graded study** → commits `computed_outcomes` (+ a baseline JSON), per the registry-derived convention test from Phase 2 (`cross_engine_agreement`-style). Its `measure.kind` evaluators register in `evaluators.py`.
- **Ray may be unavailable** (CI, or a flaky init) → the ensemble harness already falls back to serial; the speedup behavior test must then **skip/caveat** (serial has no parallel speedup), not FAIL.
- **Bounded sweeps.** The offline grade may use a modest model-size set and instance set (minutes, not an hour); CI tests must not re-run the full sweep — they read committed outcomes plus one tiny-N harness smoke test.

## Review Focus

- **A grade keys off absolute wall-clock and flips across machines** (CI's slower runner fails a "< X s" or "≥ 7.5× speedup" assertion) → every behavior test grades a qualitative property (exponent > 1; speedup > 1); Task 2's tests assert the *property*, never a magnitude, and the baseline JSON is documented as machine-specific.
- **Ray-unavailable turns a real pass into a failure** (no parallel backend → speedup ≈ 1) → Task 1's harness reports the backend used; Task 2's speedup test skips-with-caveat when the backend is serial, and its test exercises the serial-fallback branch.
- **The sweep makes CI time out** (a full model-size × instance sweep runs during the suite) → the offline grade owns the heavy sweep; Task 1's committed-baseline artifact feeds the figure, and the CI tests use tiny N and read committed outcomes. Task 5's full-suite run confirms the added time is bounded.
- **The study duplicates / contradicts `ensemble-and-multirate` or `bigger-models-and-abm-limit`** (re-measures the same numbers, possibly disagreeing) → the study CITES those studies' committed findings for the headline numbers and adds the *systematic curve + baseline record*; Task 4 confirms the investigation verdict stays coherent and no finding contradicts a sibling.
- **The superlinear-build fit is fragile at tiny N** (2 points, or noise makes the exponent < 1 on a fast machine) → Task 1 fits over ≥ 3 sizes with a floor, and Task 2's test tolerates noise by asserting exponent > 1.0 with a margin the committed data clears, else records the honest measured exponent.

---

## File Structure

- `viva_sbml_decomposition/harness/scaling.py` — **new**, the bounded systematic sweep + exponent fit (Task 1).
- `viva_sbml_decomposition/visualizations/scaling_benchmarks.json` — **new**, committed baseline (Task 1).
- `workspace/investigations/sbml-decomposition/studies/scaling-benchmarks/study.yaml` — **new** study (Task 2).
- `viva_sbml_decomposition/evaluators.py` — **+2** evaluator kinds `scaling_build_exponent`, `scaling_ensemble_speedup` (Task 2).
- `scripts/grade_scaling.py` — **new**, runs the sweep + grades (Task 2).
- `viva_sbml_decomposition/visualizations/scaling_benchmarks.py` — **new** consolidated scaling figure (Task 3).
- `workspace/investigations/sbml-decomposition/investigation.yaml`, `report.html`, `README.md` — wired + rendered (Task 4).
- `tests/test_scaling.py` — **new** (Tasks 1–2); `test_verdict_honesty.py`, `test_artifact_convention.py`, `test_visualizations.py` stay green (Task 4).

---

### Task 1: Scaling sweep harness + committed baseline

**Files:** Create `viva_sbml_decomposition/harness/scaling.py`, `viva_sbml_decomposition/visualizations/scaling_benchmarks.json`; test `tests/test_scaling.py`.

**Interfaces:**
- Consumes: `composites._multiorgan.multiorgan_scaling(ns, repeats, run_decomposed_max_n)` (model-size curve); `composites.ensemble.run_ensemble` / the `ensemble` composite (instance scaling + the backend it used); `numpy.polyfit` on log-log for the build exponent.
- Produces: `scaling_benchmarks(model_sizes, instance_counts, *, repeats, seed) -> dict` returning `{"model_size": [{n, species, mono_total_s, decomp_total_s, accuracy}...], "build_exponent": float, "ensemble": {"instances": [P...], "speedup": [float...], "backend": "ray"|"serial"}, "machine": <platform tag>}`. Deterministic in structure (timings vary); never raises on Ray-unavailable (records `backend: serial`).

- [ ] **Step 1: Write the failing test** `tests/test_scaling.py`:

```python
def test_scaling_harness_shape_and_exponent():
    from viva_sbml_decomposition.harness.scaling import scaling_benchmarks
    r = scaling_benchmarks(model_sizes=(1, 2, 4), instance_counts=(1, 2), repeats=1, seed=0)
    assert [p["species"] for p in r["model_size"]] == [4, 7, 13]      # 3n+1
    assert r["build_exponent"] > 0                                    # a finite positive fit
    assert r["ensemble"]["backend"] in ("ray", "serial")
    assert len(r["ensemble"]["speedup"]) == 2

def test_scaling_harness_tolerates_serial(monkeypatch):
    # Ray-unavailable must yield backend="serial" with finite speedups, never raise.
    ...  # force the serial-fallback path; assert backend=="serial", no exception
```

- [ ] **Step 2: Run it, verify it fails** — `pytest tests/test_scaling.py -v` → FAIL (harness missing).
- [ ] **Step 3: Implement `scaling_benchmarks`** — call `multiorgan_scaling(model_sizes)` for the model-size curve; fit `build_exponent` = slope of `log(mono_build_s)` vs `log(species)` via `numpy.polyfit` (≥3 points); run the ensemble harness at each instance count, record wall-clock speedup vs P=1 and the backend used; tag `machine` from `platform`.
- [ ] **Step 4: Run it** — `pytest tests/test_scaling.py -v` → PASS (tiny N; seconds).
- [ ] **Step 5: Commit** — the harness + test (the baseline JSON is written by the grade script in Task 2). `git commit -m "feat(scaling): model-size + ensemble scaling sweep harness (serial-safe, exponent fit)"`

---

### Task 2: The scaling-benchmarks study + derived verdict + baseline JSON

**Files:** Create `studies/scaling-benchmarks/study.yaml`, `scripts/grade_scaling.py`; modify `evaluators.py` (+2 kinds); extend `tests/test_scaling.py`.

**Interfaces:**
- Consumes: Task 1's `scaling_benchmarks`; the `kind`→evaluator registry (`register_evaluators`); `discovery_engine` compute/verdict (nested).
- Produces: a code-graded study whose canonical run carries `computed_outcomes` (+ a committed `scaling_benchmarks.json` baseline the figure reads), with two behavior tests graded on **robust properties**: (a) `scaling_build_exponent` — the monolithic build scales superlinearly (exponent > 1); (b) `scaling_ensemble_speedup` — the same-rate ensemble gives a real parallel speedup (> 1) **when a parallel backend ran**, else PASS-with-caveat (serial). Verdict derived; `reconcile: no_authored`.

- [ ] **Step 1: Write the failing test** — add to `tests/test_scaling.py` a test that the study exists, declares `investigation: sbml-decomposition`, and (after grade) its canonical run has `computed_outcomes` for both behavior tests with `evaluated_by: code` and `reconcile: no_authored`. FAILS until built.
- [ ] **Step 2: Add the two evaluators** in `evaluators.py` (registered in `register_evaluators`): `scaling_build_exponent` (PASS if measured exponent > 1.0, detail reports the exponent + that wall-clock is machine-specific); `scaling_ensemble_speedup` (PASS if backend parallel and speedup > 1; if serial, PASS with a "no parallel backend — speedup not measured" caveat). Store-less (`@storeless`); re-run the harness internally.
- [ ] **Step 3: Author the study.yaml** (schema 3, mirrors the cross-engine-validation study): title, question (how does decomposition scale with model size and instance count, and what is the Phase-6 baseline?), baseline composite (`engine_monolithic` glycolysis nominal), the two `kind:` behavior tests with explicit thresholds + the machine-specificity caveat, a `visualizations:` entry (`address: local:ScalingBenchmarks`, Task 3), and parent_studies citing `ensemble-and-multirate` + `bigger-models-and-abm-limit`. No authored favorable `verdict_status`.
- [ ] **Step 4: Grade + write baseline** — `scripts/grade_scaling.py` creates one canonical run, runs the sweep, `compute_outcomes` → `computed_outcomes`, and writes `visualizations/scaling_benchmarks.json` (the committed baseline + the Phase-6 reference numbers, machine-tagged). Also update `tests/test_recovery_evaluators.py`'s registry assertion to include the two new kinds.
- [ ] **Step 5: Verify** — `pytest tests/test_scaling.py tests/test_verdict_honesty.py tests/test_artifact_convention.py tests/test_recovery_evaluators.py -v` → PASS (derived verdict; no overclaim; code-graded artifact present; registry updated).
- [ ] **Step 6: Commit** — `git commit -m "feat(study): scaling-benchmarks — model-size + ensemble scaling, Phase-6 baseline, derived verdict"`

---

### Task 3: Consolidated scaling-curves figure

**Files:** Create `viva_sbml_decomposition/visualizations/scaling_benchmarks.py`; extend `tests/test_visualizations.py` (add to `FIGS` + a serial/empty-data graceful test).

**Interfaces:** Consumes the committed `scaling_benchmarks.json`; the viz framework. Produces `ScalingBenchmarks` — a figure with the model-size build curve (mono vs decomposed, log-log, the superlinear monolith visible) and the ensemble speedup vs instance count; degrades gracefully when the ensemble backend was serial (speedup ≈ 1) or data is missing.

- [ ] Steps mirror Phase 2 Task 3: failing viz test → implement reading the JSON (+ `load_metrics` if useful) → `test_visualizations.py` green → commit. Figure declares `address: local:ScalingBenchmarks` + `config: {}` + `covers:` in the study (so it is not a `broken_stub` — the Phase-2 review lesson).

---

### Task 4: Wire into the investigation; re-derive; render

**Files:** Modify `investigation.yaml` (members + `at_a_glance` + authored `acceptance_criteria` pair), refresh `report.html`, `README.md`.

- [ ] Add `scaling-benchmarks` to `studies:` + `at_a_glance:` + an `acceptance_criteria` pair (`{study: scaling-benchmarks, behavior: <the build-exponent or speedup test name>}`); re-run `write_investigation_acceptance`; confirm the new criterion resolves and the verdict stays honest (`test_verdict_honesty` green). Render the report (`publish_investigation_reports.py --only sbml-decomposition`) — **with the worktree source reachable so the new figure renders** (Phase-2 lesson: the env-worker imports the installed package; a worktree `uv sync` or post-merge publish is required for the figure to appear in the committed snapshot — decide per the Phase-2 flag). `gen-readme --workspace .`. Commit.

---

### Task 5: Full-suite green + PR

- [ ] Run every CI gate (pytest tests/; lint-workspace; uv lock --check; gen-readme --check; check-workflow-freshness; vivarium-workbench-publish render-check). Confirm the added test time is bounded. Verify branch/HEAD + only this phase's commits. Push + `gh pr create --base main` (title `feat: Phase 3 — scaling/performance benchmarks study`, body: the derived verdict, robust-property grading + machine-specific baseline caveat, reuse of the scaling harnesses, the figure-render note). **Do NOT merge.**

---

## Decisions flagged for your review

1. **Grade robust properties, not wall-clock** (build exponent > 1; ensemble speedup > 1), because wall-clock is not cross-machine reproducible. The committed baseline JSON is **this machine's numbers**, labelled machine-specific — the honest analogue of Phase 2's COPASI caveat. This is the single biggest design choice; alternative (absolute-second thresholds) would make CI flaky and the baseline misleading.
2. **Reuse + cite, don't re-measure the headline numbers.** The 7.5× Ray win and the O(N^1.6) build strain are already committed in `ensemble-and-multirate` / `bigger-models-and-abm-limit`; this study runs a bounded systematic sweep for the *curve shape + baseline record* and cites those studies for the headlines, to avoid duplication/contradiction. Alternative: a fully independent large sweep (slower, risks disagreeing with the siblings).
3. **Baseline artifact = a committed JSON** (`scaling_benchmarks.json`) the figure and Phase 6 read, machine-tagged. Phase 6 compares viva-core against it with the same machine-specificity caveat.
4. **Figure-in-snapshot** carries the Phase-2 limitation (worktree env-worker can't import the new viz module); either `uv sync` the worktree to render pre-merge, or let publish-from-main render it. Your call, as in Phase 2.
5. **Two scaling axes, not three.** The design names model size, #instances, and #engines. This study quantifies the two that scale meaningfully (model size, #instances). The **#engines** axis is already settled by Phase 2's `cross-engine-validation`: engines are interchangeable at ~1e-7 with no scaling penalty, and a composite rarely spans more than 2–3 engines, so there is no #engines *curve* to draw — the study cites Phase 2 for that axis rather than inventing a thin one. Flag if you want an explicit #engines timing sweep anyway.
