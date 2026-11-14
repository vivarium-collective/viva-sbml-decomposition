# Phase 4 — Global Sensitivity / UQ Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one new study to the **decomposition** investigation that answers "which parameters drive the glycolysis model's behavior, and how much?" with a global sensitivity analysis — a **hand-rolled, seeded Morris elementary-effects** screening over the enzyme-capacity (Vmax) parameters, run in parallel on the existing Ray ensemble — producing a reproducible parameter-importance ranking with interaction/nonlinearity bands. The verdict is **derived** and graded on the **reproducible ranking** (not wall-clock), which Morris on a deterministic ODE provides.

**Architecture:** Reuse the `ensemble` harness (per-instance `param_draws` parameter overrides, Ray-parallel, tellurium) to evaluate the Morris trajectory samples. Hand-roll Morris (no new dependency — `SALib` is not installed and adds a heavy dep; Morris is ~40 lines of seeded numpy and gives exactly the design's "importance ranking + uncertainty bands": `mu*` = importance, `sigma` = interaction/nonlinearity). The indices are deterministic given the seed, so the study grades the ranking; only raw last-digits vary across BLAS/platforms (the recovery/Phase-2 lesson).

**Tech Stack:** Python 3.12, tellurium (parameter overrides), the Ray ensemble (serial fallback), numpy (Morris sampling + index estimation), `discovery_engine` (nested `study.*`), the viz framework, pytest.

**Spec:** `docs/superpowers/specs/2026-11-13-sbml-decomposition-clarity-and-capabilities-design.md` (Phase 4 of §6, capability 3). Phases 0–3 merged (`main` @ `8e78e52`).

## Global Constraints

- Worktree only (`~/code/viva-sbml-decomposition--cleanup`, branch `cleanup/phase4-sensitivity-uq` off `origin/main` @ `8e78e52`); canonical read-only. Verify branch + HEAD before each commit.
- No AI attribution; squash-merge; no auto-merge; one PR.
- Canonical venv + `PYTHONPATH=.`; nested `discovery_engine.study.*`; derive-first (no hand-authored favorable `verdict_status`; `test_verdict_honesty` stays green).
- **No new dependency.** Hand-roll Morris with numpy; do NOT add `SALib`. If a reviewer wants `SALib` later it is a separate dep PR.
- **Reproducible indices.** Morris on the deterministic glycolysis ODE with a fixed RNG seed is reproducible; grade the **ranking** (top-k parameters) and non-degeneracy, not absolute `mu*` magnitudes (last digits vary across BLAS/platforms). No test asserts a raw index to many digits.
- **Code-graded study** → commits `computed_outcomes` (+ a committed indices JSON baseline), per the registry-derived convention test. New `measure.kind` evaluators register in `evaluators.py`.
- **Ray may be unavailable** (CI) → the ensemble falls back to serial; the harness must run the same samples serially and produce identical indices (parallelism is only a speed detail, never changes the result). The behavior tests do not depend on the backend.
- **Bounded sample.** Morris cost is `r*(D+1)` model runs; keep `r` (trajectories) and `D` (parameters) modest so the offline grade is minutes and the CI harness smoke test is seconds (tiny r, D).

## Review Focus

- **A raw `mu*` magnitude is asserted and flips across machines** (BLAS noise changes the 4th digit) → Task 2 grades the *ranking* (which parameter is most important) and that indices are finite/non-degenerate, never an absolute magnitude; Task 1's test asserts determinism *on this machine* (same seed → same indices) and structure, not a cross-machine magnitude.
- **The sample is not reproducible** (unseeded RNG → different ranking each run) → Task 1 threads a fixed `seed` through the Morris sampler; its test asserts two runs at the same seed give identical indices.
- **Serial vs Ray changes the result** (a backend difference alters indices) → Task 1 computes indices from the per-sample outputs, which are backend-independent; the serial-fallback test asserts identical indices with Ray forced off.
- **A degenerate QoI makes every parameter look equally (un)important** (the chosen output barely responds to the Vmax sweep) → Task 1 picks a QoI that responds (a hub flux/species at steady state), and Task 2 asserts the ranking is non-degenerate (the top `mu*` is clearly above the bottom), else records the honest finding.
- **The figure misreads `sigma` as an error bar on `mu*`** (sigma is interaction/nonlinearity, not sampling error) → Task 3 labels the band as interaction/nonlinearity (Morris `sigma`), not a confidence interval, and the viz test checks the figure builds with the committed indices.

---

## File Structure

- `viva_sbml_decomposition/harness/sensitivity.py` — **new**, seeded Morris over Vmax params via the ensemble (Task 1).
- `viva_sbml_decomposition/visualizations/sensitivity_indices.json` — **new**, committed baseline (Task 2).
- `workspace/investigations/sbml-decomposition/studies/sensitivity-uq/study.yaml` — **new** study (Task 2).
- `viva_sbml_decomposition/evaluators.py` — **+1–2** kinds (`sensitivity_ranking`, optionally `sensitivity_nondegenerate`) (Task 2).
- `scripts/grade_sensitivity.py` — **new**, runs Morris + grades + writes the baseline (Task 2).
- `viva_sbml_decomposition/visualizations/sensitivity_indices.py` — **new** ranked-importance figure with interaction bands (Task 3).
- `investigation.yaml`, `report.html`, `README.md` — wired + rendered (Task 4).
- `tests/test_sensitivity.py` — **new**; `test_verdict_honesty.py`, `test_artifact_convention.py`, `test_recovery_evaluators.py`, `test_visualizations.py` stay green (Task 4).

---

### Task 1: Seeded Morris sensitivity harness (via the ensemble)

**Files:** Create `viva_sbml_decomposition/harness/sensitivity.py`; test `tests/test_sensitivity.py`.

**Interfaces:**
- Consumes: the glycolysis model + its Vmax parameter ids (`te.loada(GLYCOLYSIS_ANTIMONY).getGlobalParameterIds()` filtered to `*Vm*`); `composites.ensemble.run_ensemble(model, engine="tellurium", n_instances, param_draws=[{pid: value}...], full=True)` (Ray-parallel, serial fallback) to evaluate the Morris samples; a scalar QoI from each run's trajectory.
- Produces: `morris_sensitivity(params=None, *, r=4, levels=4, span=0.5, seed=0, qoi=None) -> dict` → `{"params": [pid...], "mu_star": [float...], "sigma": [float...], "ranking": [pid sorted by mu_star desc], "qoi": <name>, "backend": "ray"|"serial", "machine": <tag>}`. The Morris sample (r trajectories over D params, `levels` grid, `±span` fractional range around nominal) is seeded; indices are `mu_star` (mean |elementary effect|) and `sigma` (std of effects). Backend-independent; never raises on Ray-unavailable.

- [ ] **Step 1: Write the failing test** `tests/test_sensitivity.py`:

```python
def test_morris_shape_and_determinism():
    from viva_sbml_decomposition.harness.sensitivity import morris_sensitivity
    a = morris_sensitivity(r=3, levels=4, span=0.5, seed=0)
    b = morris_sensitivity(r=3, levels=4, span=0.5, seed=0)
    assert a["params"] and len(a["mu_star"]) == len(a["params"]) == len(a["sigma"])
    assert a["ranking"][0] in a["params"]
    assert a["mu_star"] == b["mu_star"]           # seeded → reproducible
    assert a["backend"] in ("ray", "serial")

def test_morris_serial_matches(monkeypatch):
    # Ray forced off must give the SAME indices (parallelism is a speed detail).
    import viva_sbml_decomposition.composites.ensemble as ens
    ...  # force serial; assert mu_star equals the default-backend run at the same seed
```

- [ ] **Step 2: Run it, verify it fails** — `pytest tests/test_sensitivity.py -v` → FAIL (harness missing).
- [ ] **Step 3: Implement `morris_sensitivity`** — default `params` = the Vmax ids; build r seeded Morris trajectories (one-at-a-time ±delta steps on the `levels` grid, scaled to `±span` around each nominal value); flatten all sample points to `param_draws`; `run_ensemble(..., full=True)`; reduce each run to the scalar QoI (default: a responsive hub quantity at steady state — builder picks and documents, e.g. final ATP or a glycolytic flux); compute elementary effects → `mu_star`, `sigma`; rank. Seed numpy's RNG explicitly.
- [ ] **Step 4: Run it** — `pytest tests/test_sensitivity.py -v` → PASS (tiny r; seconds–tens of seconds).
- [ ] **Step 5: Commit** — `git commit -m "feat(sensitivity): seeded Morris elementary-effects over Vmax params via the ensemble"`

---

### Task 2: The sensitivity-uq study + derived verdict + indices baseline

**Files:** Create `studies/sensitivity-uq/study.yaml`, `scripts/grade_sensitivity.py`; modify `evaluators.py` (+1–2 kinds); extend `tests/test_sensitivity.py`; update `tests/test_recovery_evaluators.py` registry set.

**Interfaces:**
- Consumes: Task 1's `morris_sensitivity`; the `kind`→evaluator registry; `discovery_engine` compute/verdict (nested).
- Produces: a code-graded study whose canonical run carries `computed_outcomes` (+ committed `sensitivity_indices.json` baseline), graded on **reproducible, robust** properties: (a) `sensitivity_ranking` — the Morris ranking is non-degenerate (top `mu*` ≥ k× the median, so the analysis actually discriminates parameters) and reproducible; optionally (b) the top parameter matches the biologically-expected rate-limiting enzyme (builder confirms from the run; if it does not, record the honest measured ranking rather than assert a wrong expectation). Verdict derived; `reconcile: no_authored`.

- [ ] **Step 1: Write the failing test** — add to `tests/test_sensitivity.py`: the study exists, `investigation: sbml-decomposition`, and after grade its canonical run has `computed_outcomes` with `evaluated_by: code`, `reconcile: no_authored`, PASS. FAILS until built.
- [ ] **Step 2: Add the evaluator(s)** in `evaluators.py` (registered): `sensitivity_ranking` — `@storeless`, runs the harness, PASS if the ranking is non-degenerate (top mu* ≥ `discrimination` × median mu*, default 2×) and finite; detail reports the top-3 parameters + that magnitudes are machine-specific. Keep a shared cache if more than one kind is added (Phase-3 lesson).
- [ ] **Step 3: Author the study.yaml** (schema 3, mirrors scaling-benchmarks): title, question (which parameters drive glycolysis, by how much?), baseline composite (`engine_monolithic` glycolysis nominal), the behavior test(s) with explicit thresholds + the machine-specificity caveat, a `visualizations:` entry (`address: local:SensitivityIndices`, `config: {}`, `covers:`), and parent_studies citing `ensemble-and-multirate` (it runs on the ensemble). No authored favorable `verdict_status`.
- [ ] **Step 4: Grade + write baseline** — `scripts/grade_sensitivity.py` creates one canonical run, `compute_outcomes` → `computed_outcomes`, and writes `visualizations/sensitivity_indices.json` (params, mu_star, sigma, ranking, qoi, machine). Update the registry assertion in `test_recovery_evaluators.py`.
- [ ] **Step 5: Verify** — `pytest tests/test_sensitivity.py tests/test_verdict_honesty.py tests/test_artifact_convention.py tests/test_recovery_evaluators.py -v` → PASS.
- [ ] **Step 6: Commit** — `git commit -m "feat(study): sensitivity-uq — seeded Morris parameter-importance ranking, derived verdict"`

---

### Task 3: Parameter-importance figure

**Files:** Create `viva_sbml_decomposition/visualizations/sensitivity_indices.py`; extend `tests/test_visualizations.py` (FIGS + an empty/degenerate graceful test).

**Interfaces:** Consumes `sensitivity_indices.json`; the viz framework. Produces `SensitivityIndices` — a horizontal ranked bar of `mu*` (parameter importance, highest first) with `sigma` shown as an interaction/nonlinearity band (clearly labelled as interaction, NOT a sampling-error CI); degrades on empty/degenerate data. Entry uses `address: local:SensitivityIndices` (broken_stub lesson).

- [ ] Steps mirror Phase 2/3 Task 3: failing viz test → implement reading the JSON → `test_visualizations.py` green → commit.

---

### Task 4: Wire into the investigation; re-derive; render

**Files:** Modify `investigation.yaml` (members + `at_a_glance` [quote any role string containing a colon — the Phase-3 YAML lesson] + authored `acceptance_criteria` pair), refresh `report.html`, `README.md`.

- [ ] Add `sensitivity-uq` to `studies:` + `at_a_glance:` + an `acceptance_criteria` pair (`{study: sensitivity-uq, behavior: <ranking test name>}`); re-run `write_investigation_acceptance`; confirm the new criterion resolves and `test_verdict_honesty` stays green. Render the report (figure appears post-merge per the env-worker limitation); `gen-readme --workspace .`. Commit.

---

### Task 5: Full-suite green + PR

- [ ] Run every CI gate (pytest tests/; lint-workspace; uv lock --check; gen-readme --check; check-workflow-freshness; render-check). Verify branch/HEAD + only this phase's commits. Push + `gh pr create --base main` (title `feat: Phase 4 — global sensitivity / UQ study (Morris parameter importance)`, body: the derived verdict, hand-rolled-Morris (no SALib dep) + reproducible-ranking grading, the QoI + parameter set, the figure-render note). **Do NOT merge.**

---

## Decisions flagged for your review

1. **Hand-rolled Morris, not SALib, not Sobol.** `SALib` is not installed and adds a heavy dependency; Morris elementary effects is a ~40-line seeded-numpy screening method that gives exactly the design's "importance ranking + uncertainty bands" (`mu*` importance, `sigma` interaction/nonlinearity) and is cheap (`r*(D+1)` runs). Sobol (variance-based, Saltelli sampling) is more rigorous but ~10× the runs and error-prone to hand-roll. *Alternative:* add `SALib` as a dep for Sobol+Morris — a separate dep PR.
2. **Grade the reproducible ranking, not raw magnitudes.** Morris indices on the deterministic ODE are reproducible given the seed, so the behavior test grades the ranking's non-degeneracy (and reproducibility), with magnitudes recorded as a machine-tagged baseline (last digits vary across BLAS, like recovery). This is stronger than Phase 3 (which could only grade qualitative wall-clock properties).
3. **Parameters = the Vmax enzyme capacities** (~a dozen `*Vm*` params), the biologically-meaningful "which enzyme limits flux" question, over a ±50% range. *Alternative:* all 85 parameters (Km/Keq included) — far more runs, and Km/Keq sensitivities are harder to interpret for a reviewer.
4. **QoI = a responsive hub quantity at steady state** (builder picks and documents, e.g. final ATP or a glycolytic flux). Flagged because a poorly-chosen QoI (one that barely responds) would make the ranking degenerate; Task 1 verifies responsiveness.
5. **Figure/report snapshot** carries the Phase-2 env-worker limitation (renders post-merge). Your call whether to force a pre-merge `uv sync` render.
