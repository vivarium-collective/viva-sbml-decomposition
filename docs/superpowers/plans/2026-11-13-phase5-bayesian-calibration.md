# Phase 5 — Bayesian Calibration Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one new study to the **recovery** investigation that extends the recovery pipeline from a point estimate to a **posterior**: given noisy synthetic data from the known mass-action model, infer the posterior over its three rate constants (k1, k2, k2r), and verify the calibration is honest (the true parameters are covered by the credible intervals and the data is informative). The verdict is **derived** and graded on **reproducible** Bayesian quality properties (coverage + informativeness), computed exactly on a grid.

**Architecture:** A **grid-based exact posterior** — evaluate the Gaussian likelihood of the noisy data on a grid over (k1, k2, k2r) under a uniform prior, normalize to the posterior, and read marginals / MAP / 95% credible intervals + a posterior-predictive band. No Bayesian dependency (`emcee`/`pymc`/`numpyro` are not installed and are heavy), no MCMC convergence to validate, fully deterministic given the data seed — the honest, reproducible choice for a 3-parameter, well-identified problem (mass-action is recoverable, as `recover-mass-action` showed). Reuses `SyntheticTargetGenerator` for the (noisy) data and tellurium for the forward simulations.

**Tech Stack:** Python 3.12, tellurium (forward sim), numpy + scipy (grid likelihood, no new dep), `SyntheticTargetGenerator`, `discovery_engine` (nested `study.*`), the viz framework, pytest.

**Spec:** `docs/superpowers/specs/2026-11-13-sbml-decomposition-clarity-and-capabilities-design.md` (Phase 5 of §6, capability 4 → recovery investigation). Phases 0–4 merged (`main` @ `586cd6b`).

## Global Constraints

- Worktree only (`~/code/viva-sbml-decomposition--cleanup`, branch `cleanup/phase5-bayesian-calibration` off `origin/main` @ `586cd6b`); canonical read-only. Verify branch + HEAD before each commit.
- No AI attribution; squash-merge; no auto-merge; one PR.
- Canonical venv + `PYTHONPATH=.`; nested `discovery_engine.study.*`; derive-first (no hand-authored favorable `verdict_status`; `test_verdict_honesty` stays green).
- **No new dependency.** Grid + Gaussian likelihood in numpy/scipy. Do NOT add `emcee`/`pymc`.
- **Reproducible posterior.** The synthetic data's observation noise is seeded, and the grid is deterministic, so the posterior / MAP / credible intervals reproduce exactly on a machine (last digits vary across BLAS/platforms — grade coverage + informativeness, not a raw posterior value).
- **Code-graded study** → commits `computed_outcomes` (+ a committed posterior JSON baseline), per the registry-derived convention test. New `measure.kind` evaluators register in `evaluators.py`.
- **This study lands in the RECOVERY investigation** (`discovery-sbml-recovery`), which reopens. Its committed `computed_verdict_status` is `failing` (2/5 rungs recovered); adding a passing calibration study must NOT make the investigation overclaim — recovery has **no authored** `verdict_status` (Phase 0 removed it), so the engine's re-derived verdict simply stands. Confirm `test_verdict_honesty` stays green after wiring.
- **Loader convention:** route any model load through `_tellurium.load`, not bare `te.loada` (the Phase-4 lesson).

## Review Focus

- **The calibration silently overclaims coverage** (the 95% CI does not actually contain the true value, but the grade passes) → Task 2 grades that each true parameter lies within its 95% credible interval, computed from the marginal posterior CDF; the test asserts coverage on the committed posterior, and a shifted/biased posterior fails.
- **The posterior is uninformative but passes** (the data barely constrains the params, so the "posterior" ≈ the prior, yet coverage trivially holds) → Task 2 also grades informativeness: each 95% CI width is clearly narrower than the prior width (the data constrained the parameter), so a flat posterior fails.
- **The likelihood/grid is wrong and the MAP is off** (a bug in the Gaussian log-likelihood or the grid indexing moves the posterior mass away from the truth) → Task 1's test asserts the MAP is close to the true parameters on noiseless-ish data (small sigma), and that the posterior normalizes to 1; a likelihood/index bug shows up as a MAP far from truth.
- **Noise makes it non-reproducible** (unseeded data → different posterior each run) → Task 1 threads a fixed `seed` into `SyntheticTargetGenerator`; its test asserts two runs at the same seed give identical marginals.
- **The predictive band is drawn as the parameter posterior, not the data space** (a common confusion) → Task 3's posterior-predictive check propagates posterior draws through the forward model and shows the band in observation space over the data; the viz test checks it builds.

---

## File Structure

- `viva_sbml_decomposition/harness/calibration.py` — **new**, grid posterior + predictive (Task 1).
- `viva_sbml_decomposition/visualizations/calibration_posterior.json` — **new**, committed baseline (Task 2).
- `workspace/investigations/discovery-sbml-recovery/studies/bayesian-calibration/study.yaml` — **new** study (Task 2).
- `viva_sbml_decomposition/evaluators.py` — **+1–2** kinds (`calibration_coverage`, optionally `calibration_informative`) (Task 2).
- `scripts/grade_calibration.py` — **new**, runs the grid + grades + writes the baseline (Task 2).
- `viva_sbml_decomposition/visualizations/calibration_posterior.py` — **new** posterior + predictive figure (Task 3).
- `discovery-sbml-recovery/investigation.yaml`, `report.html`, `README.md` — wired + rendered (Task 4).
- `tests/test_calibration.py` — **new**; `test_verdict_honesty.py`, `test_artifact_convention.py`, `test_recovery_evaluators.py`, `test_visualizations.py` stay green (Task 4).

---

### Task 1: Grid-posterior calibration harness

**Files:** Create `viva_sbml_decomposition/harness/calibration.py`; test `tests/test_calibration.py`.

**Interfaces:**
- Consumes: `processes.recovery.SyntheticTargetGenerator` (config `{antimony, t_end, n_points, observed, sigma}` → noisy target trajectories, seeded); the mass-action antimony + `MASS_ACTION_META` (`k_true = {k1:0.5, k2:0.3, k2r:0.1}`, species A/B/C, t_end 10); `_tellurium.load` for forward sims; numpy for the grid; a Gaussian likelihood.
- Produces: `grid_calibration(*, params=("k1","k2","k2r"), grid_points=11, span=0.5, sigma=0.05, n_points=40, seed=0) -> dict` → `{"params", "true": {p: v}, "grid": {p: [values]}, "marginals": {p: [posterior over the grid]}, "map": {p: v}, "ci95": {p: [lo, hi]}, "prior_width": {p: w}, "predictive": {"time": [...], "observed": {sp: {"data": [...], "lo": [...], "hi": [...]}}}, "machine"}`. Grid posterior ∝ exp(sum Gaussian log-likelihood of the noisy data) under a uniform prior over `±span` around each true value; marginals by summation; CI from the marginal CDF; predictive band from posterior-weighted forward sims.

- [ ] **Step 1: Write the failing test** `tests/test_calibration.py`:

```python
def test_grid_posterior_recovers_and_is_reproducible():
    from viva_sbml_decomposition.harness.calibration import grid_calibration
    a = grid_calibration(grid_points=9, sigma=0.03, seed=0)
    b = grid_calibration(grid_points=9, sigma=0.03, seed=0)
    for p in a["params"]:
        lo, hi = a["ci95"][p]
        assert lo <= a["true"][p] <= hi                 # 95% CI covers truth
        assert abs(a["map"][p] - a["true"][p]) < 0.5 * a["true"][p]  # MAP near truth
        assert a["marginals"][p] == b["marginals"][p]    # seeded -> reproducible
    assert abs(sum(a["marginals"][a["params"][0]]) - 1.0) < 1e-9     # normalized

def test_predictive_band_is_in_observation_space():
    from viva_sbml_decomposition.harness.calibration import grid_calibration
    r = grid_calibration(grid_points=7, sigma=0.05, seed=0)
    pc = r["predictive"]; sp = r["params"]  # observation-space band over the data
    assert pc["time"] and all(k in pc["observed"] for k in ("A", "B", "C"))
    one = pc["observed"]["A"]
    assert len(one["lo"]) == len(one["hi"]) == len(pc["time"])
```

- [ ] **Step 2: Run it, verify it fails** — `pytest tests/test_calibration.py -v` → FAIL (harness missing).
- [ ] **Step 3: Implement `grid_calibration`** — generate the noisy data once (`SyntheticTargetGenerator`, seeded, `sigma`); build the `grid_points`-per-axis grid over `±span` around each true k; for each grid cell simulate the mass-action model with those k's (`_tellurium.load` + `simulate`) and accumulate the Gaussian log-likelihood vs the data; exponentiate (stabilized) → joint posterior; marginals, MAP, 95% CI (marginal CDF), prior width; predictive band by simulating a sample of high-posterior cells and taking per-time percentiles in observation space. Deterministic given `seed`.
- [ ] **Step 4: Run it** — `pytest tests/test_calibration.py -v` → PASS (coarse grid; seconds).
- [ ] **Step 5: Commit** — `git commit -m "feat(calibration): grid-posterior Bayesian calibration of the mass-action rate constants (no MCMC dep)"`

---

### Task 2: The bayesian-calibration study + derived verdict + posterior baseline

**Files:** Create `studies/bayesian-calibration/study.yaml`, `scripts/grade_calibration.py`; modify `evaluators.py` (+1–2 kinds); extend `tests/test_calibration.py`; update `tests/test_recovery_evaluators.py` registry set.

**Interfaces:**
- Consumes: Task 1's `grid_calibration`; the `kind`→evaluator registry; `discovery_engine` compute/verdict (nested).
- Produces: a code-graded study (in `discovery-sbml-recovery`) whose canonical run carries `computed_outcomes` (+ committed `calibration_posterior.json`), graded on **reproducible Bayesian quality**: (a) `calibration_coverage` — all three true rate constants lie inside their 95% credible intervals; (b) the posterior is **informative** — each CI width is clearly below the prior width (shared-cache both kinds, Phase-3 lesson). Verdict derived; `reconcile: no_authored`.

- [ ] **Step 1: Write the failing test** — add to `tests/test_calibration.py`: the study exists, `investigation: discovery-sbml-recovery`, and after grade its canonical run has `computed_outcomes` (`evaluated_by: code`, `reconcile: no_authored`, PASS). FAILS until built.
- [ ] **Step 2: Add the evaluator(s)** in `evaluators.py` (registered): `calibration_coverage` — `@storeless`, runs `grid_calibration`, PASS if every true param is within its 95% CI and each CI is informative (width < `informative_ratio` × prior width, default 0.8); detail reports MAP + CIs + that magnitudes are machine-specific.
- [ ] **Step 3: Author the study.yaml** (schema 3, mirrors a recovery study; `investigation: discovery-sbml-recovery`): title, question (can we calibrate the mass-action rate constants with honest uncertainty?), the behavior test(s), a `visualizations:` entry (`address: local:CalibrationPosterior`), parent_studies citing `recover-mass-action` (point estimate → posterior). No authored favorable `verdict_status`.
- [ ] **Step 4: Grade + write baseline** — `scripts/grade_calibration.py` creates one canonical run, `compute_outcomes` → `computed_outcomes`, writes `visualizations/calibration_posterior.json`. Update the registry assertion.
- [ ] **Step 5: Verify** — `pytest tests/test_calibration.py tests/test_verdict_honesty.py tests/test_artifact_convention.py tests/test_recovery_evaluators.py -v` → PASS.
- [ ] **Step 6: Commit** — `git commit -m "feat(study): bayesian-calibration — grid posterior with honest coverage, derived verdict (recovery investigation)"`

---

### Task 3: Posterior + predictive figure

**Files:** Create `viva_sbml_decomposition/visualizations/calibration_posterior.py`; extend `tests/test_visualizations.py` (FIGS + an empty graceful test).

**Interfaces:** Consumes `calibration_posterior.json`; the viz framework. Produces `CalibrationPosterior` — the three marginal posteriors with the true value + 95% CI marked, and a posterior-predictive check (the observation-space band over the data). `address: local:CalibrationPosterior`; degrades on empty.

- [ ] Steps mirror Phase 2–4 Task 3: failing viz test → implement reading the JSON → `test_visualizations.py` green → commit.

---

### Task 4: Wire into the RECOVERY investigation; re-derive; render

**Files:** Modify `discovery-sbml-recovery/investigation.yaml` (members + any at_a_glance [quote colon-containing roles — Phase-3 lesson] + authored `acceptance_criteria` pair), refresh its `report.html`, `README.md`.

- [ ] Add `bayesian-calibration` to the recovery investigation's `studies:` + an `acceptance_criteria` pair (`{study: bayesian-calibration, behavior: <coverage test name>}`); re-run `write_investigation_acceptance`; **confirm the recovery `computed_verdict_status` stays honest** (a passing calibration study does not erase the failing recovery rungs; recovery has no authored verdict, so no overclaim) and `test_verdict_honesty` is green. Render the recovery `report.html` (figure post-merge per the env-worker limitation); `gen-readme`. Commit.

---

### Task 5: Full-suite green + PR

- [ ] Run every CI gate (pytest tests/; lint-workspace; uv lock --check; gen-readme --check; check-workflow-freshness; render-check). Verify branch/HEAD + only this phase's commits. Push + `gh pr create --base main` (title `feat: Phase 5 — Bayesian calibration study (grid posterior, recovery investigation)`, body: the derived verdict, grid-posterior (no MCMC dep) + reproducible coverage/informativeness grading, the figure-render note). **Do NOT merge.**

---

## Decisions flagged for your review

1. **Grid-based exact posterior, not MCMC/ABC/a new dependency.** No Bayesian library is installed; `emcee`/`pymc` are heavy deps. For 3 well-identified parameters a grid gives the exact posterior with no MCMC convergence to validate and full determinism (the Phase-4 lesson: hand-rolled inference math must be unarguably correct — a grid is). *Alternative:* add `emcee` (affine-invariant MCMC) as a dep for higher-dimensional problems later — a separate dep PR. *Cost:* the grid is 3-D only; it would not scale to many parameters, but mass-action has three.
2. **Target = the mass-action recovery model with added observation noise.** It is the recovery ladder's floor and is fully identifiable (recover-mass-action: nrmse ~1e-9), so the posterior is well-behaved and coverage is a meaningful test. Noise (`sigma≈0.05`) is added so the posterior is non-degenerate (width reflects the data's information). *Alternative:* a harder rung (enzyme-kinetics) — but its recovery already fails, so a calibration there would conflate two failure modes.
3. **Grade coverage + informativeness (reproducible), not a raw posterior value.** The true params inside the 95% CIs is the honest calibration-quality check; informativeness rules out a trivial flat-posterior pass. Both reproduce given the seed; magnitudes are a machine-tagged baseline.
4. **Lands in the recovery investigation, which stays honestly `failing`.** Adding a passing calibration study does not flip the recovery verdict (the failing rungs remain, and recovery carries no authored verdict to overclaim). Task 4 confirms `test_verdict_honesty` green.
5. **Figure/report snapshot** carries the Phase-2 env-worker limitation (renders post-merge). Your call whether to force a pre-merge `uv sync` render.
