# Phase 6 — Rust-engine frontier: viva-core spike + report section Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (Native). Steps use checkbox (`- [ ]`) syntax.

**Goal:** Honestly situate the Rust engines (viva-core, viva-hybrid) on the performance frontier: a best-effort measured viva-core spike plus a committed, code-graded report section that says where a Rust path pays off (bigraph dispatch) and where it does not (ODE integration).

**Architecture:** A probe harness attempts to run our one portable arm (`native_composite`, the hand-ported glycolysis RHS) through viva-core's engine and writes a machine-tagged `rust_frontier.json` recording either the measured number **or** the honest "requires a Rust-side process port" finding. A code-graded `rust-engine-frontier` study in the **decomposition** investigation reads that committed JSON — never importing Rust — and grades the honest characterization (no false ≥5× ODE-acceleration claim). A figure renders the dispatch-vs-ODE frontier with viva-hybrid's negative result cited.

**Tech Stack:** Python 3.12, tellurium/process-bigraph (existing), viva-core (Rust, maturin — `~/code/viva-core`, prebuilt `_native.cpython-312-darwin.so`), discovery_engine 0.3.1.

**Spec:** `docs/superpowers/specs/2026-11-13-sbml-decomposition-clarity-and-capabilities-design.md` (§6 Phase 6, G6).

## Global Constraints

- **No Rust toolchain in CI.** No test may import `viva_core` or require a build; every CI-run test reads the committed `rust_frontier.json` or exercises the probe's no-viva_core degradation path. The spike itself is run manually (`scripts/probe_rust_frontier.py`), its result committed.
- **Honest negatives, derived verdicts.** The decomposition investigation is computed `passing`; a synthesis/report study must not flip that dishonestly. The grade passes on the honest characterization, not on a Rust win. No authored favorable verdict.
- **No AI attribution** in commits or PR. **No merge** — open PR, await the owner.
- Loader convention (`_tellurium.load`), evaluator registry (`register_evaluators`), figure `address: local:ClassName` (not broken_stub), `PYTHONPATH=.`, canonical venv.

## Review Focus

- **CI imports no Rust.** A reviewer runs the suite with no viva_core installed; confirm no test fails for lack of it (probe degrades, JSON is committed). *(Task 1 test.)*
- **The probe never raises.** viva_core missing, `.so` ABI mismatch, or native_composite rejected by the Rust registry all return a structured "not portable" result, not a traceback. *(Task 1 test.)*
- **The grade bites on dishonesty.** A `rust_frontier.json` claiming a ≥5× ODE speedup (a false Rust win) must FAIL the evaluator; the honest "dispatch-only / requires port" result must PASS. *(Task 2 test.)*
- **Decomposition stays honestly passing.** Adding this study re-derives the investigation without flipping or faking its verdict. *(Task 2 verification.)*
- **Figure degrades on empty.** Renders with the honest "requires port" JSON (ode_speedup null) and with an empty state. *(Task 2 test.)*

---

### Task 1: viva-core spike probe (best-effort, outside CI)

**Files:**
- Create: `viva_sbml_decomposition/harness/rust_frontier.py`
- Create: `scripts/probe_rust_frontier.py`
- Create: `viva_sbml_decomposition/visualizations/rust_frontier.json` (committed baseline, written by the script)
- Test: `tests/test_rust_frontier.py`

**Interfaces:**
- Produces: `rust_frontier_probe(*, viva_core_python: str | None = None) -> dict` with keys `viva_core_available: bool`, `native_composite_runs: bool`, `ode_speedup: float | None`, `dispatch_speedup: float | None`, `dispatch_note: str`, `ode_note: str`, `viva_hybrid_note: str`, `requires_port: bool`, `caveat: str`, `machine: str`. Never raises.

- [ ] **Step 1: Write the failing test** `tests/test_rust_frontier.py`:
  - `test_probe_degrades_without_viva_core`: call `rust_frontier_probe(viva_core_python="/nonexistent")`; assert it returns a dict, `viva_core_available is False`, `native_composite_runs is False`, `requires_port is True`, `ode_speedup is None`, and `dispatch_note`/`ode_note` are non-empty strings. No exception.
  - `test_committed_baseline_is_well_formed`: load `visualizations/rust_frontier.json`; assert all interface keys present and typed; assert `ode_speedup is None or isinstance(ode_speedup,(int,float))`; assert `caveat` non-empty and `machine` non-empty.
- [ ] **Step 2: Run → FAIL** (`ModuleNotFoundError` / missing JSON). `PYTHONPATH=. pytest tests/test_rust_frontier.py -q`.
- [ ] **Step 3: Implement `rust_frontier_probe`** in `harness/rust_frontier.py`: optionally prepend `viva_core_python` (default `~/code/viva-core/python`) to `sys.path`; `try: import viva_core` → on any `Exception`, return the honest not-available result (`requires_port=True`, notes explaining viva-core runs only processes in its native Rust registry). If import succeeds, attempt to execute `native_composite(model="glycolysis")` through viva-core's engine; on any failure return `native_composite_runs=False, requires_port=True` with the rejection reason; on success time it against the Python native monolith and set `ode_speedup`. Always set the cited-context notes (`dispatch_note` ≈ "viva-core accelerates bigraph dispatch ~1.27× scalar, regresses on array-heavy state", `ode_note` ≈ "does not accelerate ODE integration", `viva_hybrid_note` ≈ "viva-hybrid: rigorous negative — ~2× over numba, no regime both ≥5× and faithful to exact SSA") and `machine` (platform tag). Wrap the whole body so it never raises.
- [ ] **Step 4: Run the real spike** `scripts/probe_rust_frontier.py` (prepends `~/code/viva-core/python`, calls the probe, writes `visualizations/rust_frontier.json`). Record the actual outcome (expected: `requires_port=True` unless the native arm happens to be registered). This is the only step that touches Rust; it is not run in CI.
- [ ] **Step 5: Run → PASS** both tests. `PYTHONPATH=. pytest tests/test_rust_frontier.py -q`.
- [ ] **Step 6: Commit** harness + script + committed JSON + test.

### Task 2: frontier evaluator, study, figure

**Files:**
- Modify: `viva_sbml_decomposition/evaluators.py` (+ `_rust_frontier_honesty`, register `rust_frontier_honesty`)
- Modify: `tests/test_recovery_evaluators.py` or the decomposition evaluator test (add the new kind to the registered-set assertion)
- Create: `workspace/investigations/sbml-decomposition/studies/rust-engine-frontier/study.yaml`
- Create: `scripts/grade_rust_frontier.py`
- Create: `viva_sbml_decomposition/visualizations/rust_frontier.py` (`RustEngineFrontier`, `address: local:RustEngineFrontier`)
- Modify: `tests/test_visualizations.py` (FIGS + a degrade test)
- Modify: `workspace/investigations/sbml-decomposition/investigation.yaml` (studies + acceptance_criteria + at_a_glance)
- Test: `tests/test_rust_frontier.py` (+ study-is-graded test)

**Interfaces:**
- Consumes: `rust_frontier_probe` result committed as `rust_frontier.json` (Task 1).
- Produces: evaluator kind `rust_frontier_honesty`; figure class `RustEngineFrontier`.

- [ ] **Step 1: Write the failing tests**:
  - In `test_rust_frontier.py`: `test_honesty_grade_passes_on_honest_json` (feeds the committed honest JSON → PASS) and `test_honesty_grade_fails_on_false_ode_win` (feeds a JSON with `ode_speedup=6.0, requires_port=False` → FAIL). Drive these through the evaluator directly.
  - In `test_visualizations.py`: add `("rust_frontier","RustEngineFrontier")` to FIGS and a `test_rust_frontier_figure_degrades_on_empty` (empty state + honest-null state both build a valid plotly spec).
  - `test_rust_frontier_study_is_code_graded_and_passes`: canonical run has `computed_outcomes` with `result == "PASS"`, `evaluated_by == "code"`, `reconcile == "no_authored"`.
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement**:
  - `_rust_frontier_honesty` (`@storeless`): reads the committed JSON; PASS iff the characterization is honest — `requires_port or (ode_speedup is None) or ode_speedup < 5.0` (no false ≥5× ODE-acceleration claim) AND `ode_note`/`dispatch_note` present; FAIL if a ≥5× ODE win is asserted while `requires_port` is False. `measured_value` = `ode_speedup or 0.0`. Register `rust_frontier_honesty`.
  - `RustEngineFrontier` figure: a two-region frontier — "dispatch (Rust helps)" vs "ODE integration (Rust does not)" — showing viva-core's dispatch ~1.27×, the ODE result (our composite's `requires_port` or measured `ode_speedup`), and viva-hybrid's ~2× negative, with caveat annotations. Reads `rust_frontier.json`; `generate_state` degrades to empty.
  - `rust-engine-frontier` study.yaml (decomposition inv), behavior_test kind `rust_frontier_honesty`, viz `local:RustEngineFrontier`, parent `fully-native` (or the native-arm study), honest narrative: dispatch-vs-ODE, viva-hybrid negative, explicitly not a Rust ODE kernel.
  - `scripts/grade_rust_frontier.py`: canonical run + `compute_outcomes` + `write_gate_evaluator` + `roll_up_verdict`; mirrors `grade_calibration.py`.
- [ ] **Step 4: Grade** `PYTHONPATH=. python scripts/grade_rust_frontier.py` → gate `passed`; study.yaml carries `computed_outcomes`.
- [ ] **Step 5: Wire + re-derive**: add the study to the decomposition investigation (studies, acceptance_criteria, at_a_glance); `write_investigation_acceptance`; confirm decomposition `computed_verdict_status` stays `passing` and `test_verdict_honesty` green. Render the decomposition report.
- [ ] **Step 6: Run → PASS** (new tests + `test_verdict_honesty` + `test_artifact_convention` + the evaluator-registry test).
- [ ] **Step 7: Commit** evaluator + study + script + figure + baseline + wiring + report.

### Task 3: whole-branch verify, review, PR

- [ ] **Step 1:** Revert any study/report churn; run the 5 CI gates individually (lint, uv-lock, gen-readme --check, freshness, render-check) + the full suite `PYTHONPATH=. pytest tests/ -q` → all green, no Rust required.
- [ ] **Step 2:** Final whole-branch review (fresh reviewer, most capable model): focus = the Review Focus list (CI imports no Rust; probe never raises; grade bites on a false ODE win; decomposition stays honestly passing; figure degrades). Re-grade by effect; fix Critical/Important in one TDD pass; ledger minors.
- [ ] **Step 3:** Push `cleanup/phase6-rust-engine-frontier`; `gh pr create` against main (no merge). Honest caveat section in the body: the Rust number is best-effort, not CI-gated.

## Self-Review

- **Spec coverage:** measured spike (Task 1), report section + figure + viva-hybrid negative (Task 2), not-a-Rust-ODE-kernel scope (study narrative + grade). ✓
- **Review Focus → tests:** CI-no-Rust + probe-degrades (Task 1 tests), grade-bites + figure-degrades (Task 2 tests), decomposition-honest (Task 2 Step 5). ✓
- **Type consistency:** `rust_frontier_probe` dict keys identical across harness, script, evaluator, figure. ✓
- **Proportion:** three tasks, one new harness + evaluator + study + figure — matches the Phase 2–5 shape. ✓
- **Honest-negative risk:** the headline deliverable may be "viva-core cannot run our composite without a Rust-side port." That is the finding, graded PASS as an honest characterization — not a failure of the phase.
