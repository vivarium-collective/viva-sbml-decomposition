# Phase 1 — Code-Consistency Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the code-consistency debt the design (G4) flags — one tellurium loader instead of ~8 scattered `te.loada` sites, one recovery composite instead of two (with the identity baseline made explicit so it can never again mask a false PASS), the vestigial always-`None` accuracy placeholder dropped, and one documented + enforced sim-artifact convention — without changing any science, verdict, or rendered result.

**Architecture:** Pure consolidation/DRY. Each change is behavior-preserving and guarded by a test: a loader seam that every semantic load routes through (timing-measurement sites excepted, by design), a single generic `recover_candidate` composite, a slimmer `ComposabilityScore`, and a committed convention test. No investigation YAML verdict, grade, or figure changes.

**Tech Stack:** Python 3.12, process-bigraph composites, tellurium/libroadrunner, `discovery_engine` (nested `study.*` API — see Global Constraints), pytest.

**Spec:** `docs/superpowers/specs/2026-11-13-sbml-decomposition-clarity-and-capabilities-design.md` (Phase 1 of §6, goal G4). Phase 0 (merged, PR #23) is the baseline.

## Global Constraints

- Worktree only (`~/code/viva-sbml-decomposition--cleanup`, branch `cleanup/phase1-code-consistency` off `origin/main` @ `518d0f8`); canonical checkout read-only. Verify `git branch --show-current` + HEAD before each commit.
- No AI attribution in commits/PRs. Squash-merge; no auto-merge; one PR for the phase.
- Run all tooling with the canonical venv **and** `PYTHONPATH=.` so the worktree source wins: `PYTHONPATH=. /Users/eranagmon/code/viva-sbml-decomposition/.venv/bin/python …`. For CLI tools also prepend `/Users/eranagmon/code/viva-sbml-decomposition/.venv/bin` to `PATH`.
- **Engine API is the nested `study.*` layout.** CI resolves `branch=main` deps to current tips where the flat `discovery_engine.study_verdict`/`study_evaluator`/`study_outcomes` modules are removed. Any new engine import uses `discovery_engine.study.verdict` / `.study.evaluator` / `.study.outcomes` (nested-first, with a flat fallback only if an import must also run on an older engine). This is why Phase 0 added those fallbacks.
- **Do not change any science, verdict, grade, figure, or investigation/study YAML outcome.** Phase 1 is consistency only. The Phase 0 honesty tests (`tests/test_verdict_honesty.py`) and reproducibility test must stay green unchanged.
- **Cross-platform reproducibility:** recovery's raw fit floats are not bit-reproducible across BLAS/platforms; only the thresholded PASS/FAIL verdict is. Do not add any test asserting raw-float reproducibility.

## Review Focus

- **The tellurium timing baseline gets silently corrupted** (a loader that adds any work replaces the two `_multiorgan.py` sites that deliberately time raw `te.loada` as the monolithic-build benchmark) → Task 1 excludes those sites by design and its test asserts they still call `te.loada` directly. Most likely to bite; it would quietly skew a headline scaling number.
- **The identity baseline silently returns as a real "recovery"** (a caller invokes the recovery composite with an empty candidate and the oracle runs, grading as a pass) → Task 2 makes the identity baseline require an explicit opt-in and its test asserts an empty candidate without that opt-in raises, not silently runs the oracle.
- **A consumer actually read the dropped `accuracy` field** → Task 3's test asserts `decision_table` and `multiorgan_scaling` still render (they read the real accuracy metrics from `load_metrics`, not the `ComposabilityScore` field), and a grep step confirms no reader of `ComposabilityScore(...).accuracy`.
- **The loader import creates a circular import** (`_tellurium` imported by a module it imports) → Task 1 keeps `_tellurium` dependency-free (imports only `tellurium`) and its test imports it in isolation.
- **A study's committed artifact silently disappears** (a decomposition `runs.db` or a recovery `computed_outcomes` goes missing and nothing notices) → Task 4's convention test asserts every emitter-backed study commits a `runs.db` and every code-graded recovery study commits `computed_outcomes`.

---

## File Structure

- `viva_sbml_decomposition/_tellurium.py` — **new**, the single tellurium model-load seam (Task 1).
- `viva_sbml_decomposition/{processes/submodel.py, processes/recovery.py, evaluators.py, composites/_multiorgan.py, visualizations/recovery_rejection.py}` + `scripts/recover.py` — route semantic loads through the loader (Task 1).
- `viva_sbml_decomposition/composites/recovery.py` — drop `recover_mass_action`; make `recover_candidate`'s identity baseline explicit (Task 2).
- `viva_sbml_decomposition/composites/__init__.py` — update the stale comment (Task 2).
- `viva_sbml_decomposition/harness/composability.py` — drop the `accuracy` field from `ComposabilityScore` + all six rows (Task 3).
- `docs/CONVENTIONS.md` — **new**, the sim-artifact convention (Task 4).
- `tests/test_tellurium_loader.py`, `tests/test_artifact_convention.py` — **new** (Tasks 1, 4).
- `tests/test_recovery_capabilities.py`, `tests/test_composability.py` — updated (Tasks 2, 3).

---

### Task 1: Single tellurium loader seam

**Files:**
- Create: `viva_sbml_decomposition/_tellurium.py`
- Modify (semantic load sites → loader): `viva_sbml_decomposition/processes/submodel.py:28`, `viva_sbml_decomposition/processes/recovery.py:33,171`, `viva_sbml_decomposition/evaluators.py:188,214`, `viva_sbml_decomposition/composites/_multiorgan.py:41`, `viva_sbml_decomposition/visualizations/recovery_rejection.py:80`, `scripts/recover.py:47`
- **Leave raw (timing baseline — do NOT route through the loader):** `viva_sbml_decomposition/composites/_multiorgan.py:249,262` (inside `_median_time(...)`, they measure raw `te.loada` build time)
- Create test: `tests/test_tellurium_loader.py`

**Interfaces:**
- Produces: `viva_sbml_decomposition._tellurium.load(antimony: str) -> roadrunner.RoadRunner` — the single place a flat Antimony/SBML string becomes a loaded libroadrunner model. Thin by design (one `tellurium.loada` call); it exists so future integrator settings / error handling land in one seam. Imports only `tellurium`.

- [ ] **Step 1: Write the failing test** `tests/test_tellurium_loader.py`:

```python
def test_load_returns_simulatable_model():
    from viva_sbml_decomposition import _tellurium
    rr = _tellurium.load("species S1=1; S1 -> ; k*S1; k=0.1;")
    out = rr.simulate(0, 1, 3)
    assert out.shape[0] == 3  # a real libroadrunner result

def test_no_bare_teloada_outside_loader_and_timing():
    # Every semantic load routes through _tellurium.load; the only raw te.loada
    # calls allowed are the loader itself and the multiorgan timing baseline.
    import pathlib, re
    allowed = {"viva_sbml_decomposition/_tellurium.py",
               "viva_sbml_decomposition/composites/_multiorgan.py"}  # timing sites
    offenders = []
    for p in pathlib.Path("viva_sbml_decomposition").rglob("*.py"):
        if str(p) in allowed:
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if re.search(r"\bte\.loada\(", line) and not line.lstrip().startswith("#"):
                offenders.append(f"{p}:{i}")
    assert not offenders, f"bare te.loada outside the loader: {offenders}"
```

- [ ] **Step 2: Run it, verify it fails** — `pytest tests/test_tellurium_loader.py -v`: first test FAILS (`_tellurium` missing), second FAILS (bare `te.loada` sites still present).
- [ ] **Step 3: Create the loader** `viva_sbml_decomposition/_tellurium.py`: `import tellurium as te` and `def load(antimony: str): return te.loada(antimony)`, with a module docstring stating it is the single tellurium load seam and that `_multiorgan.py`'s timing baseline intentionally bypasses it.
- [ ] **Step 4: Route the semantic sites** — in each modified file, replace `te.loada(x)` with `from viva_sbml_decomposition._tellurium import load` (module-level) then `load(x)`. Preserve chaining (`recovery.py:171` → `load(antimony).getCurrentSBML()`). Remove the now-unused `import tellurium as te` where no raw call remains (keep it in `_multiorgan.py`). Add a one-line comment at `_multiorgan.py:249,262` noting the raw `te.loada` is the timing baseline.
- [ ] **Step 5: Run it** — `pytest tests/test_tellurium_loader.py -v` → PASS (both).
- [ ] **Step 6: Commit** — `git add viva_sbml_decomposition/_tellurium.py viva_sbml_decomposition tests/test_tellurium_loader.py scripts/recover.py && git commit -m "refactor(tellurium): route semantic model loads through a single _tellurium.load seam"`

---

### Task 2: One recovery composite; identity baseline made explicit

**Blast radius (verified):** the `recover_mass_action` composite is referenced only by its own definition (`composites/recovery.py:29,47`), a stale `__init__.py:19` comment, `tests/test_recovery_capabilities.py` (import + 2 tests), and **one study config** — `workspace/investigations/discovery-sbml-recovery/studies/recover-mass-action/study.yaml` (`composite:` line 33, `base_model:` line 176; plus a historical run name at line 135). Every *other* recovery study already uses `recover_candidate`.

**Files:**
- Modify: `viva_sbml_decomposition/composites/recovery.py` (remove `recover_mass_action`; make `recover_candidate`'s identity baseline explicit & safe)
- Modify: `viva_sbml_decomposition/composites/__init__.py:19` (stale comment)
- Modify: `workspace/investigations/discovery-sbml-recovery/studies/recover-mass-action/study.yaml` (`composite:`/`base_model:` → `recover_candidate`)
- Modify: `tests/test_recovery_capabilities.py` (drop `recover_mass_action` import/tests; assert the guard)

**Interfaces:**
- Consumes: `recover_candidate(core=None, target=..., candidate_model="", n_steps=...)` (existing). Other recovery studies already set `composite: …recover_candidate`.
- Produces: `recover_candidate` as the single recovery composite, with the identity baseline (candidate == oracle) made **explicit** via a new `baseline: bool = False` parameter. `recover_mass_action` is removed.

**Decision to VERIFY before choosing the guard (implementer, first step):** the recovery studies' baseline runs set `composite: …recover_candidate` — determine whether any invoke it with an **empty** `candidate_model` (relying on the old oracle default). Note Phase 0 already removed the *grading* footgun: grades come from `@recovered` (the real search in `evaluators.py`), not from the composite's candidate, and `tests/test_verdict_honesty.py` enforces that. So Phase 1's job is to stop the *composite default* from silently running the oracle, **without breaking a legitimate baseline run.** Thread `baseline=True` into any study config that genuinely wants the identity wiring check; a bare empty candidate with no opt-in must raise. If threading `baseline` into a study config is not supported by the run path, fall back to: keep the default but tag identity runs so they cannot be read as recovery, and record the choice as a ruling.

- [ ] **Step 1: Verify the baseline-run path** — `grep -rn "candidate_model\|baseline\|composite:" workspace/investigations/discovery-sbml-recovery/studies/*/study.yaml` and inspect how `composite:`/params reach the composite; confirm none of the committed baseline runs depends on an empty-candidate oracle default in a way the guard would break. Record the finding as a one-line ruling in the ledger before editing.
- [ ] **Step 2: Write the failing test** — in `tests/test_recovery_capabilities.py`, replace the `recover_mass_action` import and `test_recover_mass_action_composite_builds` with:

```python
def test_recover_candidate_rejects_implicit_identity():
    import pytest
    from viva_sbml_decomposition.composites.recovery import recover_candidate
    with pytest.raises(ValueError):
        recover_candidate(target="recover-mass-action")  # empty candidate, no opt-in
    assert recover_candidate(target="recover-mass-action", baseline=True)  # explicit identity builds
    assert recover_candidate(target="recover-mass-action",
                             candidate_model="species A=1; A -> ; k*A; k=0.1;")  # explicit candidate builds

def test_recover_mass_action_composite_removed():
    import viva_sbml_decomposition.composites.recovery as r
    assert not hasattr(r, "recover_mass_action")
```

- [ ] **Step 3: Run it, verify it fails** — `pytest tests/test_recovery_capabilities.py -k "rejects_implicit_identity or composite_removed" -v` → FAIL.
- [ ] **Step 4: Edit `recovery.py`** — delete the `recover_mass_action` `@composite_generator` block. Add `baseline: bool = False` to `recover_candidate`'s signature and `parameters`; body: `model = candidate_model or (entry["antimony"] if baseline else None)`; `if not model: raise ValueError("recover_candidate: pass candidate_model, or baseline=True for the identity wiring check")`. Update the docstring/description: identity baseline is explicitly opt-in.
- [ ] **Step 5: Migrate references** — `composites/__init__.py:19` comment → `recover_candidate`; `recover-mass-action/study.yaml` `composite:`+`base_model:` → `…recover_candidate` (add whatever params the run path needs so its baseline still resolves — `baseline: true` if the identity baseline is intended, per Step 1's finding); drop the `recover_mass_action` import/call in the test file. Leave the historical run-name string as a past record (or regenerate the run — grades must stay equivalent; honesty + reproducibility tests must stay green).
- [ ] **Step 6: Verify** — `pytest tests/test_recovery_capabilities.py tests/test_verdict_honesty.py -v` → PASS; `grep -rn "recover_mass_action" viva_sbml_decomposition scripts tests` (`.py`) shows no live references; the recover-mass-action study still resolves its baseline composite.
- [ ] **Step 7: Commit** — `git add viva_sbml_decomposition/composites tests/test_recovery_capabilities.py workspace/investigations/discovery-sbml-recovery/studies/recover-mass-action/study.yaml && git commit -m "refactor(recovery): one recover_candidate composite; identity baseline now explicit (baseline=True)"`

---

### Task 3: Drop the vestigial `accuracy` placeholder

**Files:**
- Modify: `viva_sbml_decomposition/harness/composability.py` (remove `accuracy` from `ComposabilityScore` and all six row literals + the docstring sentence)
- Modify/create test: `tests/test_composability.py`

**Interfaces:**
- Produces: `ComposabilityScore` without the always-`None` `accuracy` field. Real accuracy is unaffected — `decision_table`/`multiorgan_scaling` read it from `load_metrics()`, not from this dataclass.

- [ ] **Step 1: Write the failing test** — in `tests/test_composability.py`:

```python
def test_composability_score_has_no_accuracy_placeholder():
    from dataclasses import fields
    from viva_sbml_decomposition.harness.composability import ComposabilityScore
    assert "accuracy" not in {f.name for f in fields(ComposabilityScore)}
    assert "accuracy_exempt" in {f.name for f in fields(ComposabilityScore)}  # the real flag stays
```

- [ ] **Step 2: Run it, verify it fails** — `pytest tests/test_composability.py::test_composability_score_has_no_accuracy_placeholder -v` → FAIL (`accuracy` field present).
- [ ] **Step 3: Edit `composability.py`** — remove `accuracy: Optional[float]` from the dataclass and the `accuracy=None,` line from each of the six `ComposabilityScore(...)` rows; drop the placeholder sentence from the docstring; remove the now-unused `Optional` import if nothing else uses it.
- [ ] **Step 4: Verify no consumer broke** — `grep -rn "\.accuracy\b" viva_sbml_decomposition --include=*.py | grep -iv "accuracy_exempt"` — confirm the remaining `.accuracy` reads are on the `load_metrics` dict (`decision_table.py`, `multiorgan_scaling.py`), not on a `ComposabilityScore`. Then `pytest tests/test_composability.py tests/test_visualizations.py -v` → PASS (figures still render).
- [ ] **Step 5: Commit** — `git add viva_sbml_decomposition/harness/composability.py tests/test_composability.py && git commit -m "refactor(composability): drop vestigial always-None accuracy placeholder (real accuracy lives in the accuracy harness)"`

---

### Task 4: Document + enforce the sim-artifact convention

**Files:**
- Create: `docs/CONVENTIONS.md`
- Create: `tests/test_artifact_convention.py`

**Interfaces:**
- Produces: one written convention plus a committed test enforcing it. **Decision (flagged for review):** the design's literal "seeded `runs.db` for both investigations" is reconciled with Phase 0's finding (recovery is code-graded, has no emitter) into **two honest artifact classes**, not an artificial emitter on recovery: (a) emitter-backed studies (decomposition) commit a seeded `runs.db`; (b) code-graded studies (recovery) commit seeded `computed_outcomes` + a rendered `report.html`. Both are reproducible and committed — the design's intent.

- [ ] **Step 1: Write the failing test** `tests/test_artifact_convention.py`:

```python
import pathlib, yaml
INV = pathlib.Path("workspace/investigations")

def test_recovery_studies_commit_computed_outcomes():
    studies = INV / "discovery-sbml-recovery" / "studies"
    for p in sorted(studies.glob("*/study.yaml")):
        d = yaml.safe_load(p.read_text()) or {}
        runs = d.get("runs") or []
        if not runs:
            continue  # scaffolded future-direction study, no run yet
        canon = next((r for r in runs if r.get("canonical")), runs[0])
        assert canon.get("computed_outcomes"), f"{p.parent.name}: canonical run missing computed_outcomes"

def test_decomposition_studies_commit_runs_db():
    studies = INV / "sbml-decomposition" / "studies"
    missing = [d.name for d in sorted(studies.iterdir())
               if d.is_dir() and not (d / "runs.db").exists()]
    assert not missing, f"emitter-backed studies missing runs.db: {missing}"
```

- [ ] **Step 2: Run it, verify its state** — `pytest tests/test_artifact_convention.py -v`. Both should PASS against the committed Phase-0 state (recovery studies carry `computed_outcomes`; decomposition studies carry `runs.db`). If either fails, that is a real convention gap to record, not a test to loosen — investigate before continuing. (This task's "failing test first" is the convention doc in Step 3; the test pins the already-true invariant so it cannot silently regress.)
- [ ] **Step 3: Write `docs/CONVENTIONS.md`** — a short page: the two artifact classes, when each applies (emitter-backed vs code-graded), that both are seeded + committed + reproducible, and a pointer to the enforcing test. Cross-reference Phase 0's `report.html` rendering.
- [ ] **Step 4: Verify** — `pytest tests/test_artifact_convention.py -v` → PASS; `docs/CONVENTIONS.md` renders as valid Markdown.
- [ ] **Step 5: Commit** — `git add docs/CONVENTIONS.md tests/test_artifact_convention.py && git commit -m "docs(conventions): document + enforce the two-class sim-artifact convention"`

---

### Task 5: Full-suite green + PR

**Files:** none (verification + PR)

- [ ] **Step 1: Run every CI gate locally** (venv + `PYTHONPATH=.`, CLI tools also with venv on `PATH`):
  - `pytest tests/ -v`
  - `python scripts/lint-workspace.py`
  - `uv lock --check`
  - `vivarium-workbench gen-readme --workspace . --check`
  - `python scripts/check-workflow-freshness.py --check`
  - `vivarium-workbench-publish --workspace . --out "$TMPDIR/render-check"`
- [ ] **Step 2: Verify branch/HEAD** — `git branch --show-current` = `cleanup/phase1-code-consistency`; `git log --oneline origin/main..HEAD` shows only this phase's commits.
- [ ] **Step 3: Push + open PR** — `git push -u origin cleanup/phase1-code-consistency`; `gh pr create --base main` titled `refactor: Phase 1 — code-consistency pass (loader, recovery composite, placeholder, artifact convention)`, body summarizing the four consolidations, the flagged two-class convention decision, and that no science/verdict/figure changed. No AI attribution.
- [ ] **Step 4: Watch CI; do NOT merge** — after green, hand to the owner for review/merge.

---

## Decisions flagged for your review

1. **Sim-artifact convention (Task 4):** two honest classes (runs.db for emitter-backed decomposition; committed `computed_outcomes`+report for code-graded recovery) rather than bolting an artificial emitter onto recovery. Consistent with merged Phase 0 (R2). *Alternative if you want literal uniformity:* add a lightweight emitter to the recovery composite so it also writes a `runs.db` — more work, and the grades still come from the evaluators, not the emitted series.
2. **Accuracy placeholder (Task 3):** dropped (design said "wire or drop"; it is always `None` and real accuracy already lives in the accuracy harness). *Alternative:* wire real per-arm accuracy into the composability rows — but that duplicates `load_metrics`.
3. **Identity baseline (Task 2):** made explicit via `baseline=True` rather than removed entirely — it still has a legitimate use (proving the recovery loop's wiring) but can no longer masquerade as a real recovery.
4. **Loader scope (Task 1):** the two `_multiorgan.py` timing-baseline sites keep raw `te.loada` (they measure its build time); everything else routes through `_tellurium.load`.
