# Phase 0 — Clarity + Honesty Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make viva-sbml-decomposition reviewer-legible: two clearly-named complementary investigations, both fully rendered + in the README/dashboard, with verdicts and AC↔study linking driven by `discovery_engine` so the recovery investigation's `complete`-over-`failing` overclaim cannot stand.

**Architecture:** Derive-first honesty. We stop hand-authoring a favorable `executive.verdict_status`; `discovery_engine.investigation_status.write_investigation_acceptance` computes `computed_verdict_status` + `computed_acceptance` from each member study's graded outcomes, and a committed pytest enforces that no authored verdict is more favorable than its computed verdict. Recovery studies are re-graded deterministically and rendered.

**Tech Stack:** Python 3.12, `discovery_engine` 0.3.1, `vivarium-workbench` 0.5.3, pytest, ruamel YAML (via the engine), tellurium/libroadrunner (existing).

**Spec:** `docs/superpowers/specs/2026-11-13-sbml-decomposition-clarity-and-capabilities-design.md` (this plan implements Phase 0 of §6, and the §6a cross-cutting foundation).

## Global Constraints

- Worktree only (`~/code/viva-sbml-decomposition--cleanup`, branch `cleanup/sbml-clarity-and-capabilities`); canonical checkout read-only. Verify `git branch --show-current` + HEAD before each commit.
- No AI attribution in commits/PRs. Squash-merge; no auto-merge; one PR per phase (Phase 0 may split 0a/0b).
- Run all tooling with the main checkout's venv: `/Users/eranagmon/code/viva-sbml-decomposition/.venv/bin/...` (the worktree has no `.venv`; set `PYTHONPATH` to the worktree so worktree source wins, or run from a venv synced to the worktree).
- Do **not** modify `discovery_engine`, `vivarium-workbench`, or the science of the methods. We consume engine APIs; we do not change them.
- `computed_verdict_status` vocabulary is exactly `{passing, failing, passing-with-caveats, in-progress}` (`discovery_engine/investigation_status.py:26-29`). `complete`/`closed` are lifecycle `status:` values, never verdict values.
- Recovery grading is deterministic (seed `0`, `noise_sigma 0.0`; stochastic target pins `seed: 3` in `models/recovery_targets.py:97`). Keep it deterministic; do not introduce unseeded randomness.

## Review Focus

- **Favorable-overclaim slips back in** (any investigation re-authors a passing-ish `verdict_status` over a failing/in-progress computed verdict) → Task 2's `test_no_verdict_overclaim` must fail. Most likely to bite; it is the whole point of the phase.
- **Benign conservative divergence misread as a violation** (decomposition authors `passing-with-caveats` over computed `passing`) → the honesty test must PASS this case; only *more-favorable* authored verdicts fail. Covered in Task 2.
- **A study with findings but no graded run silently closes a criterion** → the engine already blocks this (`unassessed` cannot close an AC); Task 3's reproducibility test asserts every recovery criterion resolves from a graded outcome, not a bare finding.
- **Recovery report renders but is empty/404** (missing figures, < 20 000 bytes) → Task 4 asserts `report.html` ≥ 20 000 bytes and the `vivarium-workbench-publish` render-check exits 0.
- **Stale README/dashboard after a title change** (investigations table omits recovery, or shows old titles) → Task 1 and Task 4 run `gen-readme --check` and assert both titles appear.

---

## File Structure

- `workspace/investigations/sbml-decomposition/investigation.yaml` — title + overview clarity (Task 1); keep benign authored verdict.
- `workspace/investigations/discovery-sbml-recovery/investigation.yaml` — title + overview (Task 1); **remove** authored `executive.verdict_status` (Task 2); refreshed `computed_*` (Task 2/3).
- `workspace/investigations/discovery-sbml-recovery/studies/*/study.yaml` — refreshed deterministic `computed_outcomes` (Task 3).
- `workspace/investigations/discovery-sbml-recovery/report.html` — **new**, rendered + committed (Task 4).
- `README.md` — regenerated generated blocks (Task 1, Task 4).
- `tests/test_verdict_honesty.py` — **new**, the committed honesty invariant (Task 2).
- `tests/test_recovery_reproducible.py` — **new**, deterministic re-grade check (Task 3).

---

### Task 1: Clarify investigation titles + direction (title-only rename)

**Files:**
- Modify: `workspace/investigations/sbml-decomposition/investigation.yaml` (`title:` line 3; `executive.what_is_this`/overview)
- Modify: `workspace/investigations/discovery-sbml-recovery/investigation.yaml` (`title:` line 3; overview)
- Modify: `README.md` (regenerated investigations table)

**Interfaces:**
- Produces: two investigations whose titles state direction. Proposed (owner may override): decomposition → `"Decomposition Tradeoffs — When to Split an SBML Model"`; recovery → `"Model Recovery from Data — The Recoverability Frontier"`.
- Title-only change: slug/dir, `name:`, package, repo URLs, and cross-refs are untouched (blast radius confirmed: a title edit touches only line 3 + regenerated README).

- [ ] **Step 1: Write the failing test** — add to `tests/test_verdict_honesty.py` (shared file created in Task 2; if authoring Task 1 first, put this as a standalone assert):

```python
def test_investigation_titles_state_direction():
    import yaml, pathlib
    inv = pathlib.Path("workspace/investigations")
    dec = yaml.safe_load((inv/"sbml-decomposition/investigation.yaml").read_text())
    rec = yaml.safe_load((inv/"discovery-sbml-recovery/investigation.yaml").read_text())
    assert "Decomposition" in dec["title"]
    assert "Recovery" in rec["title"] and "Recover" not in dec["title"]
```

- [ ] **Step 2: Run it, verify it fails** — `.../.venv/bin/python -m pytest tests/test_verdict_honesty.py::test_investigation_titles_state_direction -v` → FAIL (titles not yet set).
- [ ] **Step 3: Set the two `title:` fields** (line 3 of each `investigation.yaml`) to the proposed strings; tighten each `executive.what_is_this`/overview first sentence to name the direction ("forward: run known models decomposed" / "backward: recover model structure from data").
- [ ] **Step 4: Regenerate README** — `.../.venv/bin/vivarium-workbench gen-readme --workspace .`
- [ ] **Step 5: Verify** — `pytest tests/test_verdict_honesty.py::test_investigation_titles_state_direction -v` → PASS; `vivarium-workbench gen-readme --workspace . --check` → exit 0; confirm both titles appear in `README.md` investigations table.
- [ ] **Step 6: Commit** — `git add workspace/investigations/*/investigation.yaml README.md tests/test_verdict_honesty.py && git commit -m "docs(investigations): title+overview state forward/backward direction"`

---

### Task 2: Drive verdicts through discovery_engine + commit the honesty invariant

**Files:**
- Modify: `workspace/investigations/discovery-sbml-recovery/investigation.yaml` (remove `executive.verdict_status`; refresh `computed_*`)
- Create: `tests/test_verdict_honesty.py`

**Interfaces:**
- Consumes: `discovery_engine.investigation_status.write_investigation_acceptance(inv_dir, workspace=None) -> bool` (writes only `executive.computed_verdict_status` + `computed_acceptance`; never touches authored fields); `roll_up_acceptance(inv_spec, studies_by_name) -> dict`.
- Produces: recovery investigation with no authored `verdict_status`, `computed_verdict_status: failing` standing, `computed_acceptance.diverges_from_authored: false`. The honesty test that gates all later phases.

- [ ] **Step 1: Write the failing test** `tests/test_verdict_honesty.py`:

```python
import pathlib, yaml, pytest

_FAVOR = {"failing":0, "in-progress":1, "passing-with-caveats":2,
          "passing":3, "passed":3, "pass":3, "accepted":3,
          "complete":3, "closed":3}  # complete/closed read as passing-level overclaim

def _invs():
    for p in pathlib.Path("workspace/investigations").glob("*/investigation.yaml"):
        yield p.parent.name, yaml.safe_load(p.read_text())

@pytest.mark.parametrize("name,spec", list(_invs()))
def test_no_verdict_overclaim(name, spec):
    ex = spec.get("executive", {}) or {}
    authored = (ex.get("verdict_status") or "").strip().lower()
    computed = (ex.get("computed_verdict_status") or "").strip().lower()
    if not authored:            # derive-first: no claim, no overclaim
        return
    assert computed, f"{name}: authored verdict but no computed_verdict_status"
    assert _FAVOR.get(authored, 3) <= _FAVOR[computed], (
        f"{name}: authored '{authored}' is more favorable than computed '{computed}'")
```

- [ ] **Step 2: Run it, verify it fails** — `pytest tests/test_verdict_honesty.py::test_no_verdict_overclaim -v` → FAIL on `discovery-sbml-recovery` (authored `complete` > computed `failing`); PASS on `sbml-decomposition` (authored `passing-with-caveats` ≤ computed `passing`, benign).
- [ ] **Step 3: Remove the overclaim** — delete the `verdict_status: complete` line under `executive:` in `discovery-sbml-recovery/investigation.yaml` (leave `status: closed` lifecycle and the prose `executive.verdict` frontier framing intact).
- [ ] **Step 4: Recompute via the engine** — `.../.venv/bin/python -c "from discovery_engine.investigation_status import write_investigation_acceptance; write_investigation_acceptance('workspace/investigations/discovery-sbml-recovery', workspace='.'); write_investigation_acceptance('workspace/investigations/sbml-decomposition', workspace='.')"` (equivalently `viva-discover evaluate` if its CLI covers investigation acceptance — builder confirms against `viva-discover --help`).
- [ ] **Step 5: Verify** — `pytest tests/test_verdict_honesty.py -v` → PASS for both; confirm `computed_acceptance.diverges_from_authored: false` and `computed_verdict_status: failing` in the recovery YAML.
- [ ] **Step 6: Commit** — `git add workspace/investigations/discovery-sbml-recovery/investigation.yaml tests/test_verdict_honesty.py && git commit -m "fix(verdict): derive recovery verdict via discovery_engine; drop complete-over-failing overclaim"`

---

### Task 3: Re-grade recovery studies deterministically + prove reproducibility

**Files:**
- Modify: `workspace/investigations/discovery-sbml-recovery/studies/*/study.yaml` (refreshed `computed_outcomes`)
- Create: `tests/test_recovery_reproducible.py`

**Interfaces:**
- Consumes: `scripts/recover.py` (`recover(study_slug)` → writes `recovery-fit:<slug>` run with inline `computed_outcomes`), `scripts/grade_recovery.py` (`grade(study_slug, inv_slug="discovery-sbml-recovery")` → `discovery_engine.study_evaluator.compute_outcomes` + `study_verdict.write_gate_evaluator`/`roll_up_verdict`). Both deterministic.
- Produces: recovery studies with freshly-regenerated, reproducible `computed_outcomes`; parity artifact (`viva-backfill-runs` row) where it applies.
- Note (artifact convention): recovery has no emitter, so its reproducible artifact is the seeded `computed_outcomes`, not a `runs.db`. For gallery/convention parity we add a backfilled run row if `viva-backfill-runs` supports a code-graded study; otherwise the committed `computed_outcomes` + report are the artifact (documented in the PR body). This honors the spec's intent (reproducible, committed) over its literal "runs.db".

- [ ] **Step 1: Write the failing test** `tests/test_recovery_reproducible.py`:

```python
import subprocess, sys, yaml, pathlib, copy
VENV = "/Users/eranagmon/code/viva-sbml-decomposition/.venv/bin/python"
STUDIES = ["recover-mass-action","recover-enzyme-kinetics","recover-gene-switch",
           "recover-oscillator","recover-stochastic-abm"]

def _outcomes(slug):
    p = pathlib.Path(f"workspace/investigations/discovery-sbml-recovery/studies/{slug}/study.yaml")
    runs = yaml.safe_load(p.read_text()).get("runs", [])
    return [r.get("computed_outcomes") for r in runs]

def test_recovery_grades_are_deterministic():
    for slug in STUDIES:
        before = copy.deepcopy(_outcomes(slug))
        subprocess.run([VENV,"scripts/recover.py",slug], check=True)
        subprocess.run([VENV,"scripts/grade_recovery.py",slug], check=True)
        assert _outcomes(slug) == before, f"{slug}: re-grade changed computed_outcomes (non-deterministic)"
```

- [ ] **Step 2: Run it, verify it fails appropriately** — first run may fail if current committed outcomes are stale; that failure is the signal to regenerate. `pytest tests/test_recovery_reproducible.py -v`.
- [ ] **Step 3: Regenerate** — for each study: `.../.venv/bin/python scripts/recover.py <slug> && .../.venv/bin/python scripts/grade_recovery.py <slug>`. Then re-run `write_investigation_acceptance` (Task 2 Step 4) so the investigation roll-up reflects refreshed outcomes.
- [ ] **Step 4: Parity artifact** — attempt `.../.venv/bin/viva-backfill-runs` for the recovery studies; if it populates a `runs.db` from `computed_outcomes`, commit it; if not applicable, record in the PR body that `computed_outcomes` is the committed reproducible artifact.
- [ ] **Step 5: Verify** — `pytest tests/test_recovery_reproducible.py -v` → PASS (re-grade is idempotent); `pytest tests/test_verdict_honesty.py -v` still PASS.
- [ ] **Step 6: Commit** — `git add workspace/investigations/discovery-sbml-recovery/studies tests/test_recovery_reproducible.py && git commit -m "chore(recovery): regenerate deterministic grades + reproducibility test"`

---

### Task 4: Render + commit recovery report.html; make both investigations visible

**Files:**
- Create: `workspace/investigations/discovery-sbml-recovery/report.html`
- Modify: `README.md` (investigations table includes recovery)

**Interfaces:**
- Consumes: `scripts/publish_investigation_reports.py --workspace . --only discovery-sbml-recovery --out <tmp>` (serves `vivarium-workbench serve`, GETs `/api/investigation-report/<slug>`, validates `MIN_REPORT_BYTES = 20000`); `vivarium-workbench-publish --workspace . --out <dir>` (render-check, exit≠0 = a study would 404); `vivarium-workbench gen-readme --workspace .`.
- Produces: a committed recovery `report.html` matching the close-workflow convention; a README/dashboard that lists both investigations.

- [ ] **Step 1: Write the failing test** — add to `tests/test_verdict_honesty.py`:

```python
def test_both_investigations_have_reports():
    import pathlib
    base = pathlib.Path("workspace/investigations")
    for slug in ("sbml-decomposition","discovery-sbml-recovery"):
        rpt = base/slug/"report.html"
        assert rpt.exists() and rpt.stat().st_size >= 20000, f"{slug}: missing/empty report.html"
```

- [ ] **Step 2: Run it, verify it fails** — `pytest tests/test_verdict_honesty.py::test_both_investigations_have_reports -v` → FAIL (recovery report absent).
- [ ] **Step 3: Render + place the report** — `.../.venv/bin/python scripts/publish_investigation_reports.py --workspace . --only discovery-sbml-recovery --out "$TMPDIR/rpt"`, then copy `"$TMPDIR/rpt/investigations/discovery-sbml-recovery.html"` → `workspace/investigations/discovery-sbml-recovery/report.html` (mirrors `/viva-investigation close`).
- [ ] **Step 4: Regenerate README/dashboard** — `.../.venv/bin/vivarium-workbench gen-readme --workspace .`
- [ ] **Step 5: Verify** — `pytest tests/test_verdict_honesty.py::test_both_investigations_have_reports -v` → PASS; `.../.venv/bin/vivarium-workbench-publish --workspace . --out "$TMPDIR/render-check"` → exit 0; `gen-readme --workspace . --check` → exit 0; both investigations appear in the README table.
- [ ] **Step 6: Commit** — `git add workspace/investigations/discovery-sbml-recovery/report.html README.md tests/test_verdict_honesty.py && git commit -m "docs(recovery): render+commit report.html; list both investigations"`

---

### Task 5: Full-suite green + PR

**Files:** none (verification + PR)

- [ ] **Step 1: Run every CI gate locally** (all with the venv):
  - `pytest tests/ -v`
  - `python scripts/lint-workspace.py`
  - `uv lock --check`
  - `vivarium-workbench gen-readme --workspace . --check`
  - `python scripts/check-workflow-freshness.py --check`
  - `vivarium-workbench-publish --workspace . --out "$TMPDIR/render-check"`
- [ ] **Step 2: Verify branch/HEAD** — `git branch --show-current` = `cleanup/sbml-clarity-and-capabilities`; `git log --oneline origin/main..HEAD` shows only this phase's commits.
- [ ] **Step 3: Push + open PR** — `git push -u origin cleanup/sbml-clarity-and-capabilities`; `gh pr create --base main` titled `fix: Phase 0 — investigation clarity + discovery-engine-driven honest verdicts`, body noting the recovery-artifact convention decision (computed_outcomes vs runs.db) and that no AI attribution is added.
- [ ] **Step 4: Do NOT merge** — hand back to the owner for review/merge.

---

## Notes carried to later phases

- The `test_no_verdict_overclaim` invariant (Task 2) is the gate every capability phase (2–5) inherits: each new study reopens its investigation, the engine re-derives the verdict, and this test fails if any reopened verdict overclaims.
- Owner sign-off still open on final investigation titles (Task 1) — proposed strings used; a rename is a one-line + `gen-readme` change.
