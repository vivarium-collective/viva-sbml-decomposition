# Design: viva-sbml-decomposition — clarity cleanup + capability expansion

**Date:** 2026-11-13
**Status:** Awaiting approval
**Audience for the repo after this work:** a **scientific reviewer** — someone judging whether the methods are sound. Rigor, honest verdicts, and reproducibility lead every decision below.

---

## 1. Intent and why

`viva-sbml-decomposition` is a mature process-bigraph research workspace (~4k clean LOC) with real science: operator-splitting **decomposition** of SBML models across coupled engine processes, and a SINDy-based **recovery** pipeline that rebuilds SBML structure from time-series. The code is good. The problem is **legibility and rigor presentation**, plus three capability gaps the owner wants closed.

Two forces motivate the work:

1. **A reviewer cannot currently navigate the repo.** Two investigations face opposite directions (forward: run known models decomposed; backward: recover models from data) but have near-identical names, and one of them is effectively invisible (no report, absent from README/dashboard, zero committed sims). One investigation also presents a `complete` verdict over a *computed* `failing` — exactly the kind of overclaim a reviewer must not be allowed to miss.
2. **The owner wants to quantify and extend the tellurium+vivarium capabilities** — global sensitivity/UQ, cross-engine validation, scaling benchmarks, Bayesian calibration — and to honestly situate the Rust engines (viva-core, viva-hybrid) on the performance frontier.

**Why now:** the repo was just moved onto workbench 0.5.3 and is being readied for outside eyes. This is the moment to make it reviewer-legible before adding more.

## 2. Goals

- **G1 — Legible investigation structure.** A reviewer can tell, in 10 seconds, that there are two complementary method investigations (forward decomposition, backward recovery), which is which, and the honest status of each. Both are fully rendered (report.html) and appear in the README + dashboard.
- **G2 — Honest verdicts by construction (discovery-engine driven).** Verdicts and acceptance-criterion ↔ study linking are **derived** by `discovery_engine`, not hand-authored. The repo stops setting `verdict_status` to a favorable value by hand; the engine's pure roll-up (`study_verdict.roll_up_verdict`) computes the verdict from graded outcomes, authored-vs-derived divergence is surfaced, and the recovery investigation's `complete`-over-`failing` overclaim disappears because the standing verdict becomes the derived one. See §6a.
- **G3 — Reproducible artifacts.** Both investigations commit the same class of artifacts (seeded runs → `runs.db`, rendered `report.html`) under one convention. A reviewer can re-run and regenerate.
- **G4 — Code consistency.** Duplicate composites consolidated, the identity-baseline footgun guarded, the repeated tellurium loader DRY'd, placeholder columns either wired or dropped.
- **G5 — Quantified capabilities.** Four extended capabilities, each landed **complete** as a new study in the appropriate existing investigation, with a reported figure and an honest caveat: cross-engine validation, scaling benchmarks, global sensitivity/UQ (into the decomposition investigation), Bayesian calibration (into the recovery investigation).
- **G6 — Honest Rust-engine frontier.** A measured viva-core spike (best-effort, see §6 Phase 6), plus a report section citing viva-hybrid's negative result and the dispatch-vs-ODE distinction.

## 3. Non-goals

- **Not** rewriting the science. The decomposition and recovery methods stay; we clarify and extend them.
- **Not** integrating viva-core/viva-hybrid as a production ODE accelerator (neither accelerates ODE integration; see §6 Phase 6).
- **Not** adding new SBML model families beyond what the capabilities need.
- **Not** touching the workbench or discovery-engine packages themselves — we *consume* `discovery_engine`, we don't modify it. If a genuine engine gap is found, it's a separate upstream PR, not part of this work.

## 4. Anti-goals (these would make it a failure even if it "works")

- **A1 — Breadth over rigor.** Four half-finished capabilities for a reviewer audience is worse than two complete ones. Each capability lands complete or not at all.
- **A2 — Hiding the honest negatives.** If the recovery frontier or viva-hybrid's negative result gets softened into a false "success," the whole point (reviewer trust) is lost. Derived verdicts (G2) make this structurally hard.
- **A3 — A bigger, more tangled repo.** If cleanup adds net complexity, it failed.

## 5. Constraints

- **Worktree discipline.** All work in a dedicated worktree off `origin/main`; canonical checkout is read-only. One phase per PR; squash-merge; no auto-merge; no AI attribution.
- **CI must stay green.** `gen-readme --check`, workspace lint, `uv lock --check`, and the publish render-check all gate PRs. Any README/dashboard change must regenerate the tables.
- **Reopening is expected.** Per the owner's topology choice, adding capability studies **reopens** the two currently complete/closed investigations; their acceptance criteria and verdicts are **recomputed by the discovery-engine** on each change (that's the point of G2 — the verdict is always the honest derived one).
- **Tellurium/COPASI sharp edges.** COPASI has a documented ~1-in-6 nondeterministic C++ abort; keep the subprocess-retry mitigation and flag it, don't paper over it.
- **Reproducibility.** New and re-run artifacts must be seeded and deterministic where possible, and commit their `runs.db`.

## 6a. Cross-cutting foundation: discovery-engine drives verdicts + linking

This underlies every phase and is the mechanism for G2. `discovery_engine` (v0.3.1, already in the venv) is a pure, LLM-free, deterministic investigation/study engine.

- **Verdicts are derived, not authored.** `study_verdict.roll_up_verdict(spec)` is a pure function of the per-test graded outcomes; a study with findings but no graded run stays `unassessed` and *cannot close* an acceptance criterion. The repo's investigation/study YAML stops hand-setting a favorable `verdict_status`; where an authored value remains it is compared to the derived roll-up and the `diverges` flag is surfaced in the report. For the recovery investigation this *is* the fix: drop the authored `complete`, let the derived verdict (the honest "2/9 rungs recovered, frontier mapped") stand and render first.
- **Linking is projected, not hand-wired.** The AC ↔ study gating and relations come from the engine's projection/relations (the same linkage-index `/viva-navigate` queries), so "which studies gate which acceptance criteria" is computed, not maintained by hand.
- **The cycle drives it.** `discover.run_cycle` is derive-first: it projects from member studies and authors no prose. Rendered reports run the cycle at render time, so the Discovery/Decide view is always current. Gaps surfaced by the cycle can be filled via the engine's generation contract if desired, but that's optional.
- **Backstop.** `report_linter` / `rigor` remain available as a CI honesty gate if we want belt-and-suspenders, but the derive-first path is primary.

*Builder note:* exact function/module names may have shifted in 0.3.1 (memory warns the engine is actively edited — `build_verdict` has moved); the capability is confirmed present (`roll_up_verdict`, `run_cycle`, `relations`, `projection`, `report_linter`, `rigor`). The builder resolves the precise call sites against the installed package, and prefers the `/viva-*` skills (`/viva-study`, `/viva-investigation`, `/viva-report`, `/viva-navigate`) which already wrap these.

## 6. The plan — phased, each phase coherent on its own

> You can stop after any phase and the repo is still in a better, consistent state. Phases are ordered so rigor/legibility come before breadth. **Topology (owner choice):** capabilities are added as **new studies inside the two existing investigations**, which reopens them; the discovery-engine recomputes their verdicts honestly each time (§6a).

### Phase 0 — Clarity + honesty foundation (non-negotiable)
Makes the repo reviewer-legible and installs the derive-first verdict/linking foundation.
- **Adopt discovery-engine-driven verdicts + linking (§6a)** across both investigations: remove hand-authored favorable `verdict_status`, render the derived verdict + `diverges` flag honestly.
- **Rename for direction.** Make the forward/backward split unmistakable in titles + overviews (slugs optional). *Left to builder + owner:* final names (candidates: titles like "Decomposition tradeoffs" / "Model recovery from data"). Keep both investigations.
- **Make recovery visible + reproducible (owner choice: re-run).** Re-run the recovery pipeline **with fixed seeds**, commit `runs.db` + render `report.html`, add it to the README Investigations table + dashboard — matching the `sbml-decomposition` convention. This is the heavier, rigorous path (regenerates SINDy fits deterministically) chosen deliberately for the reviewer audience.
- **Regenerate** README tables + dashboard; verify `gen-readme --check` passes.
- *May split into two PRs:* (0a) discovery-engine verdict/linking adoption + rename + README/dashboard; (0b) recovery re-run + artifacts + report.

### Phase 1 — Code-consistency pass
- Consolidate `recover_mass_action` (identity-baseline) into the generic `recover_candidate`; guard/remove the identity-baseline default that once masked a false 5/5 PASS.
- Add a single `_tellurium.load()` loader; replace the duplicated `te.loada(...)` sites (~9 files).
- Wire or drop the `harness/composability.py` `accuracy=None` placeholder column.
- Apply one sim-artifact convention (seeded `runs.db` committed) to both investigations.

### Phase 2 — Cross-engine validation study *(capability 1 → decomposition investigation)*
First of the four: it directly serves reviewer trust and reuses existing infra.
- Tellurium vs COPASI vs native agreement harness; **decomposition splitting-error** promoted to a first-class, reported metric.
- New study in the decomposition investigation with a derived verdict; figure: agreement/divergence map across engines + seams. Honest caveat: COPASI nondeterminism.

### Phase 3 — Scaling / performance benchmarks study *(capability 2 → decomposition investigation)*
- Systematic quantification of decomposition speedups and ensemble scaling (extends the measured 7.5× ensemble win + heap-scheduler multirate path).
- Figures: scaling curves (model size, #instances, #engines). Sets the baseline the viva-core spike (Phase 6) is measured against.

### Phase 4 — Global sensitivity / UQ study *(capability 3 → decomposition investigation)*
- Sobol/Morris indices + Latin-hypercube sweeps layered on the existing Ray ensemble.
- *Left to builder:* `SALib` vs hand-rolled Sobol. Figures: parameter-importance ranking with uncertainty bands.

### Phase 5 — Bayesian calibration study *(capability 4 → recovery investigation)*
- Fit parameters to data with posterior distributions; a natural extension of the recovery pipeline from point estimates to uncertainty. Lands as a new study in the **recovery** investigation (which reopens; verdict re-derived).
- *Left to builder:* method/library (ABC-SMC vs `emcee`/`pymc`); prefer the lightest dependency giving honest posteriors. Figures: posterior + predictive check.

### Phase 6 — Rust-engine frontier: viva-core spike + report section *(G6)*
- **Measured spike (best-effort, my call on scope):** `viva_core.run()` only executes processes in the Rust engine's *native* registry, so tellurium/COPASI composites can't run there. The spike targets the one portable arm (`native_composite`, hand-written RHS). **If even that needs a Rust-side process port, Phase 6 becomes report-only** — and "the engine can't run our composites without a port" is itself the honest frontier finding. The number ships as a committed figure with caveat; **not** gated in CI (no Rust toolchain in the pipeline).
- **Report section** in the sbml-decomposition report: viva-core accelerates bigraph *dispatch*, not ODE integration (measured ~1.27× scalar, regresses on array-heavy state today); viva-hybrid cited as a rigorous **negative result** (~2× over numba; no regime both ≥5× and faithful to exact SSA). Frames where a Rust path would and wouldn't pay off.
- *Explicitly not* building a Rust ODE kernel or native-process registration.

## 7. What's left to the builder

- Exact investigation renames (final titles, with owner sign-off).
- Whether Phase 0 is one PR or split 0a/0b.
- Precise `discovery_engine` call sites / which `/viva-*` skills to drive verdicts + linking (resolve against installed 0.3.1).
- Bayesian method + library (Phase 5); sensitivity library (Phase 4).
- Per-phase acceptance criteria + figures (these become the `writing-plans` task breakdown).

## 8. Sequencing + delivery

- One PR per phase (Phase 0 may be 0a/0b). Each PR: green CI, regenerated README/dashboard where touched, a rendered figure or report delta, a derived (not authored) verdict, and an honest caveat section.
- Capabilities land as **new studies in the existing investigations**, which reopen; the discovery-engine re-derives each investigation's verdict on every change, so reopening never produces an overclaim.
- After this design is approved → `superpowers:writing-plans` turns each phase into a task-level plan with acceptance criteria.
- Phase 0 and 1 are prerequisites. Phases 2–5 are independent enough to parallelize across subagents once the foundation is in.

## 9. Scope note (flagged honestly)

Answer C ("drive linking and verdicts through the discovery-engine") turned the verdict fix from a local Phase-0 edit into a **cross-cutting foundation** (§6a) touching every investigation and every new study. This is the right call for a reviewer audience — honesty by construction — but it is more work than a one-off verdict edit, and it couples the repo more tightly to `discovery_engine`'s API. Noted so the cost is visible before approval.
