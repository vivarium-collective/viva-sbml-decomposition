# viva-sbml-decomposition — design

**Date:** 2026-10-25
**Status:** design, pending approval
**Author:** Eran (with Claude)

## Intent & why

Vivarium / process-bigraph lets you take a biochemical model and decompose it
into coupled sub-processes orchestrated by a composer. That buys modularity,
substitutability, and multi-scale coupling — but it is not free: orchestration
adds overhead, and operator-splitting a stiff ODE system introduces numerical
error. The opposite pole — the whole model as a single SBML file integrated by
libroadrunner/CVODE in Tellurium — is fast and numerically exact for that model,
but monolithic and hard to compose with anything else.

Nobody has a clear, quantified picture of **where on that spectrum you should sit
for a given model, and why.** This workspace produces that picture: a methods
investigation that runs several SBML models along the full spectrum — from
monolithic Tellurium to fully native process-bigraph — measures performance and
accuracy at each point, and distills a decision guide: *when vivarium's
decomposition earns its keep, when plain Tellurium wins, and which decomposition
patterns to avoid.*

This is a **methods result, not a biology result.** The models are vehicles
chosen for their numerical character, not for a biological question.

## Goals

- A single investigation whose **deliverable is a 3-way tradeoff map**:
  **speed × accuracy × composability**. A 2-way accuracy-vs-speed map is
  insufficient on its own — the monolithic Tellurium run dominates every
  decomposed arm by construction (decomposition can only *add* overhead and
  splitting error), so that plane alone can only ever recommend "use monolithic"
  and buries the actual reason vivarium exists. The deliverable therefore adds a
  **composability dimension** the monolith cannot play on (below) + a decision
  table (use / avoid / why) covering the decomposition patterns.
- Cover a **spectrum of implementations** for each model (the "arms", below),
  with the monolithic Tellurium run as the accuracy reference.
- Cover **three models of different numerical character** so the guidance
  generalizes: a stiff oscillator, a modular signaling cascade, a metabolic
  network.
- Include an **ensemble arm** — many SBML instances orchestrated by vivarium on a
  **real parallel backend** — to show where vivarium's value is clearest
  (embarrassingly-parallel / parameter-ensemble / replicate workloads); this is
  the one arm where vivarium can win on the *speed* axis, not only lose.
- Include a **coupling-demonstration arm** that does something monolithic SBML
  *structurally cannot* — so the payoff shows capability bought, not just speed
  lost.
- Every arm **shows its model source (Antimony) cleanly in the study Model tab**,
  reusing the mosaic-compose inline-Antimony pattern.
- The investigation is **driven by the RFC-0005 discovery engine**: real
  acceptance criteria + per-study verdicts, so the frontier picks the next study
  and closure signals completion.

## Non-goals

- Not calibrating any model to data or making a biological claim.
- Not benchmarking Tellurium/libroadrunner against other SBML simulators.
- Not a general-purpose SBML→process-bigraph auto-partitioner (the splits are
  authored by hand per model, deliberately).

## Anti-goals (would make this a failure even if it "works")

- A tradeoff map with no clear recommendation — numbers but no "use X, avoid Y".
- Accuracy numbers that aren't anchored to a trustworthy reference (the
  monolithic run must be validated as the ground truth, same tolerances across
  arms).
- Arms that differ in more than the decomposition (e.g. different integrator
  tolerances, different output cadence) — confounds the comparison. Every arm
  must integrate the *same* model with matched settings; only the decomposition
  changes.
- An "ensemble" that is secretly just one run — it must actually orchestrate N
  independent instances.

## The spectrum (arms), per model

1. **Monolithic Tellurium** — whole model as one `TelluriumProcess`
   (`make_tellurium_document`, inline Antimony). libroadrunner/CVODE integrates
   the full system. **This is the accuracy reference.**
2. **Operator-split Tellurium** — the *same* model, integration broken into N
   coupled sub-steps that exchange state through shared stores every vivarium
   step. Swept over coupling interval. Isolates: vivarium orchestration overhead
   + operator-splitting error, with no change to the biology.
3. **Modular submodels as processes** — the model partitioned along natural seams
   (reaction subsets / compartments / modules) into several `TelluriumProcess`es
   coupled through shared species stores. The "break it along biological seams"
   case.
4. **Fully native process-bigraph** — the ODE network reimplemented as native pbg
   `Process`/`Step`es (no Tellurium / libroadrunner). The far pole: maximum
   composability, hand-written integration.
5. **Ensemble** — many instances of the monolithic SBML model orchestrated by
   vivarium (parameter ensemble and/or stochastic replicates) on a **real
   parallel backend** (Ray / multiprocessing), so the N instances run
   concurrently and a throughput speedup is actually measurable. Serial execution
   would prove nothing on performance and is explicitly out. Tests vivarium as an
   orchestrator of embarrassingly-parallel SBML workloads.
6. **Coupling demonstration** — take a decomposed form (arm 3's submodels) and do
   something the monolith structurally cannot: swap one submodel for a native-PBG
   or different-simulator version, or couple a submodel to an external input /
   different time-scale. This arm is *not* scored on accuracy-vs-reference (there
   is no monolithic equivalent — that's the point); it is the existence proof of
   the composability payoff.

Not every model needs all six arms; the metabolic and oscillator models are the
richest (arms 1–5), the signaling cascade anchors arm 3 (natural modularity) and
hosts the arm-6 coupling demonstration.

## Models (vehicles, chosen for numerical character)

- **Oscillator** — repressilator (Elowitz–Leibler) or Brusselator. Stiff,
  periodic → operator-splitting corrupts period/amplitude. Stresses *accuracy*.
- **Modular signaling cascade** — MAPK / Huang–Ferrell. Clean module seams →
  the best case for modular decomposition (arm 3).
- **Metabolic network** — Teusink glycolysis. Larger reaction set → stresses
  *performance* and scaling.

Sourced from BioModels where a curated entry exists; authored as inline Antimony
so the Model tab shows readable source. Each model's identity + provenance
recorded in the study.

## Measurement

- **Performance:** the headline metric is the **ratio vs the monolithic arm**
  (orchestration overhead factor), not raw wall-clock — so background load on a
  shared machine can't corrupt the comparison. All arms for a given model are
  timed **together in one session on one quiesced host** (the Mac mini is the
  clean option), repeated, warm vs cold separated. Absolute wall-clock per unit
  sim-time + one-time setup/compile cost are reported as secondary with that
  caveat. Scaling measured vs step count, coupling interval (arm 2), and
  #instances (arm 5, with the parallel backend).
- **Accuracy:** per-species RMSE and max relative error against the monolithic
  reference trajectory on a common time grid; plus period/amplitude fidelity for
  the oscillator and conservation/steady-state checks where applicable. (Arm 6 is
  exempt — no monolithic reference exists for it.)
- **Composability:** a scored dimension per arm — substitutability (can a part be
  swapped?), coupling-readiness (can it wire to an external process / store?),
  multi-scale fit — anchored by the arm-6 coupling demonstration as the concrete
  existence proof. This is what keeps the recommendation from collapsing to
  "monolithic always wins."
- **Tradeoff map:** per-model **3-way** figure (speed × accuracy × composability;
  e.g. accuracy-vs-speed scatter with composability encoded as size/color, plus
  the arm-6 capability callout) + a cross-model decision table whose columns
  include a composability score. Interactive Plotly visualizations in the
  workspace package, rendered against real xarray/zarr runs (mosaic-compose
  foundations pattern).

## How it gets made — decided

- **Workspace:** new **standalone** viva workspace (clone viva-template), repo
  `viva-sbml-decomposition`, package `viva_sbml_decomposition` (viva_ prefix per
  convention). Dedicated worktree per session for any commits.
- **Simulator bridge:** `viva-tellurium` (already in the ecosystem) for arms 1–3
  and 5; inline Antimony via `make_tellurium_document(model=..., model_format=
  "antimony")`.
- **Native arm:** native pbg Processes authored in the workspace package.
- **Study runtime:** `runtime.default_emitter: xarray`, workspace
  `runtime.max_generations: 1`; strip library RAM emitters (mosaic-compose
  `_emit.strip_emitter` pattern).
- **Investigation structure:** one investigation; member studies = model × arm;
  acceptance criteria are the tradeoff questions; per-study canonical verdicts set
  so the discovery-engine roll-up is real.
- **Driving loop:** author a study → run it → turn the crank
  (`/viva-navigate decisions <inv>`) → frontier names the next study → repeat to
  closure.
- **Ensemble backend:** a real parallel executor (Ray preferred, matching the
  ecosystem's mini Ray-sweep pattern; multiprocessing fallback) so arm 5's
  throughput speedup is genuine and measurable.
- **Timing environment:** arms for a model timed together, one session, one
  quiesced host; the Mac mini is the designated clean host. Primary metric =
  ratio to monolithic.
- **Visualizations:** v2 decorated-function Visualizations in the package.
- **Publish:** read-only workbench to gh-pages at the end (optional, on request).

## Left to the builder / plan

- Exact BioModels IDs / Antimony for each of the three models (pick curated
  entries; validate they integrate cleanly in monolithic Tellurium first).
- Precise seam partitioning for arm 3 per model.
- The native-PBG integration scheme for arm 4 (which integrator, how reactions map
  to Processes/Steps).
- Ensemble arm 5 design: parameter sweep vs stochastic replicates vs both; how
  many instances; how the ensemble is emitted/aggregated (backend is decided:
  real parallel / Ray).
- Arm 6 coupling demonstration: which concrete coupling to show (submodel swap to
  native-PBG, submodel swap to a different simulator, or coupling to an external
  input / different time-scale) — pick the one that most clearly can't be done
  monolithically.
- The composability scoring rubric: the exact criteria and scale for the
  substitutability / coupling-readiness / multi-scale columns.
- The exact acceptance-criterion tolerances (T for "within tolerance", overhead
  thresholds) — provisional in the plan, calibrated against the monolithic
  reference once it runs.
- Timing harness details (repeats, warm/cold, what's included in "setup").
- Build order: start with the metabolic model full spectrum (richest, proves the
  harness end-to-end), then oscillator, then signaling cascade — driven by the
  frontier.
