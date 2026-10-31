# Capability ledger — discovery-engine SBML recovery

Living record of the capabilities the **discovery-sbml-recovery** investigation
needs to close the gap between "run the discovery cycle" and "actually build good
flat SBML models from data" — and the modular, vivarium-style process that
implements each one.

**How this is used.** The discovery-engine cycle drives a closed
`build -> simulate -> compare -> refine` loop per study. When the cycle's frontier
surfaces something no existing process can do, that's a **capability gap**: record
it here, build it as a process-bigraph process (`/viva-expert --lightweight` ->
`viva_sbml_decomposition/<...>` process + test), wire it into the loop, and flip
its status. Capabilities are built **frontier-driven** — only when a study
actually needs one — so this grows from real gaps, not speculation.

**Status values:** `needed` (identified, not started) · `building` (process in
progress) · `built` (process exists + tested + wired into the loop).

## Ledger

| Capability | Role in the build->simulate->compare->refine loop | Status | Process (module) | Surfaced by |
|---|---|---|---|---|
| synthetic-target-generator | Author + simulate a ground-truth SBML -> emit the synthetic target trajectories (optional noise). Sets up each study's oracle. | built | `viva_sbml_decomposition.processes.recovery.SyntheticTargetGenerator` | recover-mass-action |
| trajectory-comparator | Score a candidate model's simulation against the target (normalized / phase-aligned RMSE, feature residuals). The "compare" step; also yields the residual the frontier reads. | built | `viva_sbml_decomposition.processes.recovery.TrajectoryComparator` | recover-mass-action |
| rate-law-fitter | Fit a candidate SBML's parameters to the target (mass-action constants, Km/Vmax, Hill n/K). The "build/refine" step. | built | `viva_sbml_decomposition.processes.recovery.RateLawFitter` | recover-enzyme-kinetics |
| reaction-proposer | Propose candidate reactions / rate laws from the residual (what the current model fails to explain). The frontier-driven core of "refine". | built | `ReactionProposer` (enumerate + SINDy `from_data`: sparse structure discovery from data) | recover-gene-switch |
| structural-scorer | Compare a recovered model's reactions + parameters to the ground truth (reactions present, params within tol). Grades structural recovery. | built | `viva_sbml_decomposition.processes.recovery.StructuralScorer` | recover-mass-action |
| parsimony-evaluator | Penalize excess reactions / parameters; arbitrate between observationally-equivalent mechanisms (e.g. reduced MM vs elementary steps). Grades parsimony. | built | `StructuralScorer.parsimonious` (folded into StructuralScorer) | recover-enzyme-kinetics |
| non-recoverability-detector | Decide when no faithful flat-SBML form can fit (honest failure) rather than overfitting forever. Stalls the loop with a verdict. | built | `NonRecoverabilityDetector` (polynomial-RHS R^2; flags stochastic adversarial) | recover-oscillator (adversarial, planned) |

## Design notes per capability

Each becomes a process-bigraph process (a Step for the pure scoring/proposal ones,
a Process only where it must advance state). Build order follows the ladder:

- **recover-mass-action** needs `synthetic-target-generator`, `trajectory-comparator`,
  `structural-scorer` — the minimum loop. Build these three first.
- **recover-enzyme-kinetics** adds `rate-law-fitter` (non-mass-action saturating law)
  and `parsimony-evaluator` (reduced vs elementary).
- **recover-gene-switch** adds / stresses `reaction-proposer` (propose the Hill
  nonlinearity from the residual, not a linear surrogate).
- **recover-oscillator** stresses `reaction-proposer` + `rate-law-fitter` on a
  limit-cycle topology, and is the natural trigger for `non-recoverability-detector`.

When a capability is built, replace `_tbd_` with its module path (e.g.
`viva_sbml_decomposition.processes.trajectory_comparator.TrajectoryComparator`),
flip status to `built`, and link the study/run that first exercised it.
