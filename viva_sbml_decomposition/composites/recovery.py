"""Recovery composite: run a CANDIDATE flat-SBML model for a recovery study.

The discovery-engine SBML-recovery loop compares a candidate model's simulation to
the hidden synthetic target. The *candidate run* is an ordinary monolithic
simulation, so this composite is a thin specialization of ``engine_monolithic``:
it runs ``candidate_model`` (raw Antimony) in tellurium and emits its trajectories.
The target generation, trajectory comparison, and structural scoring are applied
AROUND the run by the recovery capabilities
(:mod:`viva_sbml_decomposition.processes.recovery`), not as composite nodes.

``candidate_model`` defaults to the ground-truth mass-action model — the identity
baseline, which recovers the target exactly and establishes that the loop wiring
(generate target -> run candidate -> compare -> score) is sound before any real
recovery search exists.
"""
from __future__ import annotations

from process_bigraph.composite_generator import composite_generator

from ..models.recovery_targets import MASS_ACTION_ANTIMONY, MASS_ACTION_META, RECOVERY_TARGETS
from ._emit import strip_emitter
from ._engine import register_engines
from .engine_monolithic import build_document

DEFAULT_N_STEPS = 100


@composite_generator(
    name="recover_mass_action",
    description=(
        "Candidate-model run for the recover-mass-action study: a flat SBML/Antimony "
        "candidate integrated monolithically in tellurium (libroadrunner), emitting "
        "A/B/C trajectories to compare against the hidden mass-action oracle. Default "
        "candidate = the ground-truth model (identity baseline). Target generation, "
        "trajectory comparison and structural scoring are applied by the recovery "
        "capabilities around this run."
    ),
    parameters={
        "candidate_model": {"type": "string", "default": MASS_ACTION_ANTIMONY},
        "n_steps": {"type": "integer", "default": DEFAULT_N_STEPS},
    },
    default_n_steps=DEFAULT_N_STEPS,
    core_extensions=[register_engines],
    fidelity="real",
    methods=["one tellurium process over the candidate flat-SBML model"],
)
def recover_mass_action(core=None, candidate_model=MASS_ACTION_ANTIMONY, n_steps=DEFAULT_N_STEPS):
    """Run ``candidate_model`` monolithically in tellurium (emitter stripped)."""
    return strip_emitter(
        build_document(
            model=candidate_model,
            engine="tellurium",
            n_steps=n_steps,
            t_end=MASS_ACTION_META["t_end"],
        )
    )


@composite_generator(
    name="recover_candidate",
    description=(
        "Generic candidate-model run for any recovery-ladder study: integrates the "
        "candidate flat-SBML monolithically in tellurium and emits its species "
        "trajectories. `target` selects the ground-truth oracle (recover-mass-action "
        "| recover-enzyme-kinetics | recover-gene-switch | recover-oscillator); "
        "`candidate_model` (optional raw Antimony) overrides it with a proposed model. "
        "Default (empty candidate_model) = the oracle itself, the identity baseline. "
        "Target generation, comparison and structural scoring are applied around the "
        "run by the recovery capabilities."
    ),
    parameters={
        "target": {"type": "string", "default": "recover-mass-action"},
        "candidate_model": {"type": "string", "default": ""},
        "n_steps": {"type": "integer", "default": DEFAULT_N_STEPS},
    },
    default_n_steps=DEFAULT_N_STEPS,
    core_extensions=[register_engines],
    fidelity="real",
    methods=["one tellurium process over the candidate flat-SBML model"],
)
def recover_candidate(core=None, target="recover-mass-action", candidate_model="", n_steps=DEFAULT_N_STEPS):
    """Run a recovery study's candidate model (default = its oracle) in tellurium."""
    entry = RECOVERY_TARGETS[target]
    model = candidate_model or entry["antimony"]
    return strip_emitter(
        build_document(model=model, engine="tellurium", n_steps=n_steps, t_end=entry["meta"]["t_end"])
    )
