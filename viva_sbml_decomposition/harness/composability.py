"""Composability scoring rubric — the composability axis of the 3-way tradeoff.

Accuracy and speed are *measured* (the accuracy + timing harnesses). Composability
is not a single number you can read off a run; it is a **rubric judgment** about
how the study's build is structured — can you swap a piece, wire in a coupling, embed it in
a larger multiscale model? This module fixes that rubric: three criteria, each on
a documented 0..3 integer scale (:class:`Level`: Low / Med / High / Max), scored
per study (rubric ids arm1..arm6 are internal keys; see ``STUDY_NAMES``) with a written rationale. The scores are deliberately coarse and
structural — they follow the criteria below, they are NOT reverse-engineered to
make any plot come out a particular way.

Criteria (what each 0..3 level means)
--------------------------------------

**substitutability** — can you replace one piece of the model with a DIFFERENT
implementation without touching the rest?

* 0 (Low)  — one opaque SBML/libroadrunner object; nothing is individually
  addressable, so nothing can be swapped. *(monolith)*
* 1 (Med)  — reactions are partitioned into reaction groups, but the split lives
  *inside one process*; the groups are an implementation detail, not independently
  swappable named nodes. *(mechanical split)*
* 2 (High) — the model is named, separate nodes (submodel processes / a native
  process), each independently replaceable at its port. *(modular, native)*
* 3 (Max)  — a substitution is actually DEMONSTRATED: one node is swapped for a
  different engine and the composite still runs. *(cross-engine coupling)*

**coupling_readiness** — can you wire a NEW coupling (another process, an external
driver, a second model) into the model's state?

* 0 (Low)  — no shared store; coupling anything in means editing the SBML itself.
  *(monolith)*
* 1 (Med)  — a single process boundary; state is internal, so an external coupler
  still has only the one coarse interface to attach to. *(mechanical split)*
* 2 (High) — a shared vivarium store with named species leaves; new nodes couple
  additively through it without touching the existing nodes. *(modular, native)*
* 3 (Max)  — a cross-engine / cross-implementation coupling is actually
  DEMONSTRATED through that shared store. *(cross-engine coupling)*

**multiscale_fit** — how readily does the arm slot into a larger multiscale
composite (other time scales, other state representations, other processes)?

* 0 (Low)  — fixed single scale, single engine, single representation. *(monolith)*
* 1 (Med)  — mechanical split: sub-steppable in time, but still one engine / one
  process. *(mechanical split)*
* 2 (High) — store-coupled nodes: sub-steppable AND composable with other store
  writers. *(modular, ensemble, coupling demo's Tellurium half)*
* 3 (Max)  — a pure-Python process with no simulator dependency and a swappable
  leaf representation (scalar map or contiguous array), embeddable at any scale.
  *(native)*

Accuracy exemption
------------------

``accuracy_exempt`` is the real distinguishing flag the Task-12 figure keys off
to decide which arms appear on the accuracy axis. It is ``True`` ONLY for cross-engine-coupling (arm6)
and ``False`` for the other five. They each have a monolithic reference (sbml-in-tellurium) and
so carry a measured accuracy; cross-engine coupling is the composability EXISTENCE PROOF and has
no dedicated accuracy benchmark of its own, so it is kept off the accuracy axis by
design. (The exemption is NOT because cross-engine coupling's numbers differ — it solves the same
ODEs and reproduces the biological modular seam to ~1e-7; see ``reaction_split``.)

The ``accuracy`` field here is a **placeholder** (``None`` for every arm): this
module scores composability only. The real per-arm accuracy numbers come from the
accuracy harness on an actual run (Task 11/12), keyed on ``accuracy_exempt`` to
skip cross-engine coupling.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional


class Level(IntEnum):
    """Fixed composability scale. Higher = more composable."""

    LOW = 0
    MED = 1
    HIGH = 2
    MAX = 3


MIN_LEVEL = int(Level.LOW)
MAX_LEVEL = int(Level.MAX)

ARM_IDS = ["arm1", "arm2", "arm3", "arm4", "arm5", "arm6"]

# Stable internal rubric ids -> the study each one scores (reader-facing names).
# The ids are internal keys only; every rendered surface uses these display names.
STUDY_SLUGS = {
    "arm1": "sbml-in-tellurium",
    "arm2": "mechanical-split-cost",
    "arm3": "biological-modular-seam",
    "arm4": "fully-native-pbg",
    "arm5": "ensemble-and-multirate",
    "arm6": "cross-engine-coupling",
}
STUDY_NAMES = {
    "arm1": "SBML in Tellurium",
    "arm2": "Mechanical split (additive coupling)",
    "arm3": "Biological modular seam",
    "arm4": "Fully native PBG",
    "arm5": "Ensemble & multi-rate",
    "arm6": "Cross-engine coupling",
}


@dataclass(frozen=True)
class ComposabilityScore:
    """A rubric judgment for one arm, plus its accuracy-axis eligibility.

    ``substitutability`` / ``coupling_readiness`` / ``multiscale_fit`` are each a
    :class:`Level` (0..3). ``accuracy_exempt`` is the distinguishing flag the
    Task-12 viz assembler keys off: ``True`` only for arm6 / cross-engine-coupling (the composability
    existence proof, kept off the accuracy axis), ``False`` for the other five. The
    ``accuracy`` field is a placeholder (``None`` here for every arm) — this module
    scores composability only; real accuracy comes from the accuracy harness on a
    run (Task 11/12). ``rationale`` is the written justification tying the scores
    to the documented criteria.
    """

    arm_id: str
    substitutability: int
    coupling_readiness: int
    multiscale_fit: int
    accuracy_exempt: bool
    accuracy: Optional[float]
    rationale: str


# The rubric table: each arm's scores applied from the criteria documented above.
_RUBRIC = {
    "arm1": ComposabilityScore(
        arm_id="arm1",
        substitutability=Level.LOW,
        coupling_readiness=Level.LOW,
        multiscale_fit=Level.LOW,
        accuracy_exempt=False,
        accuracy=None,
        rationale=(
            "SBML in Tellurium (monolithic): one opaque SBML/libroadrunner object. Nothing is "
            "individually addressable to swap, there is no shared store to wire a "
            "coupling into, and it runs at a single fixed scale — lowest on every "
            "axis."
        ),
    ),
    "arm2": ComposabilityScore(
        arm_id="arm2",
        substitutability=Level.MED,
        coupling_readiness=Level.MED,
        multiscale_fit=Level.MED,
        accuracy_exempt=False,
        accuracy=None,
        rationale=(
            "Mechanical split (additive coupling): reactions are partitioned into two groups, but the "
            "split lives inside one process and advances the full state — the "
            "groups are an implementation detail, not swappable named nodes, and an "
            "external coupler still sees only the one process boundary. Low-ish."
        ),
    ),
    "arm3": ComposabilityScore(
        arm_id="arm3",
        substitutability=Level.HIGH,
        coupling_readiness=Level.HIGH,
        multiscale_fit=Level.HIGH,
        accuracy_exempt=False,
        accuracy=None,
        rationale=(
            "Biological modular seam: separate named SubmodelProcess nodes at the "
            "aldolase seam, each independently replaceable, sharing one species "
            "store that new nodes couple into additively — high substitutability "
            "and coupling-readiness."
        ),
    ),
    "arm4": ComposabilityScore(
        arm_id="arm4",
        substitutability=Level.HIGH,
        coupling_readiness=Level.HIGH,
        multiscale_fit=Level.MAX,
        accuracy_exempt=False,
        accuracy=None,
        rationale=(
            "Fully native PBG process: a named, replaceable node sharing the species "
            "store (high substitutability + coupling-readiness), AND a "
            "simulator-free pure-Python RHS with a swappable scalar/array leaf — "
            "maximally embeddable across scales (multiscale_fit = Max)."
        ),
    ),
    "arm5": ComposabilityScore(
        arm_id="arm5",
        substitutability=Level.HIGH,
        coupling_readiness=Level.HIGH,
        multiscale_fit=Level.HIGH,
        accuracy_exempt=False,
        accuracy=None,
        rationale=(
            "Ensemble & multi-rate: N monolithic instances orchestrated as independent nodes on "
            "a real parallel backend — each instance is a swappable node and the "
            "harness composes them, so it is store/node-composable (high), though "
            "each leaf instance is still a monolith."
        ),
    ),
    "arm6": ComposabilityScore(
        arm_id="arm6",
        substitutability=Level.MAX,
        coupling_readiness=Level.MAX,
        multiscale_fit=Level.MAX,
        accuracy_exempt=True,
        accuracy=None,
        rationale=(
            "Cross-engine coupling: one submodel node is actually swapped for a "
            "different engine (native pbg) while the other stays Tellurium, and the "
            "two reconcile through the shared store — a substitution AND a "
            "cross-engine coupling both demonstrated, the maximum of the axis. "
            "The impossibility for the monolith is STRUCTURAL (you cannot mix a "
            "hand-written process and an SBML submodel in one libroadrunner "
            "integrator), not numeric — cross-engine coupling solves the same ODEs and reproduces "
            "the biological modular seam to ~1e-7. It is accuracy-exempt because it is the composability "
            "existence proof with no dedicated accuracy benchmark, not because its "
            "numbers differ."
        ),
    ),
}


def score_arm(arm_id: str) -> ComposabilityScore:
    """Return the rubric :class:`ComposabilityScore` for ``arm_id`` (e.g. ``"arm3"``).

    Raises ``KeyError`` for an unknown arm id.
    """
    try:
        return _RUBRIC[arm_id]
    except KeyError:
        raise KeyError(f"unknown arm id {arm_id!r} (have {ARM_IDS})") from None
