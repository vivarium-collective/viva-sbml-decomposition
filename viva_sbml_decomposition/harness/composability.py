"""Composability scoring rubric — the composability axis of the 3-way tradeoff.

Accuracy and speed are *measured* (the accuracy + timing harnesses). Composability
is not a single number you can read off a run; it is a **rubric judgment** about
how the arm is structured — can you swap a piece, wire in a coupling, embed it in
a larger multiscale model? This module fixes that rubric: three criteria, each on
a documented 0..3 integer scale (:class:`Level`: Low / Med / High / Max), scored
per arm with a written rationale. The scores are deliberately coarse and
structural — they follow the criteria below, they are NOT reverse-engineered to
make any plot come out a particular way.

Criteria (what each 0..3 level means)
--------------------------------------

**substitutability** — can you replace one piece of the model with a DIFFERENT
implementation without touching the rest?

* 0 (Low)  — one opaque SBML/libroadrunner object; nothing is individually
  addressable, so nothing can be swapped. *(monolith)*
* 1 (Med)  — reactions are partitioned into operator groups, but the split lives
  *inside one process*; the groups are an implementation detail, not independently
  swappable named nodes. *(operator split)*
* 2 (High) — the model is named, separate nodes (submodel processes / a native
  process), each independently replaceable at its port. *(modular, native)*
* 3 (Max)  — a substitution is actually DEMONSTRATED: one node is swapped for a
  different engine and the composite still runs. *(coupling demo)*

**coupling_readiness** — can you wire a NEW coupling (another process, an external
driver, a second model) into the model's state?

* 0 (Low)  — no shared store; coupling anything in means editing the SBML itself.
  *(monolith)*
* 1 (Med)  — a single process boundary; state is internal, so an external coupler
  still has only the one coarse interface to attach to. *(operator split)*
* 2 (High) — a shared vivarium store with named species leaves; new nodes couple
  additively through it without touching the existing nodes. *(modular, native)*
* 3 (Max)  — a cross-engine / cross-implementation coupling is actually
  DEMONSTRATED through that shared store. *(coupling demo)*

**multiscale_fit** — how readily does the arm slot into a larger multiscale
composite (other time scales, other state representations, other processes)?

* 0 (Low)  — fixed single scale, single engine, single representation. *(monolith)*
* 1 (Med)  — operator-split: sub-steppable in time, but still one engine / one
  process. *(operator split)*
* 2 (High) — store-coupled nodes: sub-steppable AND composable with other store
  writers. *(modular, ensemble, coupling demo's Tellurium half)*
* 3 (Max)  — a pure-Python process with no simulator dependency and a swappable
  leaf representation (scalar map or contiguous array), embeddable at any scale.
  *(native)*

Accuracy exemption
------------------

Every arm except arm 6 has a monolithic reference (arm 1) and so carries a
measured accuracy elsewhere; its ``accuracy_exempt`` is ``False``. Arm 6 mixes
engines, which changes the numerics, so it has NO monolithic reference by
construction: its record carries ``accuracy=None`` and ``accuracy_exempt=True``,
the flag the Task-12 figure consumes to keep arm 6 OFF the accuracy axis.
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


@dataclass(frozen=True)
class ComposabilityScore:
    """A rubric judgment for one arm, plus its accuracy-axis eligibility.

    ``substitutability`` / ``coupling_readiness`` / ``multiscale_fit`` are each a
    :class:`Level` (0..3). ``accuracy`` is the arm's accuracy-axis value: ``None``
    marks the arm accuracy-exempt (no monolithic reference), which ``accuracy_exempt``
    mirrors as an explicit boolean for the Task-12 viz assembler. ``rationale`` is
    the written justification tying the scores to the documented criteria.
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
            "Monolithic: one opaque SBML/libroadrunner object. Nothing is "
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
            "Operator split: reactions are partitioned into two groups, but the "
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
            "Modular submodels: separate named SubmodelProcess nodes at the "
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
            "Native pbg process: a named, replaceable node sharing the species "
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
            "Ensemble: N monolithic instances orchestrated as independent nodes on "
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
            "Coupling demonstration: one submodel node is actually swapped for a "
            "different engine (native pbg) while the other stays Tellurium, and the "
            "two reconcile through the shared store — a substitution AND a "
            "cross-engine coupling both demonstrated, the maximum of the axis. "
            "Mixing engines changes the numerics, so it has no monolithic reference "
            "and is accuracy-exempt (kept off the accuracy axis)."
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
