"""Ground-truth models for the discovery-engine SBML-recovery benchmark.

Each entry is a small, fully-specified Antimony model that serves as the HIDDEN
oracle for a recovery study: a SyntheticTargetGenerator simulates it to produce
the target trajectories the build loop sees, and a StructuralScorer later grades
a recovered model against the known reactions/parameters here.

These are deliberately minimal canonical motifs (the recovery difficulty ladder),
not biology — the point is an exact answer key.
"""
from __future__ import annotations

# --- Rung 1: mass-action ----------------------------------------------------
# A -> B (k1); A + B <-> C (k2 / k2r). Three species, three reactions, all
# mass-action — the identifiable floor of the ladder.
MASS_ACTION_ANTIMONY = r"""
model mass_action()
  species A, B, C;
  A = 10.0; B = 0.0; C = 0.0;
  k1 = 0.5; k2 = 0.3; k2r = 0.1;
  r1: A -> B; k1*A;
  r2: A + B -> C; k2*A*B;
  r2r: C -> A + B; k2r*C;
end
""".strip()

MASS_ACTION_META = {
    "name": "mass_action_ladder_r1",
    "species": ["A", "B", "C"],
    "t_end": 10.0,
    # The answer key the structural scorer grades against.
    "reactions": {
        "r1": {"reactants": {"A": 1}, "products": {"B": 1}, "rate": "k1*A", "params": {"k1": 0.5}},
        "r2": {"reactants": {"A": 1, "B": 1}, "products": {"C": 1}, "rate": "k2*A*B", "params": {"k2": 0.3}},
        "r2r": {"reactants": {"C": 1}, "products": {"A": 1, "B": 1}, "rate": "k2r*C", "params": {"k2r": 0.1}},
    },
}

# Registry of ground-truth targets by study slug (extended as rungs are added:
# recover-enzyme-kinetics, recover-gene-switch, recover-oscillator).
RECOVERY_TARGETS = {
    "recover-mass-action": {"antimony": MASS_ACTION_ANTIMONY, "meta": MASS_ACTION_META},
}
