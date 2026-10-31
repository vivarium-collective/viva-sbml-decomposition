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

# --- Rung 2: Michaelis-Menten enzyme kinetics -------------------------------
# Elementary mechanism S + E <-> ES -> E + P. The reduced saturating law is the
# observationally-equivalent alternative when only S/P are observed.
ENZYME_ANTIMONY = r"""
model enzyme_mm()
  species S, E, ES, P;
  S = 10.0; E = 1.0; ES = 0.0; P = 0.0;
  k1 = 0.5; k2 = 0.1; kcat = 0.4;
  r1: S + E -> ES; k1*S*E;
  r2: ES -> S + E; k2*ES;
  r3: ES -> E + P; kcat*ES;
end
""".strip()

ENZYME_META = {"name": "enzyme_mm", "species": ["S", "E", "ES", "P"], "t_end": 20.0}

# --- Rung 3: Hill gene switch (bistable toggle) -----------------------------
# Two mutually-repressing genes; Hill repression gives two stable steady states.
GENE_SWITCH_ANTIMONY = r"""
model gene_toggle()
  species u, v;
  u = 1.0; v = 0.1;
  a = 4.0; n = 3.0; g = 1.0;
  Ju_prod: => u; a/(1 + v^n);
  Ju_deg:  u => ; g*u;
  Jv_prod: => v; a/(1 + u^n);
  Jv_deg:  v => ; g*v;
end
""".strip()

GENE_SWITCH_META = {"name": "gene_toggle", "species": ["u", "v"], "t_end": 20.0}

# --- Rung 4: Brusselator oscillator -----------------------------------------
# Limit-cycle oscillator (A=1, B=3 is well past the Hopf bifurcation).
OSCILLATOR_ANTIMONY = r"""
model brusselator()
  species X, Y;
  X = 1.0; Y = 1.0;
  A = 1.0; B = 3.0;
  r1: => X; A;
  r2: 2 X + Y => 3 X; X^2*Y;
  r3: X => Y; B*X;
  r4: X => ; X;
end
""".strip()

OSCILLATOR_META = {"name": "brusselator", "species": ["X", "Y"], "t_end": 20.0}

# Registry of ground-truth targets by study slug — one oracle per recovery rung.
RECOVERY_TARGETS = {
    "recover-mass-action": {"antimony": MASS_ACTION_ANTIMONY, "meta": MASS_ACTION_META},
    "recover-enzyme-kinetics": {"antimony": ENZYME_ANTIMONY, "meta": ENZYME_META},
    "recover-gene-switch": {"antimony": GENE_SWITCH_ANTIMONY, "meta": GENE_SWITCH_META},
    "recover-oscillator": {"antimony": OSCILLATOR_ANTIMONY, "meta": OSCILLATOR_META},
}
