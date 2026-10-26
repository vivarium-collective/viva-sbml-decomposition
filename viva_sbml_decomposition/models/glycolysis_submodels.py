"""Biological partition of the Teusink glycolysis model for arm 3.

Arm 3 breaks the model along a **biologically-meaningful seam**, not a mechanical
operator split (contrast arm 2, which carved off the two weakly-coupled G6P
storage sinks purely because they keep both halves well-posed).

The seam is the textbook **upper/lower glycolysis boundary at aldolase**
(fructose-1,6-bisphosphate, ``F16P``):

* ``upper_glycolysis`` — the *preparatory / ATP-investment* phase: glucose
  transport and phosphorylation through to F16P (``vGLT``, ``vGLK``, ``vPGI``,
  ``vPFK``), together with the two hexose-phosphate storage sinks that branch
  off G6P (``vGLYCO`` glycogen, ``vTreha`` trehalose). Net ATP *consumer*.
* ``lower_glycolysis`` — the *payoff* phase: F16P cleaved to trioses and carried
  to pyruvate (``vALD``, ``vGAPDH``, ``vPGK``, ``vPGM``, ``vENO``, ``vPYK``),
  the fermentation / redox branches (``vPDC``, ``vADH``, ``vG3PDH``, ``vSUC``),
  and the ATPase maintenance load (``vATP``). Net ATP + NADH *producer*.

The two submodels share the boundary metabolites they both touch — the aldolase
substrate ``F16P``, the ``P``/``ATP``/``ADP``/``AMP`` high-energy-phosphate pool,
and the ``NAD``/``NADH`` redox pair — which are reconciled through a common
vivarium store.

**Caveat (carried from arm 2):** this seam cuts straight through the strong
phosphate/energy coupling. ``ATP``/``ADP``/``AMP`` are algebraic functions of
the settable phosphate pool ``P`` (and the conserved sum ``SUM_P``) via a sqrt
assignment rule whose radicand goes negative once ``P`` drifts too far. A pure
Lie split (each submodel advancing a whole coupling interval in isolation) drives
``P`` out of that domain and CVODE blows up for *every* interval — arm 2 found
this seam intractable that way. Arm 3 keeps it well-posed by sub-stepping the
additive reconciliation (see :mod:`viva_sbml_decomposition.composites.reaction_split`).
"""
from __future__ import annotations

# Upper glycolysis: preparatory / ATP-investment phase + G6P storage sinks.
UPPER_GLYCOLYSIS = ["vGLT", "vGLK", "vPGI", "vPFK", "vGLYCO", "vTreha"]

# Lower glycolysis: payoff phase + fermentation / redox branches + ATPase.
LOWER_GLYCOLYSIS = [
    "vALD", "vGAPDH", "vPGK", "vPGM", "vENO", "vPYK",
    "vPDC", "vADH", "vG3PDH", "vSUC", "vATP",
]

# The biological partition consumed by arm 3 — one entry per submodel process
# node. Both submodels write to the shared boundary pool they cross (F16P, the
# P/ATP/ADP/AMP energy pool, NAD/NADH), which reconciles additively.
SUBMODELS = [
    ("upper_glycolysis", UPPER_GLYCOLYSIS),
    ("lower_glycolysis", LOWER_GLYCOLYSIS),
]
