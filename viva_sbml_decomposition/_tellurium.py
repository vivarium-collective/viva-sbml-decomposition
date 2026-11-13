"""The single tellurium model-load seam for the workspace.

Every place that turns a flat SBML/Antimony string into a loaded libroadrunner
model goes through :func:`load`, so any future integrator settings, tolerances,
or error handling can be applied in exactly one place instead of being scattered
across ~8 `te.loada(...)` call sites.

Deliberate exception: ``composites/_multiorgan.py`` calls ``te.loada`` directly
inside its timing harness, because it *measures* the raw monolithic-build time as
a benchmark — routing that through any wrapper would skew the number it reports.
"""
from __future__ import annotations

import tellurium as te


def load(antimony: str):
    """Load a flat SBML/Antimony string into a libroadrunner model."""
    return te.loada(antimony)
