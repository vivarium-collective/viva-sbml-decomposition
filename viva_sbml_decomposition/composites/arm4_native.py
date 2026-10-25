"""Arm 4: the fully native process-bigraph glycolysis network (two leaf variants).

The far pole of the decomposition spectrum. Where arms 1-3 wrap libroadrunner,
arm 4 reimplements the Teusink glycolysis ODE RHS as native Python/numpy inside
a single :class:`~viva_sbml_decomposition.processes.native_glycolysis.NativeGlycolysisProcess`
and advances it with an explicit scipy integrator — maximum composability,
hand-written integration, zero simulator at runtime (see that module's docstring
for the faithfulness + stiffness story).

Two composite generators are provided that differ **only** in the species leaf
representation, so the comparison isolates the bigraph framework tax that comes
from state shape alone:

* :func:`glycolysis_native_scalar` — species as a ``map[float]`` (17 float
  leaves), the shape arms 1-3 use.
* :func:`glycolysis_native_array` — species as ONE numpy ``array`` leaf.

Same RHS, same integrator, same tick count — only the leaf type differs. The
array variant additionally publishes a thin ``map[float]`` view of the state for
name-paired emission/accuracy readout (see the process docstring); that view is
not the canonical state leaf.
"""
from __future__ import annotations

import numpy as np
from process_bigraph import allocate_core
from process_bigraph.composite_generator import composite_generator

from viva_sbml_decomposition.models.glycolysis import GLYCOLYSIS_META
from viva_sbml_decomposition.processes.native_glycolysis import (
    ABSOLUTE_TOLERANCE,
    DEFAULT_METHOD,
    RELATIVE_TOLERANCE,
    SPECIES,
    initial_concentrations,
    register_native_glycolysis,
)

from ._emit import strip_emitter
from ._run import extract_run

# Match arm 1's tick count so the arms are directly comparable (same grid).
DEFAULT_N_STEPS = 300


def _interval(n_steps: int) -> float:
    return GLYCOLYSIS_META["t_end"] / n_steps


def _emitter_node(species_source):
    """A RAM-emitter step reading a ``map[float]`` ``species`` store by name."""
    return {
        "_type": "step",
        "address": "local:ram-emitter",
        "config": {
            "emit": {
                "species": "map[float]",
                "time": "float",
                "global_time": "float",
            },
        },
        "inputs": {
            "species": species_source,
            "time": ["global_time"],
            "global_time": ["global_time"],
        },
    }


def _process_node(leaf, interval):
    node = {
        "_type": "process",
        "address": "local:NativeGlycolysisProcess",
        "config": {
            "leaf": leaf,
            "absolute_tolerance": ABSOLUTE_TOLERANCE,
            "relative_tolerance": RELATIVE_TOLERANCE,
            "method": DEFAULT_METHOD,
        },
        "interval": interval,
        "inputs": {"species": ["species"]},
    }
    if leaf == "array":
        node["outputs"] = {
            "species": ["species"],
            "species_named": ["species_named"],
        }
    else:
        node["outputs"] = {"species": ["species"]}
    return node


def _build_document(leaf, n_steps):
    """Build the unstripped arm-4 document for the given ``leaf`` representation.

    ``scalar``: the ``species`` store is a ``map[float]`` seeded with the 17
    initial concentrations; the process reads/writes it and the emitter reads it.

    ``array``: the ``species`` store is ONE numpy array (17,) seeded with the
    initial concentrations in :data:`SPECIES` order; the process reads/writes it
    and also writes a ``map[float]`` ``species_named`` view that the emitter
    reads (so the trajectory round-trips by species name).
    """
    interval = _interval(n_steps)
    initial = initial_concentrations()
    if leaf == "array":
        doc = {
            "species": np.array([initial[s] for s in SPECIES], dtype=float),
            "species_named": dict(initial),
            "glycolysis": _process_node("array", interval),
            "emitter": _emitter_node(["species_named"]),
        }
    else:
        doc = {
            "species": dict(initial),
            "glycolysis": _process_node("scalar", interval),
            "emitter": _emitter_node(["species"]),
        }
    return doc


@composite_generator(
    name="glycolysis_native_scalar",
    description=(
        "Arm 4 (scalar leaf): the 17-species Teusink glycolysis ODE network "
        "reimplemented as a fully native process-bigraph process — hand-ported "
        "rate laws, explicit scipy (LSODA) integration, no libroadrunner at "
        "runtime — with the species state carried as a map[float] (17 float "
        "leaves), the same leaf shape as arms 1-3."
    ),
    default_n_steps=DEFAULT_N_STEPS,
    core_extensions=[register_native_glycolysis],
    fidelity="real",
    methods=["native Python/numpy Teusink RHS", "scipy solve_ivp (LSODA)"],
)
def glycolysis_native_scalar(core=None):
    """Native glycolysis composite, scalar-dict leaf (emitter stripped for the workbench)."""
    return strip_emitter(_build_document("scalar", DEFAULT_N_STEPS))


@composite_generator(
    name="glycolysis_native_array",
    description=(
        "Arm 4 (array leaf): the SAME fully native Teusink glycolysis ODE "
        "network as glycolysis_native_scalar — identical hand-ported RHS and "
        "scipy (LSODA) integration — but with the species state carried as ONE "
        "numpy array leaf instead of 17 scalar map leaves, isolating the bigraph "
        "framework tax attributable purely to state shape."
    ),
    default_n_steps=DEFAULT_N_STEPS,
    core_extensions=[register_native_glycolysis],
    fidelity="real",
    methods=["native Python/numpy Teusink RHS", "scipy solve_ivp (LSODA)"],
)
def glycolysis_native_array(core=None):
    """Native glycolysis composite, numpy-array leaf (emitter stripped for the workbench)."""
    return strip_emitter(_build_document("array", DEFAULT_N_STEPS))


# The task-9 brief's single-name interface; defaults to the scalar leaf.
glycolysis_native = glycolysis_native_scalar


def run_arm(core=None, n_steps: int = DEFAULT_N_STEPS, leaf: str = "scalar"):
    """Run arm 4 locally through process-bigraph; return ``(traj, t)``.

    Uses the *unstripped* document (library RAM emitter) so the history can be
    read back. ``traj`` is ``{species: ndarray}`` over ``GLYCOLYSIS_META["species"]``,
    ``t`` the aligned 1-D time array (n_steps + 1 points, t=0..t_end).

    ``leaf`` selects the species-state representation (``"scalar"`` or
    ``"array"``); both run the identical native RHS + integrator.
    """
    if core is None:
        core = allocate_core()
    register_native_glycolysis(core)
    doc = _build_document(leaf, n_steps)
    return extract_run(doc, core, GLYCOLYSIS_META["t_end"], GLYCOLYSIS_META["species"])
