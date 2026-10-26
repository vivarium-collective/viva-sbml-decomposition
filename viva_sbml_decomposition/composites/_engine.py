"""Engine-dispatch helper shared by the generic composites.

Given an ``engine`` in :data:`ENGINES`, a ``model`` (Antimony string) and a role,
:func:`engine_node` returns an :class:`EngineNode`: the document node for the
engine's process, the ``register`` function its ``core_extensions`` needs, and the
emitter wiring for its output ports. The generic composites
(``engine_monolithic``, and the later ``reaction_split`` / ``ensemble``) assemble
documents from these pieces instead of hand-writing one document per arm.

Roles
-----
* **monolithic** (``reactions=None``): the engine integrates the WHOLE ``model``
  and writes the absolute state (tellurium ``TelluriumProcess`` with bulk species,
  COPASI ``CopasiProcess(coupling_mode="overwrite")``, native
  ``NativeGlycolysisProcess``).
* **subset** (``reactions=[...]``): the engine integrates only those reactions and
  writes an additive per-interval DELTA onto a shared ``map[float]`` species store
  (tellurium ``SubmodelProcess``, COPASI ``CopasiProcess(coupling_mode="delta")``,
  native ``NativeSubmodelProcess`` driven by ``make_subset_rhs``).

Every node's ``config["model"]`` is the Antimony source, so the workbench Model tab
shows it for EVERY engine. Building a node never loads COPASI (abort-safe).

``native`` is glycolysis-specific (a hand-ported RHS): any other model raises.
"""
from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from process_bigraph import allocate_core
from viva_tellurium import TelluriumProcess
from viva_tellurium.composites import register_tellurium, seed_initial_stores

from viva_sbml_decomposition.models.glycolysis import GLYCOLYSIS_ANTIMONY
from viva_sbml_decomposition.processes.copasi_glycolysis import register_copasi
from viva_sbml_decomposition.processes.native_glycolysis import (
    DEFAULT_METHOD,
    NativeSubmodelProcess,
    make_subset_rhs,
    register_native_glycolysis,
)

from .arm3_modular import register_modular_submodels

ENGINES = ("tellurium", "copasi", "native")

ABSOLUTE_TOLERANCE = 1e-10
RELATIVE_TOLERANCE = 1e-8


def register_engines(core=None):
    """Register every engine's process links (+ RAM emitter) on ``core``.

    Registration is cheap and abort-safe (no COPASI model is loaded), so one
    generic composite can declare a single ``core_extensions`` entry for all
    engines.
    """
    if core is None:
        core = allocate_core()
    register_tellurium(core)
    register_modular_submodels(core)
    register_copasi(core)
    register_native_glycolysis(core)
    core.register_link("NativeSubmodelProcess", NativeSubmodelProcess)
    return core


@dataclass
class EngineNode:
    """The pieces needed to place one engine process in a composite document."""

    engine: str
    role: str                       # "monolithic" | "subset"
    node: dict                      # process document node (address/config/interval/wiring)
    register: Callable              # core_extensions fn registering this engine
    emit: dict = field(default_factory=dict)      # emitter ``emit`` schema for the output ports
    emit_inputs: dict = field(default_factory=dict)   # emitter port -> store path
    stores: dict = field(default_factory=dict)    # initial store entries to merge into the doc
    species_order: Optional[tuple] = None         # index order if ``species`` is an array leaf
    finalize: Callable = lambda doc: doc          # post-assembly hook (e.g. tellurium t=0 seeding)


def is_glycolysis(model: str) -> bool:
    return model.strip() == GLYCOLYSIS_ANTIMONY.strip()


def _require_native_ok(model: str) -> None:
    if not is_glycolysis(model):
        raise ValueError(
            "engine='native' is glycolysis-specific (a hand-ported Teusink RHS); "
            "it cannot run an arbitrary model. Use engine='tellurium' or 'copasi'.")


def tellurium_species_order(model: str) -> tuple:
    """Floating-species id order of the bulk ``species`` array (getFloatingSpeciesIds)."""
    core = allocate_core()
    register_tellurium(core)
    proc = TelluriumProcess(
        config={"model": model, "model_format": "antimony", "bulk_species": True},
        core=core)
    return tuple(proc.get_species_ids())


def _tellurium_monolithic(model, interval, parameter_overrides) -> EngineNode:
    order = tuple(tellurium_species_order(model))
    n = len(order)
    out = {k: ["stores", k] for k in ("species", "rates", "parameters", "time")}
    node = {
        "_type": "process",
        "address": "local:TelluriumProcess",
        "config": {
            "model": model, "model_format": "antimony", "integrator": "cvode",
            "absolute_tolerance": ABSOLUTE_TOLERANCE,
            "relative_tolerance": RELATIVE_TOLERANCE,
            "seed": -1, "species_overrides": {},
            "parameter_overrides": parameter_overrides or {},
            "bulk_species": True,
        },
        "interval": interval,
        "inputs": {},           # single process, no upstream writer: it only emits
        "outputs": out,
    }

    def finalize(doc):
        # Seed the wired stores with the model's t=0 state (viva-tellurium issue #7).
        seed_core = allocate_core()
        register_tellurium(seed_core)
        return seed_initial_stores(doc, seed_core)

    return EngineNode(
        "tellurium", "monolithic", node, register_tellurium,
        emit={"species": f"array[({n}),float]", "rates": "map[float]",
              "parameters": "map[float]", "time": "float"},
        emit_inputs=dict(out), stores={"stores": {}},
        species_order=order, finalize=finalize)


def _copasi_monolithic(model, interval) -> EngineNode:
    node = {
        "_type": "process",
        "address": "local:CopasiProcess",
        "config": {
            "model": model, "model_format": "antimony",
            "reactions": [],            # full model
            "coupling_mode": "overwrite",
            "absolute_tolerance": ABSOLUTE_TOLERANCE,
            "relative_tolerance": RELATIVE_TOLERANCE,
        },
        "interval": interval,
        # Re-inject the previous absolute state and integrate one interval
        # (equivalent to continuous integration; keeps the cached model reusable).
        "inputs": {"species": ["stores", "species"]},
        "outputs": {"species": ["stores", "species"], "time": ["stores", "time"]},
    }
    return EngineNode(
        "copasi", "monolithic", node, register_copasi,
        emit={"species": "map[float]", "time": "float"},
        emit_inputs={"species": ["stores", "species"], "time": ["stores", "time"]},
        stores={"stores": {}})


def _native_monolithic(model, interval) -> EngineNode:
    from viva_sbml_decomposition.processes.native_glycolysis import initial_concentrations

    _require_native_ok(model)
    node = {
        "_type": "process",
        "address": "local:NativeGlycolysisProcess",
        "config": {
            "model": model, "leaf": "scalar",
            "absolute_tolerance": ABSOLUTE_TOLERANCE,
            "relative_tolerance": RELATIVE_TOLERANCE,
            "method": DEFAULT_METHOD,
        },
        "interval": interval,
        "inputs": {"species": ["species"]},
        "outputs": {"species": ["species"]},
    }
    return EngineNode(
        "native", "monolithic", node, register_native_glycolysis,
        emit={"species": "map[float]"}, emit_inputs={"species": ["species"]},
        stores={"species": dict(initial_concentrations())})


def _subset(engine, model, reactions, interval, species_path) -> EngineNode:
    reactions = list(reactions)
    if not reactions:
        raise ValueError("a reaction-subset role needs a non-empty reaction list")
    wires = {"inputs": {"species": list(species_path)},
             "outputs": {"species": list(species_path)}}
    tol = {"absolute_tolerance": ABSOLUTE_TOLERANCE,
           "relative_tolerance": RELATIVE_TOLERANCE}
    if engine == "tellurium":
        address, cfg, reg = ("local:SubmodelProcess",
                             {"model": model, "reactions": reactions, **tol},
                             register_modular_submodels)
    elif engine == "copasi":
        address, cfg, reg = ("local:CopasiProcess",
                             {"model": model, "model_format": "antimony",
                              "reactions": reactions, "coupling_mode": "delta", **tol},
                             register_copasi)
    else:
        _require_native_ok(model)
        make_subset_rhs(reactions)      # validate the reaction names up front
        address, cfg, reg = ("local:NativeSubmodelProcess",
                             {"model": model, "reactions": reactions, **tol,
                              "method": DEFAULT_METHOD},
                             register_engines)
    node = {"_type": "process", "address": address, "config": cfg,
            "interval": interval, **wires}
    return EngineNode(engine, "subset", node, reg)


def engine_node(engine: str, model: str, *, interval: float,
                reactions: Optional[Sequence[str]] = None,
                species_path: Sequence[str] = ("species",),
                parameter_overrides: Optional[dict] = None) -> EngineNode:
    """Dispatch ``engine`` -> :class:`EngineNode` for the monolithic or subset role.

    ``reactions=None`` -> monolithic (whole ``model``); a reaction list -> subset
    (delta onto the shared ``species_path`` store). ``parameter_overrides`` applies
    to the tellurium monolith only.
    """
    if engine not in ENGINES:
        raise ValueError(f"unknown engine {engine!r}; expected one of {ENGINES}")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("engine_node needs a non-empty Antimony `model` string")
    if reactions is not None:
        return _subset(engine, model, reactions, interval, species_path)
    if engine == "tellurium":
        return _tellurium_monolithic(model, interval, parameter_overrides)
    if engine == "copasi":
        return _copasi_monolithic(model, interval)
    return _native_monolithic(model, interval)


# --------------------------------------------------------------------------- #
# Abort-safe subprocess execution (COPASI).                                   #
#                                                                             #
# COPASI's model load aborts the process NONDETERMINISTICALLY on this         #
# python-copasi build (an uncatchable C++ abort()). Same policy as arm 10:    #
# run the real composite in a fresh subprocess, retry ONLY on a signal kill   #
# (negative returncode), fail fast on a Python error, return the result by    #
# pickle. Building/registering never loads COPASI, so this is only for runs.  #
# --------------------------------------------------------------------------- #

_SUBPROC_SRC = (
    "import sys, json, pickle, importlib\n"
    "payload = json.loads(sys.argv[1]); out = sys.argv[2]\n"
    "fn = getattr(importlib.import_module(payload['module']), payload['func'])\n"
    "res = fn(**payload['kwargs'])\n"
    "with open(out, 'wb') as f:\n"
    "    pickle.dump({'traj': res[0], 't': res[1]}, f)\n"
)


def run_in_subprocess_with_retry(module: str, func: str, kwargs: dict, *, retries: int = 6):
    """Call ``module.func(**kwargs)`` (must return ``(traj, t)``) in a retrying subprocess."""
    payload = json.dumps({"module": module, "func": func, "kwargs": kwargs})
    fd, out_path = tempfile.mkstemp(suffix=".pkl")
    os.close(fd)
    try:
        last = None
        for _ in range(retries):
            proc = subprocess.run(
                [sys.executable, "-c", _SUBPROC_SRC, payload, out_path],
                capture_output=True, text=True)
            if proc.returncode == 0 and os.path.getsize(out_path) > 0:
                with open(out_path, "rb") as fh:
                    return pickle.load(fh)
            last = proc
            if proc.returncode is not None and proc.returncode > 0:
                raise RuntimeError(
                    f"subprocess {module}.{func} failed with a Python error "
                    f"(returncode={proc.returncode}); not retried.\n"
                    f"stderr tail:\n{(proc.stderr or '')[-1500:]}")
        raise RuntimeError(
            f"subprocess {module}.{func} was signal-aborted in all {retries} attempts "
            f"(last returncode={getattr(last, 'returncode', '?')}); "
            f"stderr tail: {(getattr(last, 'stderr', '') or '')[-800:]}")
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)
