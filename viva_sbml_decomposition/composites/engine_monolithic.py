"""Generic monolithic composite: ONE process running a whole model in any engine.

``engine_monolithic(model, engine, n_steps)`` replaces the per-engine monolithic
arms (arm 1 tellurium/glycolysis, arm 8 tellurium/multi-organ, arm 10 COPASI,
arm 4 native) with a single generator in which the *model is data*: ``model`` is a
registry id (``glycolysis``, ``multiorgan_scaled``, ...) or a raw Antimony string,
and ``engine`` is ``tellurium`` / ``copasi`` / ``native``. The emitted document
carries ``config.model`` = the Antimony for EVERY engine.

``native_composite(model)`` is the ``engine="native"`` monolith, kept under its own
name as the referent of the "fully native" study. Native is glycolysis-specific.

The ``run_*`` helpers run the UNSTRIPPED document (library RAM emitter) and return
``(traj, t)`` like the arms' ``run_arm``; COPASI runs go through arm 10's retrying
subprocess so its intermittent loader abort can never kill the caller.
"""
from __future__ import annotations

from process_bigraph import allocate_core
from process_bigraph.composite_generator import composite_generator

from viva_sbml_decomposition.models.glycolysis import GLYCOLYSIS_ANTIMONY, GLYCOLYSIS_META

from ._emit import strip_emitter
from ._engine import (ENGINES, EngineNode, engine_node, register_engines,
                      run_in_subprocess_with_retry, tellurium_species_order)
from ._run import extract_run, extract_run_timed

DEFAULT_N_STEPS = 300
DEFAULT_T_END = 10.0
# Simulation horizon per registry model (the arms' own horizons).
T_END_BY_MODEL = {"glycolysis": GLYCOLYSIS_META["t_end"], "multiorgan_scaled": 10.0}


def resolve_model(model: str):
    """Return ``(model_id_or_None, antimony)`` for a registry id or raw Antimony."""
    # Lazy: models.registry imports composites.abm, so a top-level import here
    # would be circular when the registry is imported first.
    from ..models.registry import MODELS

    if model in MODELS:
        return model, MODELS[model]["antimony"]
    if "\n" in model or ";" in model or model.lstrip().startswith("model "):
        return None, model
    raise KeyError(f"unknown model {model!r}; registry ids: {sorted(MODELS)}")


def _build(model, engine, n_steps, t_end, parameter_overrides=None, native_leaf="scalar"):
    model_id, antimony = resolve_model(model)
    if t_end is None:
        t_end = T_END_BY_MODEL.get(model_id, DEFAULT_T_END)
    en = engine_node(engine, antimony, interval=t_end / n_steps,
                     parameter_overrides=parameter_overrides, native_leaf=native_leaf)
    return en, t_end


def _document(en: EngineNode) -> dict:
    """Unstripped document: the engine node + shared stores + RAM emitter."""
    doc = {en.engine: en.node, **{k: dict(v) if isinstance(v, dict) else v
                                  for k, v in en.stores.items()}}
    emit = {**en.emit, "global_time": "float"}
    doc["emitter"] = {
        "_type": "step",
        "address": "local:ram-emitter",
        "config": {"emit": emit},
        "inputs": {**en.emit_inputs, "global_time": ["global_time"]},
    }
    if "time" not in en.emit_inputs:     # native: no time port; use the scheduler clock
        emit["time"] = "float"
        doc["emitter"]["inputs"]["time"] = ["global_time"]
    return en.finalize(doc)


def build_document(model="glycolysis", engine="tellurium", n_steps=DEFAULT_N_STEPS,
                   t_end=None, parameter_overrides=None) -> dict:
    """Unstripped engine_monolithic document (no engine model is loaded for COPASI)."""
    en, _ = _build(model, engine, n_steps, t_end, parameter_overrides)
    return _document(en)


@composite_generator(
    name="engine_monolithic",
    description=(
        "Generic monolith: a whole SBML/Antimony model (registry id or raw "
        "Antimony) in ONE process node of the chosen engine — tellurium "
        "(libroadrunner/CVODE, bulk species), copasi (COPASI/basico LSODA, "
        "overwrite coupling) or native (glycolysis-only hand-ported RHS). "
        "config.model carries the Antimony for every engine."
    ),
    parameters={
        "model": {"type": "string", "default": "glycolysis"},
        "engine": {"type": "string", "default": "tellurium"},
        "n_steps": {"type": "integer", "default": DEFAULT_N_STEPS},
    },
    default_n_steps=DEFAULT_N_STEPS,
    core_extensions=[register_engines],
    fidelity="real",
    methods=["one engine process over the whole model (tellurium | copasi | native)"],
)
def engine_monolithic(core=None, model="glycolysis", engine="tellurium",
                      n_steps=DEFAULT_N_STEPS):
    """Whole ``model`` in one ``engine`` process (emitter stripped for the workbench)."""
    return strip_emitter(build_document(model, engine, n_steps))


@composite_generator(
    name="native_composite",
    description=(
        "The fully-native pbg reference: the glycolysis ODE network as a hand-ported "
        "NativeGlycolysisProcess (scalar map[float] leaf, scipy LSODA, no simulator "
        "at runtime). The engine='native' monolith; glycolysis-specific."
    ),
    parameters={"model": {"type": "string", "default": "glycolysis"},
                "n_steps": {"type": "integer", "default": DEFAULT_N_STEPS}},
    default_n_steps=DEFAULT_N_STEPS,
    core_extensions=[register_engines],
    fidelity="real",
    methods=["native Python/numpy Teusink RHS", "scipy solve_ivp (LSODA)"],
)
def native_composite(core=None, model="glycolysis", n_steps=DEFAULT_N_STEPS):
    """Native glycolysis monolith (emitter stripped); raises for non-glycolysis models."""
    return strip_emitter(build_document(model, "native", n_steps))


def _species_ids(en: EngineNode, antimony: str):
    return list(en.species_order or tellurium_species_order(antimony))


def _run_inproc(model, engine, n_steps, t_end, parameter_overrides=None):
    en, t_end = _build(model, engine, n_steps, t_end, parameter_overrides)
    _, antimony = resolve_model(model)
    core = allocate_core()
    en.register(core)
    register_engines(core)
    return extract_run(_document(en), core, t_end, _species_ids(en, antimony),
                       species_order=list(en.species_order) if en.species_order else None)


def run_engine_monolithic(model="glycolysis", engine="tellurium",
                          n_steps=DEFAULT_N_STEPS, t_end=None, parameter_overrides=None):
    """Run the monolith; return ``(traj, t)``.

    COPASI runs execute in a retrying subprocess (abort-safe, same policy as arm
    10); ``model`` is passed as JSON, so it is a registry id or Antimony string.
    """
    if engine not in ENGINES:
        raise ValueError(f"unknown engine {engine!r}; expected one of {ENGINES}")
    if engine == "copasi":
        data = run_in_subprocess_with_retry(
            "viva_sbml_decomposition.composites.engine_monolithic", "_run_inproc",
            {"model": model, "engine": "copasi", "n_steps": n_steps, "t_end": t_end})
        return data["traj"], data["t"]
    return _run_inproc(model, engine, n_steps, t_end, parameter_overrides)


def run_native_composite(model="glycolysis", n_steps=DEFAULT_N_STEPS, t_end=None):
    return run_engine_monolithic(model, "native", n_steps, t_end)


def time_engine_monolithic(model="glycolysis", engine="tellurium", n_steps=DEFAULT_N_STEPS,
                           t_end=None, *, detail=True, native_leaf="scalar"):
    """Run the monolith in-process and return ``(traj, t, RunTiming)``.

    The process-vs-framework timing split behind the overhead figures
    (``detail=True`` also fills ``timing.per_process``). In-process, so not for COPASI.
    """
    en, t_end = _build(model, engine, n_steps, t_end, native_leaf=native_leaf)
    _, antimony = resolve_model(model)
    core = allocate_core()
    en.register(core)
    register_engines(core)
    return extract_run_timed(_document(en), core, t_end, _species_ids(en, antimony),
                             detail=detail,
                             species_order=list(en.species_order) if en.species_order else None)
