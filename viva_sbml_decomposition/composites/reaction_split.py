"""Generic reaction-split composite: two reaction groups, two (possibly different)
engines, ONE shared additive species store.

``reaction_split(model, seam, engine_a, engine_b, coupling_interval)`` replaces the
hand-written split/coupling arms (3 modular, 6 tellurium+native, 10 tellurium+COPASI,
and the mechanical split of 2) with a single generator in which the model, the
reaction partition (``seam``) and each side's engine are data.

Wiring (identical to arms 3/6/10): node A (``engine_a``, the *upper* reactions) and
node B (``engine_b``, the *lower* reactions) both read the shared ``["species"]``
store, integrate only their own reactions over ``coupling_interval``, and write a
per-interval DELTA that applies additively (``map[float]`` is additive; deltas from
both nodes sum). The store is seeded with the model's initial concentrations. If
the model has SBML assignment rules (glycolysis: ATP/ADP/AMP), an ``EnergyStep``
re-derives those species from the reconciled pool each step.

``seam``:

* ``"aldolase"`` - arm 3/6/10's biological upper/lower glycolysis seam.
* ``"mechanical"`` / ``"operator"`` - arm 2's partition (core network vs the
  vGLYCO/vTreha storage sinks).
* ``{"upper": [...], "lower": [...]}`` - explicit reaction lists (any model).

Arm 2 caveat: arm 2 is one bespoke ``OperatorSplitProcess`` doing a sequential Lie
split with overwrite coupling; this composite is always the two-node additive
(Jacobi) split. With the arm-2 partition it is the closest faithful reproduction,
NOT a bit-identical one (same first-order splitting error order, different
constant).
"""
from __future__ import annotations

from process_bigraph import allocate_core
from process_bigraph.composite_generator import composite_generator

from viva_sbml_decomposition.models.glycolysis import GLYCOLYSIS_META
from viva_sbml_decomposition.models.glycolysis_submodels import (
    LOWER_GLYCOLYSIS, UPPER_GLYCOLYSIS)

from ._emit import strip_emitter
from ._engine import (ENGINES, engine_node, register_engines,
                      run_in_subprocess_with_retry, tellurium_species_order)
from ._run import extract_run
from .arm2_operator_split import DEFAULT_SPLIT_B
from .arm3_modular import _load, _species_metadata
from .engine_monolithic import T_END_BY_MODEL, resolve_model

SPECIES_PATH = ("species",)

# name -> (upper node name, upper reactions, lower node name, lower reactions,
#          default coupling interval of the arm it reproduces)
SEAMS = {
    "aldolase": ("upper_glycolysis", list(UPPER_GLYCOLYSIS),
                 "lower_glycolysis", list(LOWER_GLYCOLYSIS), 0.0005),
}
SEAM_ALIASES = {"mechanical": "mechanical", "operator": "mechanical"}
DEFAULT_COUPLING_INTERVAL = 0.0005
MECHANICAL_COUPLING_INTERVAL = 0.05     # arm 2's default
EXPLICIT_COUPLING_INTERVAL = 0.005


def resolve_seam(model_id, antimony, seam):
    """Return ``(name_a, reactions_a, name_b, reactions_b, default_interval)``."""
    if isinstance(seam, dict):
        up, lo = list(seam.get("upper", [])), list(seam.get("lower", []))
        if not up or not lo or set(up) & set(lo):
            raise ValueError("explicit seam needs non-empty, disjoint 'upper' and 'lower'")
        return "upper", up, "lower", lo, EXPLICIT_COUPLING_INTERVAL
    if seam in SEAMS:
        return SEAMS[seam]
    if seam in SEAM_ALIASES:
        rr = _load(antimony)
        lower = list(DEFAULT_SPLIT_B)
        upper = [r for r in rr.getReactionIds() if r not in set(lower)]
        return "core", upper, "storage", lower, MECHANICAL_COUPLING_INTERVAL
    raise ValueError(f"unknown seam {seam!r}; named: {sorted(SEAMS) + sorted(SEAM_ALIASES)} "
                     "or {'upper': [...], 'lower': [...]}")


def _initial_species(antimony):
    rr = _load(antimony)
    ids, assigned, _ = _species_metadata(rr)
    return {s: float(rr[s]) for s in ids}, bool(assigned)


def _build(model, seam, engine_a, engine_b, coupling_interval, n_steps=None, t_end=None):
    model_id, antimony = resolve_model(model)
    name_a, rx_a, name_b, rx_b, default_dt = resolve_seam(model_id, antimony, seam)
    dt = default_dt if coupling_interval is None else coupling_interval
    if t_end is None:
        t_end = T_END_BY_MODEL.get(model_id, 10.0)
    ea = engine_node(engine_a, antimony, interval=dt, reactions=rx_a, species_path=SPECIES_PATH)
    eb = engine_node(engine_b, antimony, interval=dt, reactions=rx_b, species_path=SPECIES_PATH)
    init, has_rules = _initial_species(antimony)
    doc = {"species": init, name_a: ea.node, name_b: eb.node}
    if has_rules:
        doc["energy"] = {
            "_type": "step", "address": "local:EnergyStep",
            "config": {"model": antimony},
            "inputs": {"species": ["species"]}, "outputs": {"species": ["species"]},
        }
    doc["emitter"] = {
        "_type": "step", "address": "local:ram-emitter",
        "config": {"emit": {"species": "map[float]", "time": "float",
                            "global_time": "float"}},
        "inputs": {"species": ["species"], "time": ["global_time"],
                   "global_time": ["global_time"]},
    }
    return doc, antimony, t_end


def build_document(model="glycolysis", seam="aldolase", engine_a="tellurium",
                   engine_b="tellurium", coupling_interval=None) -> dict:
    """Unstripped reaction_split document (never loads COPASI)."""
    return _build(model, seam, engine_a, engine_b, coupling_interval)[0]


@composite_generator(
    name="reaction_split",
    description=(
        "Generic reaction split: a model's reactions partitioned by `seam` "
        "(aldolase | mechanical | explicit upper/lower lists) into two process "
        "nodes, each in its own engine (tellurium | copasi | native), coupled only "
        "through ONE shared additive species store (per-interval deltas sum). "
        "Reproduces arms 2/3/6/10-cross; config.model carries the Antimony."
    ),
    parameters={
        "model": {"type": "string", "default": "glycolysis"},
        "seam": {"type": "string", "default": "aldolase"},
        "engine_a": {"type": "string", "default": "tellurium"},
        "engine_b": {"type": "string", "default": "tellurium"},
        "coupling_interval": {"type": "float", "default": DEFAULT_COUPLING_INTERVAL},
    },
    core_extensions=[register_engines],
    fidelity="real",
    methods=["two engine nodes over one additive species store (Jacobi split)"],
)
def reaction_split(core=None, model="glycolysis", seam="aldolase", engine_a="tellurium",
                   engine_b="tellurium", coupling_interval=DEFAULT_COUPLING_INTERVAL):
    """Two-engine reaction-split composite (emitter stripped for the workbench)."""
    return strip_emitter(build_document(model, seam, engine_a, engine_b, coupling_interval))


def _run_inproc(model="glycolysis", seam="aldolase", engine_a="tellurium",
                engine_b="tellurium", coupling_interval=None, t_end=None):
    doc, antimony, t_end = _build(model, seam, engine_a, engine_b, coupling_interval,
                                  t_end=t_end)
    core = allocate_core()
    register_engines(core)
    return extract_run(doc, core, t_end, list(tellurium_species_order(antimony)))


def run_reaction_split(model="glycolysis", seam="aldolase", engine_a="tellurium",
                       engine_b="tellurium", coupling_interval=None, t_end=None):
    """Run the split; return ``(traj, t)``. COPASI runs use arm 10's abort-safe
    retrying subprocess (``seam`` must then be JSON-serialisable)."""
    for e in (engine_a, engine_b):
        if e not in ENGINES:
            raise ValueError(f"unknown engine {e!r}; expected one of {ENGINES}")
    kw = dict(model=model, seam=seam, engine_a=engine_a, engine_b=engine_b,
              coupling_interval=coupling_interval, t_end=t_end)
    if "copasi" in (engine_a, engine_b):
        data = run_in_subprocess_with_retry(
            "viva_sbml_decomposition.composites.reaction_split", "_run_inproc", kw)
        return data["traj"], data["t"]
    return _run_inproc(**kw)
