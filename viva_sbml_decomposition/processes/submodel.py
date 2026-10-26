"""Reaction-subset processes shared by the generic ``reaction_split`` composite.

* :class:`SubmodelProcess` - a standalone libroadrunner instance keeping only a
  subset of a model's reactions; each interval it writes the additive DELTA it
  produced on the settable species (a ``map[float]`` store applies float updates
  additively, so deltas from several nodes reconcile by summation).
* :class:`EnergyStep` - re-derives SBML assignment-rule species (glycolysis
  ATP/ADP/AMP) from the reconciled pool each step.

An over-coarse coupling interval leaves the sqrt domain of the ADP assignment rule
and CVODE fails; :class:`SubmodelProcess` catches that and writes a non-finite
delta, so the run surfaces as *diverged* instead of crashing the composite.
"""
from __future__ import annotations

import libsbml
from process_bigraph import Process, Step, allocate_core
from process_bigraph.emitter import RAMEmitter

# Matched integrator tolerances (identical across every engine node).
ABSOLUTE_TOLERANCE = 1e-10
RELATIVE_TOLERANCE = 1e-8


def _load(model_antimony):
    import tellurium as te

    return te.loada(model_antimony)


def _species_metadata(rr):
    """(all floating species, assignment-rule species, settable species) for a model."""
    doc = libsbml.readSBMLFromString(rr.getCurrentSBML())
    model = doc.getModel()
    assigned = {
        model.getRule(i).getVariable()
        for i in range(model.getNumRules())
        if model.getRule(i).isAssignment()
    }
    species_ids = list(rr.getFloatingSpeciesIds())
    settable = [s for s in species_ids if s not in assigned]
    return species_ids, assigned, settable


def _submodel_sbml(base_sbml, keep_reactions):
    """SBML keeping only ``keep_reactions`` (all species/params/rules kept)."""
    doc = libsbml.readSBMLFromString(base_sbml)
    model = doc.getModel()
    keep = set(keep_reactions)
    for rid in [model.getReaction(i).getId() for i in range(model.getNumReactions())]:
        if rid not in keep:
            model.removeReaction(rid)
    return libsbml.writeSBMLToString(doc)


class SubmodelProcess(Process):
    """One biological submodel as a standalone bigraph process node.

    Holds a single libroadrunner instance built from the SAME model but keeping
    only ``reactions`` (all species, parameters and assignment rules retained).
    Each ``update(interval)`` reads the shared settable species from the store,
    advances this submodel's reactions over ``interval``, and returns the **delta**
    it produced on the settable species. Because the store's ``map[float]`` applies
    numeric updates additively, the deltas of all submodel nodes reconcile by
    summation through the bigraph apply path.
    """

    config_schema = {
        "model": {"_type": "string", "_default": ""},
        "reactions": {"_type": "list[string]", "_default": []},
        "absolute_tolerance": {"_type": "float", "_default": ABSOLUTE_TOLERANCE},
        "relative_tolerance": {"_type": "float", "_default": RELATIVE_TOLERANCE},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self._rr = None
        self._settable = None
        self._init = None

    def inputs(self):
        return {"species": "map[float]"}

    def outputs(self):
        # Plain map[float] (NOT overwrite): the update is a DELTA that applies
        # additively and reconciles by summation across submodel nodes.
        return {"species": "map[float]"}

    def _build(self):
        if self._rr is not None:
            return
        cfg = self.config
        if not cfg["model"]:
            raise ValueError("SubmodelProcess requires a 'model' source.")
        full = _load(cfg["model"])
        base_sbml = full.getCurrentSBML()
        all_reactions = list(full.getReactionIds())
        reactions = list(cfg["reactions"])
        unknown = [r for r in reactions if r not in all_reactions]
        if unknown:
            raise ValueError(f"unknown reactions {unknown} (have {all_reactions})")
        if not reactions:
            raise ValueError("SubmodelProcess needs at least one reaction")
        _, _, self._settable = _species_metadata(full)
        self._init = {s: float(full[s]) for s in self._settable}
        self._rr = _load_sbml_subset(base_sbml, reactions, cfg)

    def update(self, state, interval):
        self._build()
        rr = self._rr
        incoming = (state.get("species") if state else None) or {}
        base = {
            s: float(incoming[s]) if s in incoming else self._init[s]
            for s in self._settable
        }
        for s in self._settable:
            rr[s] = base[s]
        try:
            t0 = rr.getCurrentTime()
            rr.oneStep(t0, interval)
        except RuntimeError:
            # Left the sqrt domain / CVODE failed: write a non-finite delta so the
            # run surfaces as diverged instead of crashing the composite.
            return {"species": {s: float("nan") for s in self._settable}}
        return {"species": {s: float(rr[s]) - base[s] for s in self._settable}}


class EnergyStep(Step):
    """Assignment-rule node: re-derive ATP/ADP/AMP from the reconciled pool.

    Reads the shared species store after the submodel deltas have reconciled and
    writes the delta that lands ``ATP``/``ADP``/``AMP`` on the values the SBML
    assignment rules give for the current settable phosphate pool (``P`` and the
    conserved ``SUM_P``). Only this node writes the assignment-rule species, so
    there is no double counting.
    """

    config_schema = {"model": {"_type": "string", "_default": ""}}

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self._rr = None
        self._assigned = None
        self._settable = None

    def inputs(self):
        return {"species": "map[float]"}

    def outputs(self):
        return {"species": "map[float]"}

    def _build(self):
        if self._rr is not None:
            return
        if not self.config["model"]:
            raise ValueError("EnergyStep requires a 'model' source.")
        self._rr = _load(self.config["model"])
        _, self._assigned, self._settable = _species_metadata(self._rr)

    def update(self, state):
        self._build()
        incoming = (state.get("species") if state else None) or {}
        for s in self._settable:
            if s in incoming:
                self._rr[s] = float(incoming[s])
        # Delta that overwrites each assignment-rule species with its derived value
        # (float apply is additive, so delta = derived - current).
        out = {}
        for a in self._assigned:
            current = float(incoming.get(a, 0.0))
            out[a] = float(self._rr[a]) - current
        return {"species": out}


def _load_sbml_subset(base_sbml, reactions, cfg):
    import tellurium as te

    rr = te.loadSBMLModel(_submodel_sbml(base_sbml, reactions))
    rr.integrator.absolute_tolerance = cfg["absolute_tolerance"]
    rr.integrator.relative_tolerance = cfg["relative_tolerance"]
    return rr


def register_modular_submodels(core=None):
    """Register SubmodelProcess + EnergyStep + the RAM emitter on ``core``."""
    if core is None:
        core = allocate_core()
    core.register_link("SubmodelProcess", SubmodelProcess)
    core.register_link("EnergyStep", EnergyStep)
    core.register_link("ram-emitter", RAMEmitter)
    return core
