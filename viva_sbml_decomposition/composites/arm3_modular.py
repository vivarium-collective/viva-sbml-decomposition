"""Arm 3: glycolysis as a GENUINE multi-process composite across a biological seam.

SAME glycolysis model as arm 1, but partitioned along the textbook **aldolase
seam** (upper vs. lower glycolysis; see
:mod:`viva_sbml_decomposition.models.glycolysis_submodels`) into **separate
process-bigraph nodes** — one :class:`SubmodelProcess` per submodel — that share
a single ``species`` store and couple only through it. This is the real
composability story that separates arm 3 from arm 2's single-process operator
split: here each submodel is its own bigraph node running standalone over its
reaction subset, and the composition is just shared topology.

**How the coupling works (additive deltas through the store).** The shared
``species`` store holds the absolute concentrations as ``map[float]``. In this
bigraph-schema, a plain ``float`` update *is a delta* — ``apply`` adds it to the
current value and ``reconcile`` sums all deltas written to the same path in a
step. So each :class:`SubmodelProcess` reads the shared state, integrates its own
reactions over the coupling interval, and writes back the **delta** it produced
on the settable species. The two submodels' deltas therefore reconcile
**additively** through the bigraph apply path (a Jacobi / additive operator
split) — no custom coupling process, exactly like mosaic-compose's dFBA
co-culture sharing a ``substrates`` store. The composite scheduler advances both
nodes each ``coupling_interval``, and a small :class:`EnergyStep` node re-derives
the assignment-rule species (``ATP``/``ADP``/``AMP``) from the reconciled
phosphate pool each step (the SBML assignment rules, as a composite node).

**Why the coupling interval must be small here.** The seam cuts the strong
phosphate/energy coupling that feeds the sqrt ``ADP`` assignment rule. If a
submodel advances a whole interval in isolation before the store reconciles, the
shared ``P`` pool drifts out of the sqrt domain and CVODE blows up — arm 2 found
exactly this seam intractable for a pure Lie split. Arm 3 keeps it well-posed by
reconciling through the store *frequently* (small ``coupling_interval``), so
neither submodel moves the shared pool far before the other's delta lands. The
residual is genuine first-order additive-splitting error that shrinks with the
coupling interval (task-7 report); it is NOT manufactured. An over-coarse
interval leaves the domain and CVODE fails — :class:`SubmodelProcess` catches
that and writes a non-finite delta, so the arm surfaces as *diverged* rather than
crashing the composite.

Assignment-rule species (``ATP``/``ADP``/``AMP``) cannot be set independently in
roadrunner — they are algebraic functions of the settable pool ``P`` and the
conserved sum ``SUM_P`` — so the submodels only exchange the settable floating
species; :class:`EnergyStep` reconstructs the derived trio from the shared pool.
"""
from __future__ import annotations

import libsbml
from process_bigraph import Process, Step, allocate_core
from process_bigraph.composite_generator import composite_generator
from process_bigraph.emitter import RAMEmitter

from viva_sbml_decomposition.models.glycolysis import GLYCOLYSIS_ANTIMONY, GLYCOLYSIS_META
from viva_sbml_decomposition.models.glycolysis_submodels import SUBMODELS

from ._emit import strip_emitter
from ._run import extract_run

# Default coupling interval = the scheduler step at which the shared store
# reconciles both submodels' deltas. It must stay well inside the sqrt domain of
# the ADP assignment rule (an isolation window << ~0.012 here); the default sits
# in the converged, within-tolerance regime (task-7 report).
DEFAULT_COUPLING_INTERVAL = 0.0005

# Default biological partition (the aldolase seam), as plain reaction-id lists.
DEFAULT_REACTION_GROUPS = [list(group) for _, group in SUBMODELS]

# Matched integrator tolerances (identical to arm 1 / arm 2).
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
            # arm surfaces as diverged instead of crashing the composite.
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


def _initial_species():
    """Absolute initial concentrations for the full species set (seed the store)."""
    full = _load(GLYCOLYSIS_ANTIMONY)
    species_ids, _, _ = _species_metadata(full)
    return {s: float(full[s]) for s in species_ids}


def _build_document(coupling_interval, reaction_groups=None):
    """Build the unstripped arm-3 composite: one SubmodelProcess node per group,
    an EnergyStep, a shared ``species`` store, and a RAM emitter.

    The ``species`` store is the single shared vivarium store all submodel nodes
    reconcile their deltas through each coupling interval. The ``emitter`` node
    aggregates it so :func:`extract_run` can read the trajectory back; its
    ``time`` is wired to ``global_time`` (no process writes time).
    """
    groups = (
        [list(g) for g in reaction_groups]
        if reaction_groups is not None
        else [list(g) for g in DEFAULT_REACTION_GROUPS]
    )
    names = [name for name, _ in SUBMODELS]
    if len(groups) != len(names):
        names = [f"submodel_{i}" for i in range(len(groups))]

    doc = {
        "species": _initial_species(),
        "energy": {
            "_type": "step",
            "address": "local:EnergyStep",
            "config": {"model": GLYCOLYSIS_ANTIMONY},
            "inputs": {"species": ["species"]},
            "outputs": {"species": ["species"]},
        },
        "emitter": {
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
                "species": ["species"],
                "time": ["global_time"],
                "global_time": ["global_time"],
            },
        },
    }
    for name, group in zip(names, groups):
        doc[name] = {
            "_type": "process",
            "address": "local:SubmodelProcess",
            "config": {
                "model": GLYCOLYSIS_ANTIMONY,
                "reactions": group,
                "absolute_tolerance": ABSOLUTE_TOLERANCE,
                "relative_tolerance": RELATIVE_TOLERANCE,
            },
            "interval": coupling_interval,
            "inputs": {"species": ["species"]},
            "outputs": {"species": ["species"]},
        }
    return doc


@composite_generator(
    name="glycolysis_modular",
    description=(
        "Arm 3: the 17-species glycolysis model partitioned along the biological "
        "aldolase seam into separate upper- and lower-glycolysis process nodes "
        "(one SubmodelProcess each) that share a single species store and couple "
        "only through it — each writes its per-interval deltas, which reconcile "
        "additively through the bigraph apply path (same biology as arm 1)."
    ),
    parameters={
        "coupling_interval": {
            "type": "float",
            "default": DEFAULT_COUPLING_INTERVAL,
        },
    },
    core_extensions=[register_modular_submodels],
    fidelity="real",
    methods=["libroadrunner CVODE", "additive delta-coupled process composition"],
)
def glycolysis_modular(core=None, coupling_interval=DEFAULT_COUPLING_INTERVAL):
    """Modular-submodels glycolysis composite (emitter stripped for the workbench)."""
    return strip_emitter(_build_document(coupling_interval))


def run_arm(core=None, coupling_interval=DEFAULT_COUPLING_INTERVAL):
    """Run arm 3 locally through process-bigraph; return ``(traj, t)``.

    Uses the *unstripped* document (library RAM emitter) so the history can be
    read back. ``traj`` is ``{species: ndarray}`` over ``GLYCOLYSIS_META["species"]``,
    ``t`` the aligned 1-D time array (t=0..t_end in steps of ``coupling_interval``).
    """
    if core is None:
        core = allocate_core()
    register_modular_submodels(core)
    doc = _build_document(coupling_interval)
    return extract_run(doc, core, GLYCOLYSIS_META["t_end"], GLYCOLYSIS_META["species"])
