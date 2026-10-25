"""Arm 3: glycolysis partitioned into coupled submodels along a biological seam.

SAME glycolysis model as arm 1, but partitioned along the textbook **aldolase
seam** (upper vs. lower glycolysis; see
:mod:`viva_sbml_decomposition.models.glycolysis_submodels`) into several
TelluriumProcess submodels coupled through a shared species store. Unlike arm 2
(a mechanical operator split that carved off the weakly-coupled storage sinks),
arm 3's seam is biologically meaningful and crosses the strong phosphate/energy
coupling on purpose.

Each submodel is one libroadrunner instance built from the SAME SBML, keeping
only its reaction subset (all species, parameters and assignment rules retained).
The submodels share the boundary metabolites they both write — ``F16P``, the
``P``/``ATP``/``ADP``/``AMP`` energy pool and ``NAD``/``NADH`` — through the
common vivarium ``species`` store, reconciled **additively**: given the SAME
starting state, each submodel advances a small internal sub-step, and the
per-species deltas of all submodels are summed onto the shared state (a Jacobi /
additive operator split, in contrast to arm 2's sequential Gauss-Seidel handoff).

**Why sub-stepping is mandatory here.** The seam cuts the energy coupling that
feeds the sqrt ``ADP`` assignment rule. If a submodel advances a whole coupling
interval in isolation (a pure Lie split, as in arm 2) the shared ``P`` pool
drifts out of the sqrt domain and CVODE blows up for every interval — arm 2
found exactly this seam intractable. Arm 3 advances each coupling interval in
``n_substeps`` small additive steps, so neither submodel moves the shared pool
far before reconciliation; the seam stays well-posed and the trajectory
converges to the arm-1 reference. The residual is genuine first-order
additive-splitting error that shrinks with the sub-step (task-7 report).

Assignment-rule species (``ATP``/``ADP``/``AMP``) cannot be set independently in
roadrunner — they are algebraic functions of the settable pool ``P`` and the
conserved sum ``SUM_P`` — so only the settable floating species are exchanged;
roadrunner re-derives ATP/ADP/AMP from ``P`` internally whenever ``P`` is pushed.
"""
from __future__ import annotations

import libsbml
from process_bigraph import Process, allocate_core
from process_bigraph.composite_generator import composite_generator
from process_bigraph.emitter import RAMEmitter

from viva_sbml_decomposition.models.glycolysis import GLYCOLYSIS_ANTIMONY, GLYCOLYSIS_META
from viva_sbml_decomposition.models.glycolysis_submodels import SUBMODELS

from ._emit import strip_emitter
from ._run import extract_run

# Default coupling interval (one process update) and internal additive sub-steps.
# dt = coupling_interval / n_substeps must stay well inside the sqrt domain of the
# ADP assignment rule (dt << ~0.015 here); the default sits in the converged,
# within-tolerance regime (task-7 report).
DEFAULT_COUPLING_INTERVAL = 0.1
DEFAULT_N_SUBSTEPS = 200

# Default biological partition (the aldolase seam), as plain reaction-id lists.
DEFAULT_GROUP_NAMES = [name for name, _ in SUBMODELS]
DEFAULT_REACTION_GROUPS = [list(group) for _, group in SUBMODELS]

# Matched integrator tolerances (identical to arm 1 / arm 2).
ABSOLUTE_TOLERANCE = 1e-10
RELATIVE_TOLERANCE = 1e-8


class ModularSubmodelsProcess(Process):
    """Additive, sub-stepped coupling of several SBML submodels across a seam.

    Holds one libroadrunner instance per reaction group (all built from the same
    model, each keeping only its group's reactions). Each ``update(interval)``
    advances the interval in ``n_substeps`` additive sub-steps. Per sub-step,
    every submodel starts from the SAME shared state, advances its reactions by
    ``dt = interval / n_substeps``, and its per-species delta on the settable
    floating species is summed onto the shared state (additive reconciliation of
    the shared boundary pool). The full settable species set is exchanged through
    the shared ``species`` store each coupling interval.
    """

    config_schema = {
        "model": {"_type": "string", "_default": ""},
        "model_format": {"_type": "string", "_default": "antimony"},
        "reaction_groups": {"_type": "list[list[string]]", "_default": []},
        "n_substeps": {"_type": "integer", "_default": DEFAULT_N_SUBSTEPS},
        "absolute_tolerance": {"_type": "float", "_default": ABSOLUTE_TOLERANCE},
        "relative_tolerance": {"_type": "float", "_default": RELATIVE_TOLERANCE},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self._rrs = None
        self._species_ids = None
        self._settable = None
        self._assigned = None

    def inputs(self):
        return {"species": "map[float]"}

    def outputs(self):
        return {
            "species": "overwrite[map[float]]",
            "time": "overwrite[float]",
        }

    def _build(self):
        if self._rrs is not None:
            return
        import tellurium as te

        cfg = self.config
        if not cfg["model"]:
            raise ValueError("ModularSubmodelsProcess requires a 'model' source.")
        if int(cfg["n_substeps"]) < 1:
            raise ValueError("n_substeps must be >= 1")

        if cfg["model_format"] == "sbml":
            full = te.loadSBMLModel(cfg["model"])
        else:
            full = te.loada(cfg["model"])
        base_sbml = full.getCurrentSBML()

        all_reactions = list(full.getReactionIds())
        groups = [list(g) for g in cfg["reaction_groups"]]
        if len(groups) < 2:
            raise ValueError(
                f"modular split needs at least two submodels; got {groups}"
            )
        flat = [r for g in groups for r in g]
        unknown = [r for r in flat if r not in all_reactions]
        if unknown:
            raise ValueError(
                f"reaction_groups reference unknown reactions {unknown} "
                f"(have {all_reactions})"
            )
        if len(flat) != len(set(flat)):
            raise ValueError(f"reaction_groups overlap (a reaction in two groups): {flat}")
        if set(flat) != set(all_reactions):
            missing = sorted(set(all_reactions) - set(flat))
            raise ValueError(
                f"reaction_groups must partition every reaction; missing {missing}"
            )
        if any(len(g) == 0 for g in groups):
            raise ValueError("every submodel must own at least one reaction")

        # Assignment-rule species cannot be set independently in roadrunner.
        doc = libsbml.readSBMLFromString(base_sbml)
        model = doc.getModel()
        self._assigned = {
            model.getRule(i).getVariable()
            for i in range(model.getNumRules())
            if model.getRule(i).isAssignment()
        }
        self._species_ids = list(full.getFloatingSpeciesIds())
        self._settable = [s for s in self._species_ids if s not in self._assigned]

        self._rrs = []
        for g in groups:
            rr = te.loadSBMLModel(self._submodel_sbml(base_sbml, g))
            rr.integrator.absolute_tolerance = cfg["absolute_tolerance"]
            rr.integrator.relative_tolerance = cfg["relative_tolerance"]
            self._rrs.append(rr)

    @staticmethod
    def _submodel_sbml(base_sbml, keep_reactions):
        """SBML keeping only ``keep_reactions`` (all species/params/rules kept)."""
        doc = libsbml.readSBMLFromString(base_sbml)
        model = doc.getModel()
        keep = set(keep_reactions)
        for rid in [model.getReaction(i).getId() for i in range(model.getNumReactions())]:
            if rid not in keep:
                model.removeReaction(rid)
        return libsbml.writeSBMLToString(doc)

    def _push_settable(self, rr, values):
        for sid, v in zip(self._settable, values):
            rr[sid] = float(v)

    def _read_full(self, rr):
        return {sid: float(rr[sid]) for sid in self._species_ids}

    def initial_state(self):
        self._build()
        rr0 = self._rrs[0]
        rr0.reset()
        return {"species": self._read_full(rr0), "time": 0.0}

    def update(self, state, interval):
        self._build()
        incoming = (state.get("species") if state else None) or {}
        rr0 = self._rrs[0]

        # Seed the settable pool from the shared store (fall back to rr0 state).
        svals = [
            float(incoming[sid]) if sid in incoming else float(rr0[sid])
            for sid in self._settable
        ]

        n = int(self.config["n_substeps"])
        dt = interval / n
        try:
            for _ in range(n):
                base = list(svals)
                delta = [0.0] * len(self._settable)
                for rr in self._rrs:
                    self._push_settable(rr, base)
                    tcur = rr.getCurrentTime()
                    rr.oneStep(tcur, dt)
                    for i, sid in enumerate(self._settable):
                        delta[i] += float(rr[sid]) - base[i]
                svals = [base[i] + delta[i] for i in range(len(base))]
        except RuntimeError:
            # A submodel left the sqrt domain / CVODE failed: surface as a
            # diverged (non-finite) state instead of crashing the composite.
            nan = float("nan")
            species = {sid: nan for sid in self._species_ids}
            return {"species": species, "time": float("nan")}

        # Materialize the full species set: push settable into rr0 so its
        # assignment rules re-derive ATP/ADP/AMP from the reconciled P/SUM_P.
        self._push_settable(rr0, svals)
        species = self._read_full(rr0)
        return {"species": species, "time": float(rr0.getCurrentTime())}


def register_modular_submodels(core=None):
    """Register ModularSubmodelsProcess + the RAM emitter on ``core``."""
    if core is None:
        core = allocate_core()
    core.register_link("ModularSubmodelsProcess", ModularSubmodelsProcess)
    core.register_link("ram-emitter", RAMEmitter)
    return core


def _build_document(coupling_interval, n_substeps, reaction_groups=None):
    """Build the unstripped arm-3 document (modular process + RAM emitter).

    The ``species`` store is the shared vivarium store the submodels reconcile
    through each coupling interval; the ``emitter`` node aggregates it so
    :func:`extract_run` can read the trajectory back.
    """
    groups = (
        [list(g) for g in reaction_groups]
        if reaction_groups is not None
        else [list(g) for g in DEFAULT_REACTION_GROUPS]
    )
    return {
        "submodels": {
            "_type": "process",
            "address": "local:ModularSubmodelsProcess",
            "config": {
                "model": GLYCOLYSIS_ANTIMONY,
                "model_format": "antimony",
                "reaction_groups": groups,
                "n_substeps": int(n_substeps),
                "absolute_tolerance": ABSOLUTE_TOLERANCE,
                "relative_tolerance": RELATIVE_TOLERANCE,
            },
            "interval": coupling_interval,
            "inputs": {"species": ["stores", "species"]},
            "outputs": {
                "species": ["stores", "species"],
                "time": ["stores", "time"],
            },
        },
        "stores": {},
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
                "species": ["stores", "species"],
                "time": ["stores", "time"],
                "global_time": ["global_time"],
            },
        },
    }


@composite_generator(
    name="glycolysis_modular",
    description=(
        "Arm 3: the 17-species glycolysis model partitioned along the biological "
        "aldolase seam into upper- and lower-glycolysis TelluriumProcess submodels, "
        "reconciling the shared F16P / energy / redox boundary pool additively "
        "through a shared vivarium store each coupling interval (same biology as "
        "arm 1)."
    ),
    parameters={
        "coupling_interval": {
            "type": "float",
            "default": DEFAULT_COUPLING_INTERVAL,
        },
        "n_substeps": {
            "type": "integer",
            "default": DEFAULT_N_SUBSTEPS,
        },
    },
    core_extensions=[register_modular_submodels],
    fidelity="real",
    methods=["libroadrunner CVODE", "additive sub-stepped operator splitting"],
)
def glycolysis_modular(
    core=None,
    coupling_interval=DEFAULT_COUPLING_INTERVAL,
    n_substeps=DEFAULT_N_SUBSTEPS,
):
    """Modular-submodels glycolysis composite (emitter stripped for the workbench)."""
    return strip_emitter(_build_document(coupling_interval, n_substeps))


def run_arm(
    core=None,
    coupling_interval=DEFAULT_COUPLING_INTERVAL,
    n_substeps=DEFAULT_N_SUBSTEPS,
):
    """Run arm 3 locally through process-bigraph; return ``(traj, t)``.

    Uses the *unstripped* document (library RAM emitter) so the history can be
    read back. ``traj`` is ``{species: ndarray}`` over ``GLYCOLYSIS_META["species"]``,
    ``t`` the aligned 1-D time array (t=0..t_end in steps of ``coupling_interval``).
    """
    if core is None:
        core = allocate_core()
    register_modular_submodels(core)
    doc = _build_document(coupling_interval, n_substeps)
    return extract_run(doc, core, GLYCOLYSIS_META["t_end"], GLYCOLYSIS_META["species"])
