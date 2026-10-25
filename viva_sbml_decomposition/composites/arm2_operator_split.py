"""Arm 2: the operator-split Tellurium composite.

SAME glycolysis model as arm 1, but integrated as a **Lie operator split**.
The reaction network is partitioned into two complementary groups that each
advance the *full* shared species state over one ``coupling_interval`` and then
compose:

* **operator A** — the core glycolytic + fermentation network (every reaction
  except the two storage sinks);
* **operator B** — the glycogen (``vGLYCO``) and trehalose (``vTreha``) storage
  sinks that branch off G6P.

Each vivarium step: operator A reads the shared ``species`` store, advances by
``coupling_interval``, hands its result to operator B, which advances the same
interval and writes the composed state back to the shared store. The error
introduced by freezing each group's reactions while the other advances is a
genuine first-order operator-splitting error that shrinks with
``coupling_interval``. The biology is identical to arm 1 — only the mechanical
decoupling differs — so this arm isolates vivarium orchestration overhead plus
operator-splitting error.

Why this particular split: a reaction partition that cuts the phosphate/energy
(P / ATP / ADP) coupling makes each isolated sub-model ill-posed — the P pool
runs out of the sqrt domain of the ADP assignment rule and CVODE blows up
regardless of how small the interval is (no convergence). Splitting off only
the two weakly-coupled G6P storage sinks keeps both sub-models well-posed and
yields clean, monotone, first-order convergence (verified in task-6-report).
"""
from __future__ import annotations

import libsbml
from process_bigraph import Process, allocate_core
from process_bigraph.composite_generator import composite_generator
from process_bigraph.emitter import RAMEmitter

from viva_sbml_decomposition.models.glycolysis import GLYCOLYSIS_ANTIMONY, GLYCOLYSIS_META

from ._emit import strip_emitter
from ._run import extract_run

# Reactions split OFF into operator B (the glycogen + trehalose storage sinks
# branching off G6P). Everything else forms operator A. See module docstring.
DEFAULT_SPLIT_B = ["vGLYCO", "vTreha"]

# Default coupling interval (in the stable regime; the sweepable knob).
DEFAULT_COUPLING_INTERVAL = 0.05

# Matched integrator tolerances (identical to arm 1 / TelluriumProcess defaults
# — "matched tolerances per Global Constraints").
ABSOLUTE_TOLERANCE = 1e-10
RELATIVE_TOLERANCE = 1e-8


class OperatorSplitProcess(Process):
    """Lie operator split of one SBML/Antimony model across two reaction groups.

    Holds two libroadrunner instances built from the SAME model: operator A
    runs every reaction NOT in ``reactions_b``; operator B runs exactly the
    reactions in ``reactions_b``. Both carry all species (and boundary
    conditions). Each ``update(interval)`` performs one Lie split step:

        push shared species -> A;  advance A by ``interval``  -> y*
        push y* -> B;               advance B by ``interval``  -> y_{n+1}

    and returns ``y_{n+1}`` as the composed species state. The full species set
    is exchanged through the shared store each coupling interval (the ``species``
    input/output ports both wire to it), so the composition is genuinely through
    process-bigraph stores.

    Assignment-rule species (ATP/ADP/AMP here) cannot be set independently in
    roadrunner; they are algebraic functions of the settable species (P, SUM_P)
    and are re-derived after each push, so only settable species are exchanged.
    """

    config_schema = {
        "model": {"_type": "string", "_default": ""},
        "model_format": {"_type": "string", "_default": "antimony"},
        "reactions_b": {"_type": "list[string]", "_default": []},
        "absolute_tolerance": {"_type": "float", "_default": ABSOLUTE_TOLERANCE},
        "relative_tolerance": {"_type": "float", "_default": RELATIVE_TOLERANCE},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self._rr_a = None
        self._rr_b = None
        self._species_ids = None
        self._settable = None

    def inputs(self):
        return {"species": "map[float]"}

    def outputs(self):
        return {
            "species": "overwrite[map[float]]",
            "time": "overwrite[float]",
        }

    def _build(self):
        if self._rr_a is not None:
            return
        import tellurium as te

        cfg = self.config
        if not cfg["model"]:
            raise ValueError("OperatorSplitProcess requires a 'model' source.")

        if cfg["model_format"] == "sbml":
            full = te.loadSBMLModel(cfg["model"])
        else:
            full = te.loada(cfg["model"])
        base_sbml = full.getCurrentSBML()

        all_reactions = list(full.getReactionIds())
        group_b = list(cfg["reactions_b"])
        unknown = [r for r in group_b if r not in all_reactions]
        if unknown:
            raise ValueError(
                f"reactions_b {unknown} are not reactions of the model "
                f"(have {all_reactions})"
            )
        if not group_b or len(group_b) >= len(all_reactions):
            raise ValueError(
                "operator split needs a proper non-empty subset in reactions_b; "
                f"got {group_b} against {all_reactions}"
            )
        group_a = [r for r in all_reactions if r not in set(group_b)]

        # Species governed by assignment rules cannot be set independently.
        doc = libsbml.readSBMLFromString(base_sbml)
        model = doc.getModel()
        assigned = {
            model.getRule(i).getVariable()
            for i in range(model.getNumRules())
            if model.getRule(i).isAssignment()
        }
        self._species_ids = list(full.getFloatingSpeciesIds())
        self._settable = [s for s in self._species_ids if s not in assigned]

        self._rr_a = te.loadSBMLModel(self._submodel_sbml(base_sbml, group_a))
        self._rr_b = te.loadSBMLModel(self._submodel_sbml(base_sbml, group_b))
        for rr in (self._rr_a, self._rr_b):
            rr.integrator.absolute_tolerance = cfg["absolute_tolerance"]
            rr.integrator.relative_tolerance = cfg["relative_tolerance"]

    @staticmethod
    def _submodel_sbml(base_sbml, keep_reactions):
        """Return SBML for a sub-model keeping only ``keep_reactions``.

        All species, parameters, assignment rules and boundary conditions are
        retained; only the complementary reactions are removed.
        """
        doc = libsbml.readSBMLFromString(base_sbml)
        model = doc.getModel()
        keep = set(keep_reactions)
        for rid in [model.getReaction(i).getId() for i in range(model.getNumReactions())]:
            if rid not in keep:
                model.removeReaction(rid)
        return libsbml.writeSBMLToString(doc)

    def _push(self, rr, state):
        for sid in self._settable:
            if sid in state:
                rr[sid] = float(state[sid])

    def _read(self, rr):
        return {sid: float(rr[sid]) for sid in self._species_ids}

    def initial_state(self):
        self._build()
        self._rr_a.reset()
        return {"species": self._read(self._rr_a), "time": 0.0}

    def update(self, state, interval):
        self._build()
        incoming = state.get("species") if state else None

        # Operator A: advance the core network over the interval.
        if incoming:
            self._push(self._rr_a, incoming)
        t0 = self._rr_a.getCurrentTime()
        self._rr_a.simulate(t0, t0 + interval, 2)
        y_star = self._read(self._rr_a)

        # Operator B: advance the storage sinks from A's result over the interval.
        self._push(self._rr_b, y_star)
        t0 = self._rr_b.getCurrentTime()
        self._rr_b.simulate(t0, t0 + interval, 2)
        composed = self._read(self._rr_b)

        return {"species": composed, "time": float(self._rr_b.getCurrentTime())}


def register_operator_split(core=None):
    """Register OperatorSplitProcess + the RAM emitter on ``core``."""
    if core is None:
        core = allocate_core()
    core.register_link("OperatorSplitProcess", OperatorSplitProcess)
    core.register_link("ram-emitter", RAMEmitter)
    return core


def _build_document(coupling_interval, reactions_b=None):
    """Build the unstripped arm-2 document (operator-split process + RAM emitter).

    The ``species`` store is the shared vivarium store the two operators exchange
    through each coupling interval; the emitter node named ``emitter`` aggregates
    it so :func:`extract_run` can read the trajectory back.
    """
    reactions_b = list(reactions_b) if reactions_b is not None else list(DEFAULT_SPLIT_B)
    return {
        "operator_split": {
            "_type": "process",
            "address": "local:OperatorSplitProcess",
            "config": {
                "model": GLYCOLYSIS_ANTIMONY,
                "model_format": "antimony",
                "reactions_b": reactions_b,
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
    name="glycolysis_operator_split",
    description=(
        "Arm 2: the 17-species glycolysis model integrated as a Lie operator "
        "split — core glycolytic/fermentation network vs. the glycogen/trehalose "
        "storage sinks — exchanging the full species state through a shared "
        "vivarium store each coupling interval (same biology as arm 1)."
    ),
    parameters={
        "coupling_interval": {
            "type": "float",
            "default": DEFAULT_COUPLING_INTERVAL,
        },
    },
    core_extensions=[register_operator_split],
    fidelity="real",
    methods=["libroadrunner CVODE", "Lie operator splitting"],
)
def glycolysis_operator_split(core=None, coupling_interval=DEFAULT_COUPLING_INTERVAL):
    """Operator-split glycolysis composite (emitter stripped for the workbench)."""
    return strip_emitter(_build_document(coupling_interval))


def run_arm(core=None, coupling_interval=DEFAULT_COUPLING_INTERVAL):
    """Run arm 2 locally through process-bigraph; return ``(traj, t)``.

    Uses the *unstripped* document (library RAM emitter) so the history can be
    read back. ``traj`` is ``{species: ndarray}`` over ``GLYCOLYSIS_META["species"]``,
    ``t`` the aligned 1-D time array (t=0..t_end in steps of ``coupling_interval``).
    """
    if core is None:
        core = allocate_core()
    register_operator_split(core)
    doc = _build_document(coupling_interval)
    return extract_run(doc, core, GLYCOLYSIS_META["t_end"], GLYCOLYSIS_META["species"])
