"""Arm 6: the COUPLING DEMONSTRATION — a composition the monolith CAN'T express.

This is the existence proof for the composability axis of the 3-way tradeoff. It
takes arm 3's modular decomposition along the aldolase seam (upper vs. lower
glycolysis sharing one ``species`` store, reconciling additively) and swaps ONE
submodel node for a different simulation ENGINE:

* ``upper_glycolysis`` — a Tellurium/libroadrunner
  :class:`~viva_sbml_decomposition.composites.arm3_modular.SubmodelProcess`
  (CVODE on the SBML reaction subset), exactly as in arm 3.
* ``lower_glycolysis`` — a native process-bigraph
  :class:`~viva_sbml_decomposition.processes.native_glycolysis.NativeSubmodelProcess`
  (hand-ported Teusink RHS, scipy LSODA, no libroadrunner at runtime), integrating
  only the lower-glycolysis reactions.

Both nodes read the SAME shared ``species`` store and write per-interval
**deltas** on the settable species; because the store's ``map[float]`` applies
numeric updates additively, the libroadrunner delta and the native-Python delta
reconcile by summation through the bigraph apply path. An
:class:`~viva_sbml_decomposition.composites.arm3_modular.EnergyStep` re-derives
ATP/ADP/AMP from the reconciled phosphate pool each step, exactly as in arm 3.

**Why the monolith can't express this.** A single libroadrunner model is one
integrator over one SBML document; you cannot have CVODE advance half of its
reactions while a hand-written scipy process advances the other half and have the
two meet in a shared state each tick. The composition only exists *because* the
model was decomposed into store-coupled nodes — which is the whole composability
point. Mixing engines changes the numerics, so arm 6 has **no monolithic
reference** and is deliberately kept OFF the accuracy axis (see
:mod:`viva_sbml_decomposition.harness.composability`); the tests assert only that
it builds, runs, and emits finite, sensible output.

**Coupling interval.** The aldolase seam cuts the strong phosphate/energy
coupling feeding the sqrt ADP assignment rule (arm 3's caveat). As in arm 3, the
store must reconcile frequently (small ``coupling_interval``) so neither engine
drifts the shared pool out of the sqrt domain before the other's delta lands.
"""
from __future__ import annotations

from process_bigraph import allocate_core
from process_bigraph.composite_generator import composite_generator
from process_bigraph.emitter import RAMEmitter

from viva_sbml_decomposition.composites.arm3_modular import (
    ABSOLUTE_TOLERANCE,
    RELATIVE_TOLERANCE,
    EnergyStep,
    SubmodelProcess,
    _initial_species,
)
from viva_sbml_decomposition.models.glycolysis import GLYCOLYSIS_ANTIMONY, GLYCOLYSIS_META
from viva_sbml_decomposition.models.glycolysis_submodels import (
    LOWER_GLYCOLYSIS,
    UPPER_GLYCOLYSIS,
)
from viva_sbml_decomposition.processes.native_glycolysis import (
    DEFAULT_METHOD,
    NativeSubmodelProcess,
)

from ._emit import strip_emitter
from ._run import extract_run

# Domain-safe reconciliation interval. The strong phosphate/energy seam (arm 3's
# caveat) needs frequent store reconciliation to stay inside the sqrt ADP-rule
# domain; this is arm 3's "domain-safe, fast" finiteness-check interval. Arm 6 has
# NO accuracy target (mixing engines changes the numerics), so we pick the cheap
# domain-safe interval as the default rather than arm 3's finer accuracy interval.
DEFAULT_COUPLING_INTERVAL = 0.005

# Which half runs on the native engine. Lower glycolysis (the payoff phase) is the
# node swapped for NativeSubmodelProcess; upper glycolysis stays on Tellurium.
NATIVE_SUBMODEL = ("lower_glycolysis", LOWER_GLYCOLYSIS)
TELLURIUM_SUBMODEL = ("upper_glycolysis", UPPER_GLYCOLYSIS)


def register_coupled_submodels(core=None):
    """Register the Tellurium + native submodel nodes, EnergyStep, RAM emitter."""
    if core is None:
        core = allocate_core()
    core.register_link("SubmodelProcess", SubmodelProcess)
    core.register_link("NativeSubmodelProcess", NativeSubmodelProcess)
    core.register_link("EnergyStep", EnergyStep)
    core.register_link("ram-emitter", RAMEmitter)
    return core


def _build_document(coupling_interval):
    """Build the unstripped hybrid composite (both engines + EnergyStep + emitter).

    One shared ``species`` store seeded with the full 17-species initial state; a
    Tellurium ``SubmodelProcess`` node and a native ``NativeSubmodelProcess`` node
    both wired to it (additive deltas); an ``EnergyStep`` re-deriving ATP/ADP/AMP;
    and a RAM emitter so :func:`extract_run` can read the trajectory back.
    """
    tell_name, tell_rxns = TELLURIUM_SUBMODEL
    nat_name, nat_rxns = NATIVE_SUBMODEL
    doc = {
        "species": _initial_species(),
        tell_name: {
            "_type": "process",
            "address": "local:SubmodelProcess",
            "config": {
                "model": GLYCOLYSIS_ANTIMONY,
                "reactions": list(tell_rxns),
                "absolute_tolerance": ABSOLUTE_TOLERANCE,
                "relative_tolerance": RELATIVE_TOLERANCE,
            },
            "interval": coupling_interval,
            "inputs": {"species": ["species"]},
            "outputs": {"species": ["species"]},
        },
        nat_name: {
            "_type": "process",
            "address": "local:NativeSubmodelProcess",
            "config": {
                "reactions": list(nat_rxns),
                "absolute_tolerance": ABSOLUTE_TOLERANCE,
                "relative_tolerance": RELATIVE_TOLERANCE,
                "method": DEFAULT_METHOD,
            },
            "interval": coupling_interval,
            "inputs": {"species": ["species"]},
            "outputs": {"species": ["species"]},
        },
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
    return doc


@composite_generator(
    name="glycolysis_coupled",
    description=(
        "Arm 6 (coupling demonstration): arm 3's modular glycolysis decomposition "
        "with the lower-glycolysis submodel swapped for a NATIVE process-bigraph "
        "process (hand-ported Teusink RHS + scipy) while upper glycolysis stays a "
        "Tellurium/libroadrunner submodel — two different simulation engines sharing "
        "one species store and reconciling additively. A composition a single "
        "monolithic SBML/libroadrunner model structurally cannot express; it has no "
        "monolithic accuracy reference by construction."
    ),
    parameters={
        "coupling_interval": {
            "type": "float",
            "default": DEFAULT_COUPLING_INTERVAL,
        },
    },
    core_extensions=[register_coupled_submodels],
    fidelity="real",
    methods=[
        "libroadrunner CVODE (upper submodel)",
        "native Python/numpy Teusink RHS + scipy LSODA (lower submodel)",
        "additive delta-coupled cross-engine process composition",
    ],
)
def glycolysis_coupled(core=None, coupling_interval=DEFAULT_COUPLING_INTERVAL):
    """Hybrid cross-engine glycolysis composite (emitter stripped for the workbench)."""
    return strip_emitter(_build_document(coupling_interval))


def run_arm(core=None, coupling_interval=DEFAULT_COUPLING_INTERVAL):
    """Run arm 6 locally through process-bigraph; return ``(traj, t)``.

    Uses the *unstripped* document (library RAM emitter) so the history can be
    read back. ``traj`` is ``{species: ndarray}`` over ``GLYCOLYSIS_META["species"]``,
    ``t`` the aligned 1-D time array (t=0..t_end in steps of ``coupling_interval``).
    Arm 6 has no monolithic reference — the output is validated for finiteness and
    sensibility, not matched against arm 1.
    """
    if core is None:
        core = allocate_core()
    register_coupled_submodels(core)
    doc = _build_document(coupling_interval)
    return extract_run(doc, core, GLYCOLYSIS_META["t_end"], GLYCOLYSIS_META["species"])
