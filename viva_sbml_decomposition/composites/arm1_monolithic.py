"""Arm 1: the monolithic Tellurium composite (accuracy reference) + run_arm.

Adopts viva-tellurium's opt-in **bulk-binding** (PR #24): the single
``TelluriumProcess`` runs with ``bulk_species=True`` and its ``species`` port is
a single ordered numpy Array leaf (``array[(n),float]``, ``n`` =
``len(getFloatingSpeciesIds)``) pushed/pulled through roadrunner's bulk C API
(``get/setFloatingSpeciesConcentrations``) instead of a per-species
string-keyed ``map[float]`` loop. This is **bit-identical** to the per-species
read (verified: max |Δ| = 0.0 vs. the map[float] arm 1) and removes the N
string-keyed FFI reads per step, which is arm 1's dominant framework overhead.

The ``species`` store therefore holds an array, so the RAM emitter emits it as
``array[(n),float]`` and ``run_arm`` slices species out of it by index order
(``species_order`` = ``getFloatingSpeciesIds``) rather than by key.
"""
from __future__ import annotations

from functools import lru_cache

from process_bigraph import allocate_core
from process_bigraph.composite_generator import composite_generator
from viva_tellurium import TelluriumProcess
from viva_tellurium.composites import register_tellurium, seed_initial_stores

from viva_sbml_decomposition.models.glycolysis import GLYCOLYSIS_ANTIMONY, GLYCOLYSIS_META

from ._emit import strip_emitter
from ._run import extract_run

DEFAULT_N_STEPS = 300


def _interval(n_steps: int) -> float:
    return GLYCOLYSIS_META["t_end"] / n_steps


@lru_cache(maxsize=1)
def _species_order() -> tuple[str, ...]:
    """Floating-species id order of the bulk ``species`` array (getFloatingSpeciesIds)."""
    core = allocate_core()
    register_tellurium(core)
    proc = TelluriumProcess(
        config={"model": GLYCOLYSIS_ANTIMONY, "model_format": "antimony",
                "bulk_species": True},
        core=core)
    return tuple(proc.get_species_ids())


def _build_document(interval: float, parameter_overrides=None) -> dict:
    """Build the unstripped arm-1 document: one bulk-bound TelluriumProcess + RAM emitter.

    ``bulk_species=True`` makes the process ``species`` port a single
    ``array[(n),float]`` leaf (roadrunner's bulk C API) instead of a
    ``map[float]`` loop. The leaf is wired into the shared ``stores.species``
    store and emitted as an array; :func:`run_arm` reads it back by index order.
    """
    species_order = _species_order()
    n = len(species_order)
    doc = {
        "tellurium": {
            "_type": "process",
            "address": "local:TelluriumProcess",
            "config": {
                "model": GLYCOLYSIS_ANTIMONY,
                "model_format": "antimony",
                "integrator": "cvode",
                "absolute_tolerance": 1e-10,
                "relative_tolerance": 1e-8,
                "seed": -1,
                "species_overrides": {},
                "parameter_overrides": parameter_overrides or {},
                "bulk_species": True,
            },
            "interval": interval,
            # inputs left empty (as the map[float] arm 1 was): the single process
            # has no upstream writer, so it only emits — keeping the run
            # bit-identical while the bulk GETTER removes the per-species reads.
            "inputs": {},
            "outputs": {
                "species": ["stores", "species"],
                "rates": ["stores", "rates"],
                "parameters": ["stores", "parameters"],
                "time": ["stores", "time"],
            },
        },
        "stores": {},
        "emitter": {
            "_type": "step",
            "address": "local:ram-emitter",
            "config": {
                "emit": {
                    "species": f"array[({n}),float]",
                    "rates": "map[float]",
                    "parameters": "map[float]",
                    "time": "float",
                    "global_time": "float",
                },
            },
            "inputs": {
                "species": ["stores", "species"],
                "rates": ["stores", "rates"],
                "parameters": ["stores", "parameters"],
                "time": ["stores", "time"],
                "global_time": ["global_time"],
            },
        },
    }
    # Seed the wired stores with the model's t=0 state (issue #7), as
    # make_tellurium_document does, using a minimal Tellurium-only core.
    seed_core = allocate_core()
    register_tellurium(seed_core)
    seed_initial_stores(doc, seed_core)
    return doc


@composite_generator(
    name="glycolysis_monolithic",
    description="Arm 1: whole 17-species glycolysis model in one TelluriumProcess (accuracy reference, bulk-bound species port).",
    default_n_steps=DEFAULT_N_STEPS,
    core_extensions=[register_tellurium],
)
def glycolysis_monolithic(core=None):
    """Whole glycolysis model in one TelluriumProcess (emitter stripped for the workbench)."""
    return strip_emitter(_build_document(_interval(DEFAULT_N_STEPS)))


def run_arm(core=None, n_steps: int = DEFAULT_N_STEPS, parameter_overrides=None):
    """Run arm 1 locally through process-bigraph; return ``(traj, t)``.

    Uses the *unstripped* document (library RAM emitter) so the history can be
    read back. ``traj`` is ``{species: ndarray}`` over ``GLYCOLYSIS_META["species"]``,
    ``t`` the aligned 1-D time array (n_steps + 1 points, t=0..t_end).

    ``parameter_overrides`` (optional ``{param_id: value}``) perturbs the model's
    rate constants for this instance; ``None`` reproduces the accuracy reference.
    """
    if core is None:
        core = allocate_core()
    register_tellurium(core)
    doc = _build_document(_interval(n_steps), parameter_overrides=parameter_overrides)
    return extract_run(
        doc, core, GLYCOLYSIS_META["t_end"], GLYCOLYSIS_META["species"],
        species_order=list(_species_order()))
