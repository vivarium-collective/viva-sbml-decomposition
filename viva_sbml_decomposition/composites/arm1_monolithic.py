"""Arm 1: the monolithic Tellurium composite (accuracy reference) + run_arm."""
from __future__ import annotations

from process_bigraph import allocate_core
from process_bigraph.composite_generator import composite_generator
from viva_tellurium import make_tellurium_document
from viva_tellurium.composites import register_tellurium

from viva_sbml_decomposition.models.glycolysis import GLYCOLYSIS_ANTIMONY, GLYCOLYSIS_META

from ._emit import strip_emitter
from ._run import extract_run

DEFAULT_N_STEPS = 300


def _interval(n_steps: int) -> float:
    return GLYCOLYSIS_META["t_end"] / n_steps


@composite_generator(
    name="glycolysis_monolithic",
    description="Arm 1: whole 17-species glycolysis model in one TelluriumProcess (accuracy reference).",
    default_n_steps=DEFAULT_N_STEPS,
    core_extensions=[register_tellurium],
)
def glycolysis_monolithic(core=None):
    """Whole glycolysis model in one TelluriumProcess (emitter stripped for the workbench)."""
    return strip_emitter(
        make_tellurium_document(
            model=GLYCOLYSIS_ANTIMONY,
            model_format="antimony",
            interval=_interval(DEFAULT_N_STEPS),
        )
    )


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
    doc = make_tellurium_document(
        model=GLYCOLYSIS_ANTIMONY,
        model_format="antimony",
        interval=_interval(n_steps),
        parameter_overrides=parameter_overrides,
    )
    return extract_run(doc, core, GLYCOLYSIS_META["t_end"], GLYCOLYSIS_META["species"])
