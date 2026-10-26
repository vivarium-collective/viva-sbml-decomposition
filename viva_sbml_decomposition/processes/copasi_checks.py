"""Process-level COPASI integrity check: the bulk-SET fast path is bit-identical.

``CopasiProcess`` (``processes/copasi_glycolysis.py``) injects the shared-store state
into COPASI each interval through a raw-handle bulk SET (``setInitialConcentration``
+ ``updateInitialValues``) instead of basico's per-name ``set_species``. This module
verifies, on the full Teusink model, that the two paths write the same state and give
the same post-step trajectory. It loads COPASI, so callers go through
:func:`copasi_set_bit_identity` (a retrying subprocess that survives COPASI's
intermittent loader abort).
"""
from __future__ import annotations

from viva_sbml_decomposition.composites._engine import run_in_subprocess_with_retry
from viva_sbml_decomposition.models.glycolysis import GLYCOLYSIS_ANTIMONY, GLYCOLYSIS_META


def _set_path_equivalence_inproc(perturb: float = 1.1):
    """Compare the raw-handle bulk SET against basico's per-name set_species.

    On the full Teusink model: perturb the settable species by ``perturb``,
    inject that state TWO ways — (A) the hot-loop raw path
    (``CMetab.setInitialConcentration`` + ``updateInitialValues``) and (B) basico's
    per-name ``set_species(exact=True, initial_concentration=...)`` — integrate one
    interval each (``use_initial_values=True``), and also run an UNSET baseline
    (model's own initial values). Returns max |Δ| over all 17 species for
    ``raw_vs_basico`` (should be ~0, bit-identical) and ``raw_vs_baseline``
    (should be clearly non-zero, so a silent no-op SET can't masquerade as a pass).
    In-process (loads COPASI), so callers invoke it through the retrying subprocess.
    """
    import COPASI
    import basico

    from viva_sbml_decomposition.processes.copasi_glycolysis import (
        _antimony_to_sbml, _ensure_cps)

    base = _antimony_to_sbml(GLYCOLYSIS_ANTIMONY, "antimony")
    dm = basico.load_model(_ensure_cps(base))
    basico.set_current_model(dm)
    sp = basico.get_species(model=dm)
    id2name = {str(r.sbml_id): n for n, r in sp.iterrows()}
    species = GLYCOLYSIS_META["species"]
    settable = [s for s in species if sp.loc[id2name[s]].type != "assignment"]
    model = dm.getModel()
    metabs = {model.getMetabolite(i).getSBMLId(): model.getMetabolite(i)
              for i in range(model.getNumMetabs())}
    fw = COPASI.CCore.Framework_Concentration
    interval = 0.1
    initial = {s: float(metabs[s].getConcentration()) for s in settable}
    perturbed = {s: initial[s] * perturb for s in settable}

    def _init_conc():
        return {s: float(metabs[s].getInitialConcentration()) for s in settable}

    def _step_read():
        basico.run_time_course(duration=interval, intervals=1,
                               use_initial_values=True, model=dm)
        return {s: float(metabs[s].getConcentration()) for s in species}

    # Baseline: UNSET (integrate from the model's own initial values).
    for s in settable:
        metabs[s].setInitialConcentration(initial[s])
    model.updateInitialValues(fw)
    baseline = _step_read()
    # Path A: raw-handle bulk SET of the perturbed state (the hot-loop path).
    for s in settable:
        metabs[s].setInitialConcentration(perturbed[s])
    model.updateInitialValues(fw)
    raw_init = _init_conc()
    raw = _step_read()
    # Path B: basico per-name set_species of the same perturbed state.
    for s in settable:
        basico.set_species(name=id2name[s], exact=True,
                           initial_concentration=perturbed[s], model=dm)
    basico_init = _init_conc()
    via_basico = _step_read()

    result = {
        # SET-level: the two paths write the SAME initial concentration (the pure
        # bit-identity guarantee — no integration, so ~machine epsilon).
        "set_level_max_abs_diff": max(abs(raw_init[s] - basico_init[s]) for s in settable),
        # Integration-level: same post-step trajectory (CVODE at rtol 1e-8 amplifies
        # the machine-eps SET difference to ~1e-10 — still far below rtol).
        "post_step_max_abs_diff": max(abs(raw[s] - via_basico[s]) for s in species),
        # Not a silent no-op: the perturbed SET visibly moved the result.
        "raw_vs_baseline": max(abs(raw[s] - baseline[s]) for s in species),
    }
    # ``(result, None)``: the shape ``run_in_subprocess_with_retry`` expects.
    return result, None


def copasi_set_bit_identity() -> dict:
    """Robust (subprocess) accessor for :func:`_set_path_equivalence_inproc`.

    Returns ``{set_level_max_abs_diff, post_step_max_abs_diff, raw_vs_baseline}``.
    """
    data = run_in_subprocess_with_retry(
        "viva_sbml_decomposition.processes.copasi_checks", "_set_path_equivalence_inproc", {})
    return data["traj"]
