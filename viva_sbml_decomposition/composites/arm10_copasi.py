"""Arm 10: COPASI as a SECOND SBML engine, and a cross-engine coupling composite.

Two composites, both built on :class:`~viva_sbml_decomposition.processes.
copasi_glycolysis.CopasiProcess`:

* :func:`glycolysis_copasi_mono` — the whole 17-species Teusink model run by
  **COPASI/basico** (LSODA) through a single process node, the COPASI analogue
  of arm 1's monolithic Tellurium run. It is the second-engine accuracy point:
  validated against arm 1's libroadrunner/CVODE reference (see
  :func:`copasi_mono_accuracy`; measured max-relative-error ~1.29e-7).

* :func:`glycolysis_cross_engine` — a genuine **two-engine composition**: the
  arm-3 aldolase-seam decomposition with upper glycolysis integrated by a
  Tellurium/libroadrunner :class:`~viva_sbml_decomposition.composites.
  arm3_modular.SubmodelProcess` and lower glycolysis by the COPASI process, BOTH
  sharing one ``species`` store and reconciling additive deltas each coupling
  interval, with an :class:`~viva_sbml_decomposition.composites.arm3_modular.
  EnergyStep` re-deriving ATP/ADP/AMP. This is arm 6's cross-engine demonstration
  with COPASI swapped in for the native engine: two *different SBML engines*
  interoperate through the pbg typed store and converge to the reference.

A measurement helper (:func:`measure_copasi`) reports COPASI-mono accuracy +
timing and the cross-engine accuracy; :func:`copasi_gil_factor` measures whether
basico/COPASI releases the GIL on its solve (serial vs. ThreadPool, like arm 5's
libroadrunner check).
"""
from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Optional

from process_bigraph import allocate_core
from process_bigraph.composite_generator import composite_generator
from process_bigraph.emitter import RAMEmitter

from viva_sbml_decomposition.composites.arm1_monolithic import run_arm as run_monolithic
from viva_sbml_decomposition.composites.arm3_modular import (
    ABSOLUTE_TOLERANCE,
    RELATIVE_TOLERANCE,
    EnergyStep,
    SubmodelProcess,
    _initial_species,
)
from viva_sbml_decomposition.harness.accuracy import AccuracyResult, compare_to_reference
from viva_sbml_decomposition.harness.timing import time_run
from viva_sbml_decomposition.models.glycolysis import GLYCOLYSIS_ANTIMONY, GLYCOLYSIS_META
from viva_sbml_decomposition.models.glycolysis_submodels import (
    LOWER_GLYCOLYSIS,
    UPPER_GLYCOLYSIS,
)
from viva_sbml_decomposition.processes.copasi_glycolysis import (
    ABSOLUTE_TOLERANCE as COPASI_ATOL,
    RELATIVE_TOLERANCE as COPASI_RTOL,
    CopasiProcess,
    _antimony_to_sbml,
    _get_or_build_model,
    _subset_sbml,
    register_copasi,
)

from ._emit import strip_emitter
from ._run import extract_run

DEFAULT_N_STEPS = 300

# Cross-engine reconciliation interval. Same aldolase seam as arm 3/arm 6, which
# cuts the strong phosphate/energy coupling feeding the sqrt ADP assignment rule;
# the store must reconcile frequently so neither engine drifts the shared pool
# out of the sqrt domain before the other's delta lands (arm 6's default).
DEFAULT_COUPLING_INTERVAL = 0.005

# Which half runs on COPASI. Lower glycolysis (the payoff phase) is the COPASI
# node; upper glycolysis stays on Tellurium — mirroring arm 6's split so the two
# are directly comparable (arm 6: Tellurium + native; arm 10: Tellurium + COPASI).
COPASI_SUBMODEL = ("lower_glycolysis", LOWER_GLYCOLYSIS)
TELLURIUM_SUBMODEL = ("upper_glycolysis", UPPER_GLYCOLYSIS)


def _mono_interval(n_steps: int) -> float:
    return GLYCOLYSIS_META["t_end"] / n_steps


# --------------------------------------------------------------------------- #
# Subprocess execution (isolates COPASI's intermittent loader abort).         #
#                                                                             #
# COPASI's model load (SBML import AND the native .cps loadModel) aborts the  #
# process NONDETERMINISTICALLY on this python-copasi build (~1-in-6 across     #
# composites in one long-lived process; an uncatchable C++ ``abort()``).      #
# Caching, load ordering and warming all REDUCE but do not ELIMINATE it. Since #
# a SIGABRT can't be caught in-process, the public run_* functions execute the #
# real COPASI composite in a fresh subprocess and RETRY on a crash: an abort   #
# kills only the throwaway subprocess, and the correct trajectory comes back   #
# by pickle. These runs happen once per call (not per step), so the subprocess #
# startup is amortized. The actual integration still runs real COPASI/basico   #
# (and, for the cross-engine, real libroadrunner) — nothing is stubbed.        #
# --------------------------------------------------------------------------- #

_SUBPROC_SRC = (
    "import sys, json, pickle\n"
    "payload = json.loads(sys.argv[1]); out = sys.argv[2]\n"
    "from viva_sbml_decomposition.composites import arm10_copasi as m\n"
    "kind = payload['kind']\n"
    "if kind == 'mono':\n"
    "    traj, t = m._run_copasi_mono_inproc(n_steps=payload['n_steps'])\n"
    "    data = {'traj': {k: v for k, v in traj.items()}, 't': t}\n"
    "elif kind == 'cross':\n"
    "    traj, t = m._run_cross_engine_inproc(coupling_interval=payload['coupling_interval'])\n"
    "    data = {'traj': {k: v for k, v in traj.items()}, 't': t}\n"
    "elif kind == 'setcheck':\n"
    "    data = {'setcheck': m._verify_set_path_equivalence_inproc()}\n"
    "else:\n"
    "    raise ValueError('unknown kind ' + repr(kind))\n"
    "with open(out, 'wb') as f:\n"
    "    pickle.dump(data, f)\n"
)


def _subprocess_with_retry(payload_dict, *, retries=6):
    """Run ``_SUBPROC_SRC`` with ``payload_dict`` in a retrying subprocess.

    Returns the pickled ``data`` dict. Retries ONLY a signal kill (negative
    returncode: SIGABRT=-6 / SIGSEGV=-11 — COPASI's intermittent loader abort); a
    deterministic Python error (positive returncode, a traceback) fails FAST so a
    bad config isn't retried 6x (~85s x 6 for cross@0.001).
    """
    payload = json.dumps(payload_dict)
    kind = payload_dict.get("kind")
    fd, out_path = tempfile.mkstemp(suffix=".pkl")
    os.close(fd)
    try:
        last = None
        for _ in range(retries):
            proc = subprocess.run(
                [sys.executable, "-c", _SUBPROC_SRC, payload, out_path],
                capture_output=True, text=True,
            )
            if proc.returncode == 0 and os.path.getsize(out_path) > 0:
                with open(out_path, "rb") as fh:
                    return pickle.load(fh)
            last = proc
            if proc.returncode is not None and proc.returncode > 0:
                raise RuntimeError(
                    f"COPASI subprocess {kind!r} failed with a Python error "
                    f"(returncode={proc.returncode}); not a signal abort, so not "
                    f"retried.\nstderr tail:\n{(proc.stderr or '')[-1500:]}"
                )
        raise RuntimeError(
            f"COPASI subprocess {kind!r} was signal-aborted in all {retries} "
            f"attempts (last returncode={getattr(last, 'returncode', '?')}, i.e. "
            f"SIGABRT/SIGSEGV from COPASI's loader); "
            f"stderr tail: {(getattr(last, 'stderr', '') or '')[-800:]}"
        )
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)


def _run_in_subprocess_with_retry(kind, *, n_steps=None, coupling_interval=None,
                                  retries=6):
    """Run the COPASI composite ``kind`` in a retrying subprocess; return (traj, t)."""
    data = _subprocess_with_retry(
        {"kind": kind, "n_steps": n_steps, "coupling_interval": coupling_interval},
        retries=retries)
    return data["traj"], data["t"]


_WARMED = False


def warm_copasi_models():
    """Load BOTH arm-10 COPASI models (full + lower subset) into the cache up front.

    OPTIONAL helper for callers that run MANY COPASI composites in ONE long-lived
    process (e.g. the GIL benchmark). COPASI's model load aborts the process
    *nondeterministically*, and loading every model once up front — before any
    ``run_time_course`` — concentrates the (small, fixed) number of load events
    and reduces the chance a load lands mid-session. It is deliberately NOT called
    from the composite BUILD/registration path (building the composite doc and
    registering the process must never trigger an in-process model load, so
    dashboard rendering/building can't abort), nor from the subprocess run path
    (each subprocess runs ONE composite, so it loads only the one model it needs).
    Idempotent and cheap after the first call.
    """
    global _WARMED
    if _WARMED:
        return
    base = _antimony_to_sbml(GLYCOLYSIS_ANTIMONY, "antimony")
    _get_or_build_model(base, [], COPASI_RTOL, COPASI_ATOL)                       # mono (full)
    _, cop_rxns = COPASI_SUBMODEL
    sub = _subset_sbml(base, list(cop_rxns))
    _get_or_build_model(sub, list(cop_rxns), COPASI_RTOL, COPASI_ATOL)            # cross (subset)
    _WARMED = True


# --------------------------------------------------------------------------- #
# COPASI-mono: the whole model on a second engine.                            #
# --------------------------------------------------------------------------- #

def _build_mono_document(interval: float) -> dict:
    """Unstripped COPASI-mono document: one CopasiProcess (overwrite) + RAM emitter.

    A single engine writes the absolute 17-species state to the shared
    ``stores.species`` store each interval (``overwrite`` semantics) plus the
    current time; the RAM emitter aggregates both so :func:`extract_run` can read
    the trajectory back (``map[float]`` species, like arm 2).
    """
    return {
        "copasi": {
            "_type": "process",
            "address": "local:CopasiProcess",
            "config": {
                "model": GLYCOLYSIS_ANTIMONY,
                "model_format": "antimony",
                "reactions": [],            # full model
                "coupling_mode": "overwrite",
                "absolute_tolerance": ABSOLUTE_TOLERANCE,
                "relative_tolerance": RELATIVE_TOLERANCE,
            },
            "interval": interval,
            # Read back the prior step's state and overwrite it: injecting the
            # previous absolute state and integrating one interval is equivalent
            # to a continuous integration, and keeps each step self-contained so
            # the cached COPASI model is reusable run-to-run.
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
    name="glycolysis_copasi_mono",
    description=(
        "Arm 10: the whole 17-species Teusink glycolysis model run by COPASI/basico "
        "(LSODA) in one process node — COPASI as a SECOND SBML engine on the same "
        "model, validated against arm 1's libroadrunner/CVODE reference (two "
        "different integrators agree to max_rel_err ~1.3e-7). NOTE: building this "
        "doc is abort-safe (no COPASI load), but an in-process RUN loads a COPASI "
        "model, whose loader aborts the process nondeterministically (~1-in-6) on "
        "the current python-copasi build; the measurement path runs it in a "
        "retrying subprocess to avoid this."
    ),
    default_n_steps=DEFAULT_N_STEPS,
    core_extensions=[register_copasi],
    fidelity="real",
    methods=["COPASI/basico LSODA"],
)
def glycolysis_copasi_mono(core=None):
    """Whole glycolysis model in one CopasiProcess (emitter stripped for the workbench)."""
    return strip_emitter(_build_mono_document(_mono_interval(DEFAULT_N_STEPS)))


def _run_copasi_mono_inproc(core=None, n_steps: int = DEFAULT_N_STEPS):
    """In-process COPASI-mono run (may hit COPASI's flaky loader abort)."""
    if core is None:
        core = allocate_core()
    register_copasi(core)
    doc = _build_mono_document(_mono_interval(n_steps))
    return extract_run(doc, core, GLYCOLYSIS_META["t_end"], GLYCOLYSIS_META["species"])


def run_copasi_mono(core=None, n_steps: int = DEFAULT_N_STEPS):
    """Run COPASI-mono through process-bigraph; return ``(traj, t)``.

    Executed in a retrying subprocess (see :func:`_run_in_subprocess_with_retry`)
    so COPASI's intermittent loader abort can never kill the caller.
    """
    return _run_in_subprocess_with_retry("mono", n_steps=n_steps)


# --------------------------------------------------------------------------- #
# Cross-engine: Tellurium + COPASI sharing one store.                         #
# --------------------------------------------------------------------------- #

def register_cross_engine(core=None):
    """Register the Tellurium submodel, COPASI submodel, EnergyStep, RAM emitter."""
    if core is None:
        core = allocate_core()
    core.register_link("SubmodelProcess", SubmodelProcess)   # Tellurium/libroadrunner
    core.register_link("CopasiProcess", CopasiProcess)        # COPASI/basico
    core.register_link("EnergyStep", EnergyStep)
    core.register_link("ram-emitter", RAMEmitter)
    return core


def _build_cross_document(coupling_interval: float) -> dict:
    """Unstripped cross-engine document: one Tellurium node + one COPASI node.

    One shared ``species`` store seeded with the full 17-species initial state; a
    Tellurium ``SubmodelProcess`` integrating the upper-glycolysis reactions and a
    ``CopasiProcess`` (``coupling_mode="delta"``) integrating the lower-glycolysis
    reactions, both writing additive deltas to the store; an ``EnergyStep``
    re-deriving ATP/ADP/AMP; and a RAM emitter.
    """
    tell_name, tell_rxns = TELLURIUM_SUBMODEL
    cop_name, cop_rxns = COPASI_SUBMODEL
    return {
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
        cop_name: {
            "_type": "process",
            "address": "local:CopasiProcess",
            "config": {
                "model": GLYCOLYSIS_ANTIMONY,
                "model_format": "antimony",
                "reactions": list(cop_rxns),
                "coupling_mode": "delta",
                "absolute_tolerance": ABSOLUTE_TOLERANCE,
                "relative_tolerance": RELATIVE_TOLERANCE,
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


@composite_generator(
    name="glycolysis_cross_engine",
    description=(
        "Arm 10 (cross-engine coupling): the arm-3 aldolase-seam decomposition with "
        "upper glycolysis integrated by a Tellurium/libroadrunner process and lower "
        "glycolysis by a COPASI/basico process, BOTH sharing one species store and "
        "reconciling additive deltas each coupling interval. Two different SBML "
        "engines interoperate through the pbg typed store and converge to the "
        "libroadrunner reference — a composition a single monolithic engine cannot "
        "express. NOTE: building this doc is abort-safe (no COPASI load), but an "
        "in-process RUN loads a COPASI model whose loader aborts the process "
        "nondeterministically (~1-in-6) on the current python-copasi build; the "
        "measurement path runs it in a retrying subprocess to avoid this."
    ),
    parameters={
        "coupling_interval": {"type": "float", "default": DEFAULT_COUPLING_INTERVAL},
    },
    core_extensions=[register_cross_engine],
    fidelity="real",
    methods=[
        "libroadrunner CVODE (upper submodel)",
        "COPASI/basico LSODA (lower submodel)",
        "additive delta-coupled cross-engine process composition",
    ],
)
def glycolysis_cross_engine(core=None, coupling_interval=DEFAULT_COUPLING_INTERVAL):
    """Cross-engine glycolysis composite (emitter stripped for the workbench)."""
    return strip_emitter(_build_cross_document(coupling_interval))


def _run_cross_engine_inproc(core=None, coupling_interval=DEFAULT_COUPLING_INTERVAL):
    """In-process cross-engine run (may hit COPASI's flaky loader abort)."""
    if core is None:
        core = allocate_core()
    register_cross_engine(core)
    doc = _build_cross_document(coupling_interval)
    return extract_run(doc, core, GLYCOLYSIS_META["t_end"], GLYCOLYSIS_META["species"])


def run_cross_engine(core=None, coupling_interval=DEFAULT_COUPLING_INTERVAL):
    """Run the cross-engine composite through process-bigraph; ``(traj, t)``.

    Executed in a retrying subprocess (see :func:`_run_in_subprocess_with_retry`)
    so COPASI's intermittent loader abort can never kill the caller.
    """
    return _run_in_subprocess_with_retry("cross", coupling_interval=coupling_interval)


# --------------------------------------------------------------------------- #
# Bulk-SET bit-identity check (the pinned fast-path guarantee).               #
# --------------------------------------------------------------------------- #

def _verify_set_path_equivalence_inproc(perturb: float = 1.1) -> dict:
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

    return {
        # SET-level: the two paths write the SAME initial concentration (the pure
        # bit-identity guarantee — no integration, so ~machine epsilon).
        "set_level_max_abs_diff": max(abs(raw_init[s] - basico_init[s]) for s in settable),
        # Integration-level: same post-step trajectory (CVODE at rtol 1e-8 amplifies
        # the machine-eps SET difference to ~1e-10 — still far below rtol).
        "post_step_max_abs_diff": max(abs(raw[s] - via_basico[s]) for s in species),
        # Not a silent no-op: the perturbed SET visibly moved the result.
        "raw_vs_baseline": max(abs(raw[s] - baseline[s]) for s in species),
    }


def copasi_set_bit_identity() -> dict:
    """Robust (subprocess) accessor for :func:`_verify_set_path_equivalence_inproc`."""
    return _subprocess_with_retry({"kind": "setcheck"})["setcheck"]


# --------------------------------------------------------------------------- #
# Measurement helper + GIL check.                                             #
# --------------------------------------------------------------------------- #

def copasi_mono_accuracy(n_steps: int = DEFAULT_N_STEPS) -> AccuracyResult:
    """COPASI-mono accuracy vs. the arm-1 libroadrunner reference."""
    ref_traj, ref_t = run_monolithic(n_steps=n_steps)
    traj, t = run_copasi_mono(n_steps=n_steps)
    return compare_to_reference(ref_traj, ref_t, traj, t)


def cross_engine_accuracy(coupling_interval: float = DEFAULT_COUPLING_INTERVAL,
                          n_steps: int = DEFAULT_N_STEPS) -> AccuracyResult:
    """Cross-engine composite accuracy vs. the arm-1 libroadrunner reference."""
    ref_traj, ref_t = run_monolithic(n_steps=n_steps)
    traj, t = run_cross_engine(coupling_interval=coupling_interval)
    return compare_to_reference(ref_traj, ref_t, traj, t)


@dataclass
class CopasiMeasurement:
    """COPASI-mono + cross-engine accuracy/timing numbers for the report."""

    mono_max_rel_err: float
    mono_diverged: bool
    mono_time_s: float
    cross_max_rel_err: float
    cross_diverged: bool
    cross_coupling_interval: float


def measure_copasi(n_steps: int = DEFAULT_N_STEPS,
                   coupling_interval: float = DEFAULT_COUPLING_INTERVAL,
                   *, repeats: int = 3) -> CopasiMeasurement:
    """Measure COPASI-mono accuracy + wall-clock and the cross-engine accuracy."""
    mono = copasi_mono_accuracy(n_steps=n_steps)
    timing = time_run(lambda: run_copasi_mono(n_steps=n_steps),
                      repeats=repeats, warmup=True)
    cross = cross_engine_accuracy(coupling_interval=coupling_interval, n_steps=n_steps)
    return CopasiMeasurement(
        mono_max_rel_err=mono.max_rel_err,
        mono_diverged=mono.diverged,
        mono_time_s=timing.median_s,
        cross_max_rel_err=cross.max_rel_err,
        cross_diverged=cross.diverged,
        cross_coupling_interval=coupling_interval,
    )


@dataclass
class GilCheck:
    """Serial vs. ThreadPool timing for N identical COPASI-mono runs.

    ``speedup`` = serial / threaded. A value near 1.0 means COPASI holds the GIL
    during its solve (threads do not overlap); a value near ``n_threads`` would
    mean the solve releases the GIL. The field ``releases_gil`` records the
    verdict at a 1.5x threshold.
    """

    n_threads: int
    serial_s: float
    threaded_s: float
    speedup: float
    releases_gil: bool


def copasi_gil_factor(n_threads: int = 4, n_steps: int = DEFAULT_N_STEPS,
                      *, repeats: int = 3) -> GilCheck:
    """Measure whether basico/COPASI releases the GIL on its solve.

    Runs ``n_threads`` identical COPASI-mono integrations serially, then the same
    batch on a ``ThreadPoolExecutor``; the ratio would be the GIL verdict (arm-5
    style). FINDING (honest): ``run_time_course`` HOLDS the GIL — a speedup ~1.0x,
    no release — AND, beyond that, threaded in-process COPASI SEGFAULTS: COPASI's
    global current-model is not thread-safe, so concurrent runs crash (observed
    here as a reliable SIGSEGV during concurrent Composite setup). So this
    benchmark typically ABORTS rather than returning a clean ratio, which is
    itself the result: COPASI cannot be parallelized with in-process threads — a
    COPASI ensemble must use Ray separate PROCESSES (exactly arm 5's conclusion for
    libroadrunner). Run manually, never in the test suite.
    """
    from concurrent.futures import ThreadPoolExecutor

    warm_copasi_models()   # all COPASI loads before any threaded solve (abort guard)

    # Must use the IN-PROCESS runner: the GIL only serializes threads within ONE
    # interpreter, so a subprocess-based run would not measure it. This is run
    # manually for the report (not in the test suite), so the rare in-process
    # loader abort is acceptable here; re-run on a crash.
    def _one():
        _run_copasi_mono_inproc(n_steps=n_steps)

    def _serial():
        for _ in range(n_threads):
            _one()

    def _threaded():
        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            list(ex.map(lambda _i: _one(), range(n_threads)))

    serial = time_run(_serial, repeats=repeats, warmup=True)
    threaded = time_run(_threaded, repeats=repeats, warmup=True)
    speedup = serial.median_s / threaded.median_s if threaded.median_s > 0 else 0.0
    return GilCheck(
        n_threads=n_threads,
        serial_s=serial.median_s,
        threaded_s=threaded.median_s,
        speedup=speedup,
        releases_gil=speedup > 1.5,
    )
