"""Arm 4: the Teusink glycolysis ODE network as a FULLY NATIVE pbg process.

This is the far pole of the decomposition spectrum. Arms 1-3 all delegate the
kinetics to libroadrunner/tellurium (CVODE evaluates the SBML rate laws); arm 4
does NOT. The derivatives here are computed by *this module's own Python/numpy
code* — a hand port of every Teusink et al. 2000 rate law straight from
:data:`GLYCOLYSIS_ANTIMONY` — and advanced by an explicit scipy integrator. No
libroadrunner / tellurium / roadrunner call happens at runtime; the only thing
read out of the Antimony text is the block of parameter *constants*
(``name = value;`` initializations), never a rate.

**Faithfulness.** The port reproduces the arm-1 libroadrunner reference to
~1e-7 max relative error (see :mod:`tests.test_arm4_native`), which is the
honest validation that the RHS is the real kinetics and not a curve fit.

**Stiffness.** Teusink glycolysis is moderately stiff (the GAPDH/PGK and the
sqrt adenylate-pool assignment rule couple strongly). A fixed-step explicit
method is unattractive; we use scipy ``solve_ivp(method="LSODA")`` on the native
RHS, which keeps the arm native (scipy integrates *our* function) while matching
the reference at the arm-1 integrator tolerances.

**Two leaf representations (same math).** The process carries a ``leaf`` config
selecting how the 17-species state crosses the bigraph wire each tick:

* ``"scalar"`` — species as a ``map[float]`` (17 individual float leaves), the
  same shape arms 1-3 use.
* ``"array"`` — species as ONE numpy ``array`` leaf (shape ``(17,)``).

Identical RHS, identical integrator, identical tick count; only the state leaf
type differs. This lets a sibling perf effort isolate the bigraph framework tax
that comes purely from state shape (17 scalar leaves vs one contiguous array) on
a real ODE. The array variant additionally emits a thin ``map[float]`` *view* of
the state (a second process output) purely so the name-paired RAM-emitter / the
shared accuracy harness can read the trajectory back by species name; that view
is emission-only and is not the canonical state leaf.

Assignment-rule species (``ATP``/``ADP``/``AMP``) are NOT independent ODE
variables — they are algebraic functions of the high-energy-phosphate pool ``P``
and the conserved adenylate sum ``SUM_P`` (the SBML assignment rules). They are
re-derived from ``P`` every evaluation, exactly as the reference does, and are
carried in the state only so the full 17-species set round-trips.
"""
from __future__ import annotations

import re

import numpy as np
from process_bigraph import Process, allocate_core
from process_bigraph.emitter import RAMEmitter
from scipy.integrate import solve_ivp

from viva_sbml_decomposition.models.glycolysis import GLYCOLYSIS_ANTIMONY, GLYCOLYSIS_META

# Matched integrator tolerances (identical to arm 1 / arm 2 / arm 3 defaults).
ABSOLUTE_TOLERANCE = 1e-10
RELATIVE_TOLERANCE = 1e-8
DEFAULT_METHOD = "LSODA"

# The full 17-species set (emitted order), from the shared model metadata.
SPECIES = list(GLYCOLYSIS_META["species"])
# Assignment-rule species: algebraic functions of P + SUM_P, not ODE variables.
ASSIGNED = ["ATP", "ADP", "AMP"]
# The 14 independent ODE state variables (floating, settable species).
SETTABLE = [s for s in SPECIES if s not in ASSIGNED]


def _parse_parameters(antimony: str) -> dict:
    """Extract every ``name = number;`` initialization from the Antimony text.

    This reads *constants only* (kinetic parameters, boundary-species values,
    initial concentrations, compartment sizes) — never a rate. It is how the
    native RHS gets the Teusink parameter set without transcribing ~75 numbers
    by hand (and thus without transcription error). No simulator is invoked.
    """
    params: dict = {}
    pattern = re.compile(
        r"^\s*([A-Za-z_]\w*)\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*;",
        re.M,
    )
    for match in pattern.finditer(antimony):
        params.setdefault(match.group(1), float(match.group(2)))
    return params


_P = _parse_parameters(GLYCOLYSIS_ANTIMONY)

# Boundary (fixed) species + conserved constants, read from the parsed block.
_GLCo = _P["GLCo"]
_ETOH = _P["ETOH"]
_GLY = _P["GLY"]
_SUM_P = _P["SUM_P"]
_F26BP = _P["F26BP"]
_KeqAK = _P["KeqAK"]
_KeqTPI = _P["KeqTPI"]


def _adenylates(p_pool):
    """Re-derive ``(ATP, ADP, AMP)`` from the phosphate pool ``P`` (assignment rules).

    ``ADP`` solves the adenylate-kinase sqrt relation; ``ATP = (P - ADP)/2``;
    ``AMP = SUM_P - ATP - ADP``. Ported verbatim from the SBML assignment rules.
    """
    disc = (
        p_pool * p_pool * (1.0 - 4.0 * _KeqAK)
        + 2.0 * _SUM_P * p_pool * (4.0 * _KeqAK - 1.0)
        + _SUM_P * _SUM_P
    )
    adp = (_SUM_P - np.sqrt(disc)) / (1.0 - 4.0 * _KeqAK)
    atp = (p_pool - adp) / 2.0
    amp = _SUM_P - atp - adp
    return atp, adp, amp


def _fluxes(y):
    """Every Teusink reaction rate (one float per reaction id) at state ``y``.

    ``y`` is ordered as :data:`SETTABLE`. Every expression below is a direct
    port of a reaction rate law in :data:`GLYCOLYSIS_ANTIMONY` (cytosol volume =
    1, so dropped). This is the only place the kinetics live in arm 4 / arm 6;
    :func:`_assemble` turns these fluxes into species derivatives via the
    stoichiometry in :data:`STOICH` (optionally restricted to a reaction subset,
    which is how arm 6 integrates a single native submodel).
    """
    (GLCi, G6P, F6P, F16P, TRIO, BPG, P3G, P2G, PEP, PYR, ACE, Pp, NAD, NADH) = y
    ATP, ADP, AMP = _adenylates(Pp)

    # TRIO is the pooled triose phosphate; GAP/DHAP split by the TPI equilibrium.
    GAP = (_KeqTPI / (1.0 + _KeqTPI)) * TRIO
    DHAP = (1.0 / (1.0 + _KeqTPI)) * TRIO

    P = _P
    vGLK = (P["vGLK_VmGLK"] / (P["vGLK_KmGLKGLCi"] * P["vGLK_KmGLKATP"])) * (
        GLCi * ATP - G6P * ADP / P["vGLK_KeqGLK"]
    ) / (
        (1.0 + GLCi / P["vGLK_KmGLKGLCi"] + G6P / P["vGLK_KmGLKG6P"])
        * (1.0 + ATP / P["vGLK_KmGLKATP"] + ADP / P["vGLK_KmGLKADP"])
    )
    vPGI = (P["vPGI_VmPGI_2"] / P["vPGI_KmPGIG6P_2"]) * (
        G6P - F6P / P["vPGI_KeqPGI_2"]
    ) / (1.0 + G6P / P["vPGI_KmPGIG6P_2"] + F6P / P["vPGI_KmPGIF6P_2"])
    vGLYCO = P["vGLYCO_KGLYCOGEN_3"]
    vTreha = P["vTreha_KTREHALOSE"]

    # PFK: MWC-type rate law (R / T state balance), ported term for term.
    lam1 = F6P / P["KmPFKF6P"]
    lam2 = ATP / P["KmPFKATP"]
    gR = P["gR"]
    Rpfk = 1.0 + lam1 + lam2 + gR * lam1 * lam2
    t_atp_i = (1.0 + P["CiPFKATP"] * (ATP / P["KiPFKATP"])) / (1.0 + ATP / P["KiPFKATP"])
    t_amp = (1.0 + P["CPFKAMP"] * (AMP / P["KPFKAMP"])) / (1.0 + AMP / P["KPFKAMP"])
    t_f = (
        1.0 + P["CPFKF26BP"] * _F26BP / P["KPFKF26BP"] + P["CPFKF16BP"] * F16P / P["KPFKF16BP"]
    ) / (1.0 + _F26BP / P["KPFKF26BP"] + F16P / P["KPFKF16BP"])
    t_atp = 1.0 + P["CPFKATP"] * (ATP / P["KmPFKATP"])
    vPFK = (P["vPFK_VmPFK"] * gR * lam1 * lam2 * Rpfk) / (
        Rpfk * Rpfk
        + P["Lzero"] * (t_atp_i ** 2) * (t_amp ** 2) * (t_f ** 2) * (t_atp ** 2)
    )

    Ka = P["vALD_KmALDF16P"]
    vALD = (P["vALD_VmALD"] / Ka) * (
        F16P - GAP * DHAP / P["vALD_KeqALD"]
    ) / (
        1.0
        + F16P / Ka
        + GAP / P["vALD_KmALDGAP"]
        + DHAP / P["vALD_KmALDDHAP"]
        + GAP * DHAP / (P["vALD_KmALDGAP"] * P["vALD_KmALDDHAP"])
        + F16P * GAP / (P["vALD_KmALDGAPi"] * Ka)
    )
    vGAPDH = (
        P["vGAPDH_VmGAPDHf"] * GAP * NAD / (P["vGAPDH_KmGAPDHGAP"] * P["vGAPDH_KmGAPDHNAD"])
        - P["vGAPDH_VmGAPDHr"] * BPG * NADH / (P["vGAPDH_KmGAPDHBPG"] * P["vGAPDH_KmGAPDHNADH"])
    ) / (
        (1.0 + GAP / P["vGAPDH_KmGAPDHGAP"] + BPG / P["vGAPDH_KmGAPDHBPG"])
        * (1.0 + NAD / P["vGAPDH_KmGAPDHNAD"] + NADH / P["vGAPDH_KmGAPDHNADH"])
    )
    vPGK = (P["vPGK_VmPGK"] / (P["vPGK_KmPGKP3G"] * P["vPGK_KmPGKATP"])) * (
        P["vPGK_KeqPGK"] * BPG * ADP - P3G * ATP
    ) / (
        (1.0 + BPG / P["vPGK_KmPGKBPG"] + P3G / P["vPGK_KmPGKP3G"])
        * (1.0 + ATP / P["vPGK_KmPGKATP"] + ADP / P["vPGK_KmPGKADP"])
    )
    vPGM = (P["vPGM_VmPGM"] / P["vPGM_KmPGMP3G"]) * (
        P3G - P2G / P["vPGM_KeqPGM"]
    ) / (1.0 + P3G / P["vPGM_KmPGMP3G"] + P2G / P["vPGM_KmPGMP2G"])
    vENO = (P["vENO_VmENO"] / P["vENO_KmENOP2G"]) * (
        P2G - PEP / P["vENO_KeqENO"]
    ) / (1.0 + P2G / P["vENO_KmENOP2G"] + PEP / P["vENO_KmENOPEP"])
    vPYK = (P["vPYK_VmPYK"] / (P["vPYK_KmPYKPEP"] * P["vPYK_KmPYKADP"])) * (
        PEP * ADP - PYR * ATP / P["vPYK_KeqPYK"]
    ) / (
        (1.0 + PEP / P["vPYK_KmPYKPEP"] + PYR / P["vPYK_KmPYKPYR"])
        * (1.0 + ATP / P["vPYK_KmPYKATP"] + ADP / P["vPYK_KmPYKADP"])
    )
    ratio_pdc = (PYR / P["vPDC_KmPDCPYR"]) ** P["vPDC_nPDC"]
    vPDC = P["vPDC_VmPDC"] * ratio_pdc / (1.0 + ratio_pdc)
    vSUC = P["vSUC_KSUCC"] * ACE
    vGLT = (P["vGLT_VmGLT"] / P["vGLT_KmGLTGLCo"]) * (
        _GLCo - GLCi / P["vGLT_KeqGLT"]
    ) / (
        1.0
        + _GLCo / P["vGLT_KmGLTGLCo"]
        + GLCi / P["vGLT_KmGLTGLCi"]
        + 0.91 * _GLCo * GLCi / (P["vGLT_KmGLTGLCo"] * P["vGLT_KmGLTGLCi"])
    )
    # ADH is written reverse in the SBML, hence the leading minus sign.
    adh_den = (
        1.0
        + NAD / P["vADH_KiADHNAD"]
        + P["vADH_KmADHNAD"] * _ETOH / (P["vADH_KiADHNAD"] * P["vADH_KmADHETOH"])
        + P["vADH_KmADHNADH"] * ACE / (P["vADH_KiADHNADH"] * P["vADH_KmADHACE"])
        + NADH / P["vADH_KiADHNADH"]
        + NAD * _ETOH / (P["vADH_KiADHNAD"] * P["vADH_KmADHETOH"])
        + P["vADH_KmADHNADH"] * NAD * ACE
        / (P["vADH_KiADHNAD"] * P["vADH_KiADHNADH"] * P["vADH_KmADHACE"])
        + P["vADH_KmADHNAD"] * _ETOH * NADH
        / (P["vADH_KiADHNAD"] * P["vADH_KmADHETOH"] * P["vADH_KiADHNADH"])
        + NADH * ACE / (P["vADH_KiADHNADH"] * P["vADH_KmADHACE"])
        + NAD * _ETOH * ACE
        / (P["vADH_KiADHNAD"] * P["vADH_KmADHETOH"] * P["vADH_KiADHACE"])
        + _ETOH * NADH * ACE
        / (P["vADH_KiADHETOH"] * P["vADH_KiADHNADH"] * P["vADH_KmADHACE"])
    )
    vADH = -(
        (P["vADH_VmADH"] / (P["vADH_KiADHNAD"] * P["vADH_KmADHETOH"]))
        * (NAD * _ETOH - NADH * ACE / P["vADH_KeqADH"])
        / adh_den
    )
    vG3PDH = (
        P["vG3PDH_VmG3PDH"] / (P["vG3PDH_KmG3PDHDHAP"] * P["vG3PDH_KmG3PDHNADH"])
    ) * (DHAP * NADH - _GLY * NAD / P["vG3PDH_KeqG3PDH"]) / (
        (1.0 + DHAP / P["vG3PDH_KmG3PDHDHAP"] + _GLY / P["vG3PDH_KmG3PDHGLY"])
        * (1.0 + NADH / P["vG3PDH_KmG3PDHNADH"] + NAD / P["vG3PDH_KmG3PDHNAD"])
    )
    vATP = P["vATP_KATPASE"] * ATP

    return {
        "vGLT": vGLT, "vGLK": vGLK, "vPGI": vPGI, "vGLYCO": vGLYCO,
        "vTreha": vTreha, "vPFK": vPFK, "vALD": vALD, "vGAPDH": vGAPDH,
        "vPGK": vPGK, "vPGM": vPGM, "vENO": vENO, "vPYK": vPYK, "vPDC": vPDC,
        "vSUC": vSUC, "vADH": vADH, "vG3PDH": vG3PDH, "vATP": vATP,
    }


# Stoichiometry: reaction id -> {settable species: coefficient}. A verbatim
# transcription of the ``d[i] = ...`` assembly against the SBML reaction
# definitions (cytosol volume = 1). :func:`_assemble` sums ``coeff * flux`` over
# the reactions, so restricting the reaction set yields that submodel's partial
# derivatives — exactly what arm 6's native lower-glycolysis node integrates.
STOICH = {
    "vGLT":   {"GLCi": 1.0},
    "vGLK":   {"GLCi": -1.0, "G6P": 1.0, "P": -1.0},
    "vPGI":   {"G6P": -1.0, "F6P": 1.0},
    "vGLYCO": {"G6P": -1.0, "P": -1.0},
    "vTreha": {"G6P": -2.0, "P": -1.0},
    "vPFK":   {"F6P": -1.0, "F16P": 1.0, "P": -1.0},
    "vALD":   {"F16P": -1.0, "TRIO": 2.0},
    "vGAPDH": {"TRIO": -1.0, "BPG": 1.0, "NAD": -1.0, "NADH": 1.0},
    "vPGK":   {"BPG": -1.0, "P3G": 1.0, "P": 1.0},
    "vPGM":   {"P3G": -1.0, "P2G": 1.0},
    "vENO":   {"P2G": -1.0, "PEP": 1.0},
    "vPYK":   {"PEP": -1.0, "PYR": 1.0, "P": 1.0},
    "vPDC":   {"PYR": -1.0, "ACE": 1.0},
    "vSUC":   {"ACE": -2.0, "P": -4.0, "NAD": -3.0, "NADH": 3.0},
    "vADH":   {"ACE": -1.0, "NAD": 1.0, "NADH": -1.0},
    "vG3PDH": {"TRIO": -1.0, "NAD": 1.0, "NADH": -1.0},
    "vATP":   {"P": -1.0},
}

# Fast index lookup for assembly (species name -> position in SETTABLE / y).
_SETTABLE_POS = {s: i for i, s in enumerate(SETTABLE)}


def _assemble(fluxes, active=None):
    """Species derivatives d(SETTABLE)/dt from reaction fluxes via :data:`STOICH`.

    ``active`` restricts the contributing reactions to a subset (``None`` = the
    full network, reproducing the original monolithic RHS exactly).
    """
    d = np.zeros(len(SETTABLE))
    for rxn, contribs in STOICH.items():
        if active is not None and rxn not in active:
            continue
        v = fluxes[rxn]
        for sp, coeff in contribs.items():
            d[_SETTABLE_POS[sp]] += coeff * v
    return d


def _rhs(_t, y):
    """Native Teusink glycolysis RHS: d(14 settable species)/dt (full network)."""
    return _assemble(_fluxes(y))


def make_subset_rhs(reactions):
    """RHS closure that integrates ONLY ``reactions`` (a native submodel's subset).

    Shared adenylate pool and TPI split still derive from the full state ``y``
    (the subset reads the same shared store), but only the listed reactions
    contribute to the derivative — the native analogue of arm 3's reaction-subset
    :class:`SubmodelProcess`.
    """
    active = set(reactions)
    unknown = active - set(STOICH)
    if unknown:
        raise ValueError(f"unknown reactions {sorted(unknown)} (have {sorted(STOICH)})")
    if not active:
        raise ValueError("make_subset_rhs needs at least one reaction")

    def rhs(_t, y):
        return _assemble(_fluxes(y), active)

    return rhs


def initial_concentrations():
    """Initial values for the full 17-species set (seed the state store)."""
    settable = {s: _P[s] for s in SETTABLE}
    atp, adp, amp = _adenylates(settable["P"])
    return {**settable, "ATP": float(atp), "ADP": float(adp), "AMP": float(amp)}


def _full_named(settable_vals):
    """Build the full 17-species name->value dict from a settable-species dict."""
    out = {s: float(settable_vals[s]) for s in SETTABLE}
    atp, adp, amp = _adenylates(out["P"])
    out["ATP"], out["ADP"], out["AMP"] = float(atp), float(adp), float(amp)
    return out


class NativeGlycolysisProcess(Process):
    """Fully native glycolysis ODE network as a single bigraph process.

    Owns the native Teusink RHS (:func:`_rhs`) and advances it with
    ``scipy.integrate.solve_ivp`` over each coupling ``interval``. No
    libroadrunner/tellurium is touched at runtime. The ``leaf`` config selects
    the species state representation crossing the wire:

    * ``"scalar"`` — input/output ``species`` as ``map[float]`` (17 leaves).
    * ``"array"`` — input ``species`` as ``array[(17,), float]``; outputs the
      array state plus a ``map[float]`` ``species_named`` emission view.
    """

    config_schema = {
        "leaf": {"_type": "string", "_default": "scalar"},
        "absolute_tolerance": {"_type": "float", "_default": ABSOLUTE_TOLERANCE},
        "relative_tolerance": {"_type": "float", "_default": RELATIVE_TOLERANCE},
        "method": {"_type": "string", "_default": DEFAULT_METHOD},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        leaf = self.config["leaf"]
        if leaf not in ("scalar", "array"):
            raise ValueError(f"leaf must be 'scalar' or 'array', got {leaf!r}")
        self._leaf = leaf
        self._n = len(SPECIES)
        self._settable_idx = [SPECIES.index(s) for s in SETTABLE]

    def inputs(self):
        if self._leaf == "array":
            return {"species": f"array[({self._n}),float]"}
        return {"species": "map[float]"}

    def outputs(self):
        if self._leaf == "array":
            return {
                "species": f"overwrite[array[({self._n}),float]]",
                "species_named": "overwrite[map[float]]",
            }
        return {"species": "overwrite[map[float]]"}

    def _read_settable(self, incoming):
        """Extract the 14 settable-species values (ordered as SETTABLE) from input."""
        if self._leaf == "array":
            arr = np.asarray(incoming, dtype=float)
            return arr[self._settable_idx].copy()
        return np.array([float(incoming[s]) for s in SETTABLE], dtype=float)

    def _integrate(self, y0, interval):
        sol = solve_ivp(
            _rhs,
            (0.0, interval),
            y0,
            method=self.config["method"],
            rtol=self.config["relative_tolerance"],
            atol=self.config["absolute_tolerance"],
        )
        if not sol.success:
            raise RuntimeError(sol.message)
        return sol.y[:, -1]

    def update(self, state, interval):
        incoming = state.get("species") if state else None
        if incoming is None:
            raise ValueError("NativeGlycolysisProcess requires a 'species' input")

        y0 = self._read_settable(incoming)
        try:
            y1 = self._integrate(y0, interval)
        except (RuntimeError, FloatingPointError, ValueError):
            # Integration failed / left a valid domain: surface the arm as
            # diverged (non-finite) rather than crashing the composite.
            y1 = np.full_like(y0, np.nan)

        named = _full_named(dict(zip(SETTABLE, y1)))
        if self._leaf == "array":
            arr = np.array([named[s] for s in SPECIES], dtype=float)
            return {"species": arr, "species_named": named}
        return {"species": named}


class NativeSubmodelProcess(Process):
    """One native-pbg reaction SUBMODEL as a bigraph process node (arm 6).

    The native analogue of arm 3's :class:`SubmodelProcess`: it reads the shared
    ``species`` store, integrates ONLY its ``reactions`` subset of the native
    Teusink RHS over the coupling ``interval`` with scipy, and returns the
    per-species **delta** it produced. The species leaf is ``map[float]`` (NOT
    overwrite), so the delta reconciles additively through the bigraph apply path
    with the *other* submodel node's delta — even though that other node is a
    Tellurium/libroadrunner :class:`SubmodelProcess` and this one is hand-written
    Python. That cross-engine, store-coupled composition is exactly what a single
    monolithic libroadrunner model cannot express.
    """

    config_schema = {
        "reactions": {"_type": "list[string]", "_default": []},
        "absolute_tolerance": {"_type": "float", "_default": ABSOLUTE_TOLERANCE},
        "relative_tolerance": {"_type": "float", "_default": RELATIVE_TOLERANCE},
        "method": {"_type": "string", "_default": DEFAULT_METHOD},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self._rhs = make_subset_rhs(self.config["reactions"])
        self._init = initial_concentrations()

    def inputs(self):
        return {"species": "map[float]"}

    def outputs(self):
        # Plain map[float] (NOT overwrite): the update is an additive DELTA that
        # reconciles by summation with the other submodel node's delta.
        return {"species": "map[float]"}

    def update(self, state, interval):
        incoming = (state.get("species") if state else None) or {}
        y0 = np.array(
            [float(incoming[s]) if s in incoming else self._init[s] for s in SETTABLE],
            dtype=float,
        )
        try:
            sol = solve_ivp(
                self._rhs,
                (0.0, interval),
                y0,
                method=self.config["method"],
                rtol=self.config["relative_tolerance"],
                atol=self.config["absolute_tolerance"],
            )
            if not sol.success:
                raise RuntimeError(sol.message)
            y1 = sol.y[:, -1]
        except (RuntimeError, FloatingPointError, ValueError):
            # Left the sqrt domain / integration failed: write a non-finite delta
            # so the arm surfaces as diverged instead of crashing the composite.
            return {"species": {s: float("nan") for s in SETTABLE}}
        return {"species": {s: float(y1[i]) - float(y0[i]) for i, s in enumerate(SETTABLE)}}


def register_native_glycolysis(core=None):
    """Register NativeGlycolysisProcess + the RAM emitter on ``core``."""
    if core is None:
        core = allocate_core()
    core.register_link("NativeGlycolysisProcess", NativeGlycolysisProcess)
    core.register_link("ram-emitter", RAMEmitter)
    return core
