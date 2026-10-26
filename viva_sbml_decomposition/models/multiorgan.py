"""Synthetic multi-organ body: N organs (3-species enzymatic chain each) sharing ONE
blood pool, generated as a single Antimony model.

Species = 3N + 1 (the blood pool + 3 per organ), reactions = 4N. The monolithic
model string is what the registry (``multiorgan_scaled``) and the generic
``engine_monolithic`` run; its libroadrunner compile time is what strains as N
grows. :func:`single_organ_antimony` is one organ's slice, used by the per-organ
decomposition in :mod:`viva_sbml_decomposition.composites._multiorgan`.
"""
from __future__ import annotations

from typing import List

# --- the synthetic body ------------------------------------------------------

BLOOD = "Glc_blood"          # the single shared (plasma) metabolite
BLOOD_INITIAL = 100.0        # all mass starts in the blood pool
PER_ORGAN_SPECIES = ("G", "M", "P")   # 3-species enzymatic chain per organ
T_END = 10.0                 # the body relaxes to steady state well before this

# Per-organ kinetics (shared across organs except k1, which is mildly
# heterogeneous so the organs are non-degenerate). k_up small enough that the N
# organs do not collectively drain the pool in a single coupling interval.
K_UP, K2, K_REL = 0.1, 0.5, 0.3

ABSOLUTE_TOLERANCE = 1e-10
RELATIVE_TOLERANCE = 1e-8

DEFAULT_N_ORGANS = 8
DEFAULT_N_STEPS = 200        # monolithic emitter resolution (interval = T_END/N_STEPS)


def _k1(i: int) -> float:
    return 1.0 + 0.05 * i


def organ_species(i: int) -> List[str]:
    """The private floating-species ids of organ ``i`` (``o{i}_G/M/P``)."""
    return [f"o{i}_{s}" for s in PER_ORGAN_SPECIES]


def all_species(n: int) -> List[str]:
    """Every floating species of the ``n``-organ body: the blood pool + all organs."""
    out = [BLOOD]
    for i in range(n):
        out.extend(organ_species(i))
    return out


def _organ_reactions(i: int) -> str:
    g, m, p = organ_species(i)
    return (
        f"  {g} = 0; {m} = 0; {p} = 0;\n"
        f"  o{i}_uptake: {BLOOD} -> {g}; k_up * {BLOOD};\n"
        f"  o{i}_r1: {g} -> {m}; {_k1(i):.6f} * {g};\n"
        f"  o{i}_r2: {m} -> {p}; k2 * {m};\n"
        f"  o{i}_rel: {p} -> {BLOOD}; k_rel * {p};\n"
    )


def _header() -> str:
    return (f"  {BLOOD} = {BLOOD_INITIAL};\n"
            f"  k_up = {K_UP}; k2 = {K2}; k_rel = {K_REL};\n")


def build_multiorgan_antimony(n: int) -> str:
    """The whole ``n``-organ body as ONE Antimony model (the monolithic arm).

    Species ``= 3n + 1`` (the blood pool + 3 per organ), reactions ``= 4n``. This
    is the string handed to the single libroadrunner instance; its compile time is
    what strains as ``n`` grows. Also surfaced on the monolithic generator's
    ``config.model`` so the workbench Model page can show it.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    body = "".join(_organ_reactions(i) for i in range(n))
    return f"model multiorgan_{n}()\n{_header()}{body}end"


def single_organ_antimony(i: int) -> str:
    """One organ (``blood`` pool + organ ``i``'s 3 species and 4 reactions).

    Each decomposed :class:`OrganProcess` builds its own libroadrunner instance
    from this — ``N`` small compiles (linear in ``N``) rather than one giant one.
    """
    return f"model organ_{i}()\n{_header()}{_organ_reactions(i)}end"
