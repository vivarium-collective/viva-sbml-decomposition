"""Registry of the SBML/Antimony models used by the decomposition studies.

Each entry carries the full Antimony text so the workbench Resources tab can show
it verbatim (see ``scripts/gen_structured_resources.py``). Counts are derived by
parsing the Antimony: a reaction is a ``name: ... -> / => ...;`` line; species are
the floating species (the dynamic state), taken from metadata where available.
"""
from __future__ import annotations

import re

from ..composites.arm8_multiorgan import all_species, build_multiorgan_antimony
from ..composites.arm9_abm import LV_ANTIMONY, MEANFIELD_SPECIES
from .glycolysis import GLYCOLYSIS_ANTIMONY, GLYCOLYSIS_META

_REACTION_RE = re.compile(r"^\s*\w+\s*:[^;]*(?:->|=>)[^;]*;", re.M)

MULTIORGAN_DISPLAY_N = 4


def count_reactions(antimony: str) -> int:
    """Number of reaction statements (``id: A -> B; rate;``) in an Antimony string."""
    return len(_REACTION_RE.findall(antimony))


def _build() -> dict:
    mo = build_multiorgan_antimony(MULTIORGAN_DISPLAY_N)
    return {
        "glycolysis": {
            "id": "glycolysis",
            "name": "Teusink 2000 yeast glycolysis",
            "summary": (
                "Yeast glycolysis (Teusink et al. 2000), 17 dynamic species; the "
                "reference model for the monolithic, split, modular, native, "
                "coupled and multirate arms."
            ),
            "antimony": GLYCOLYSIS_ANTIMONY,
            "n_species": len(GLYCOLYSIS_META["species"]),
            "n_reactions": count_reactions(GLYCOLYSIS_ANTIMONY),
            "source": GLYCOLYSIS_META["biomodels_id"],
        },
        "multiorgan_scaled": {
            "id": "multiorgan_scaled",
            "name": "Multi-organ body (scaled glycolysis replicas)",
            "summary": (
                "A body of n organs sharing a blood pool, one Antimony model; "
                f"parameterized by n_organs (shown here for n={MULTIORGAN_DISPLAY_N}). "
                "Species = 3n+1, reactions = 4n."
            ),
            "antimony": mo,
            "n_species": len(all_species(MULTIORGAN_DISPLAY_N)),
            "n_reactions": count_reactions(mo),
            "source": "synthetic (scaled glycolysis replicas)",
        },
        "lotka_volterra": {
            "id": "lotka_volterra",
            "name": "Lotka-Volterra predator/prey (mean-field)",
            "summary": (
                "Deterministic mean-field ODE of the stochastic predator/prey ABM; "
                "the reduction that is faithful only at large population scale."
            ),
            "antimony": LV_ANTIMONY,
            "n_species": len(MEANFIELD_SPECIES),
            "n_reactions": count_reactions(LV_ANTIMONY),
            "source": "Lotka-Volterra (mean-field of the predator/prey ABM)",
        },
    }


MODELS: dict = _build()


def get_model(model_id: str) -> dict:
    return MODELS[model_id]


def all_models() -> list:
    return list(MODELS.values())
