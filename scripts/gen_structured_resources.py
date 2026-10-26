"""Write viva_sbml_decomposition/data/vg_data/structured_resources.yaml from the model registry."""
from pathlib import Path

import yaml

from viva_sbml_decomposition.models.registry import all_models

USED_BY = {
    "glycolysis": ["sbml-in-tellurium", "mechanical-split-cost", "biological-modular-seam",
                   "fully-native-pbg", "cross-engine-coupling", "ensemble-and-multirate"],
    "multiorgan_scaled": ["bigger-models-and-abm-limit"],
    "lotka_volterra": ["bigger-models-and-abm-limit"],
}


class _Dumper(yaml.SafeDumper):
    pass


def _str(d, s):
    return d.represent_scalar("tag:yaml.org,2002:str", s, style="|" if "\n" in s else None)


_Dumper.add_representer(str, _str)

OUT = Path(__file__).resolve().parents[1] / "viva_sbml_decomposition/data/vg_data/structured_resources.yaml"


def build() -> dict:
    res = []
    for m in all_models():
        res.append({"id": m["id"], "type": "sbml-model", "name": m["name"], "summary": m["summary"],
                    "antimony": m["antimony"], "n_species": m["n_species"],
                    "n_reactions": m["n_reactions"], "source": m["source"],
                    "used_by": USED_BY[m["id"]]})
    return {"name": "viva-sbml-decomposition resources", "version": 1,
            "description": "SBML models used by the decomposition studies, with full Antimony.",
            "resources": res}


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(yaml.dump(build(), Dumper=_Dumper, sort_keys=False, allow_unicode=True, width=10000))
    print("wrote", OUT)
