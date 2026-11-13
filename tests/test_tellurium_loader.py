"""The single tellurium model-load seam.

Every semantic flat-SBML/Antimony load routes through `viva_sbml_decomposition
._tellurium.load`, so future integrator settings or error handling land in one
place. The only raw `te.loada` calls allowed are the loader itself and the
`_multiorgan` timing baseline, which deliberately measures raw `te.loada` build
time as the monolithic-build benchmark.
"""
import pathlib
import re


def test_load_returns_simulatable_model():
    from viva_sbml_decomposition import _tellurium

    rr = _tellurium.load("species S1=1; S1 -> ; k*S1; k=0.1;")
    out = rr.simulate(0, 1, 3)
    assert out.shape[0] == 3  # a real libroadrunner result


def test_no_bare_teloada_outside_loader_and_timing():
    allowed = {
        "viva_sbml_decomposition/_tellurium.py",
        "viva_sbml_decomposition/composites/_multiorgan.py",  # timing baseline
    }
    offenders = []
    for p in pathlib.Path("viva_sbml_decomposition").rglob("*.py"):
        if str(p) in allowed:
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if re.search(r"\bte\.loada\(", line) and not line.lstrip().startswith("#"):
                offenders.append(f"{p}:{i}")
    assert not offenders, f"bare te.loada outside the loader: {offenders}"
