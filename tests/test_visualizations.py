import importlib

import pytest
from process_bigraph import allocate_core

V = "viva_sbml_decomposition.visualizations."
FIGS = [
    ("runtime_comparison", "RuntimeComparison"),
    ("overhead_breakdown", "OverheadBreakdown"),
    ("accuracy_divergence", "AccuracyDivergence"),
    ("convergence_curves", "ConvergenceCurves"),
    ("ensemble_scaling", "EnsembleScaling"),
    ("multirate_scaling", "MultirateScaling"),
    ("composability_scorecard", "ComposabilityScorecard"),
    ("tradeoff_map", "TradeoffMap"),
    ("species_dynamics", "SpeciesDynamics"),
    ("decision_table", "DecisionTable"),
    ("abm_meanfield", "ABMMeanField"),
]


def _fig(mod, cls):
    c = getattr(importlib.import_module(V + mod), cls)
    return c, c.generate_state()


def _spec(mod, cls):
    c, s = _fig(mod, cls)
    return c(config={}, core=allocate_core()).figure(s)


@pytest.mark.parametrize("mod,cls", FIGS)
def test_figure_spec_valid(mod, cls):
    c, s = _fig(mod, cls)
    assert s
    assert isinstance(c(config={}, core=allocate_core()).inputs(), dict)
    inst = c(config={}, core=allocate_core())
    out = inst.update(s)
    assert set(out) == {"html"} and isinstance(out["html"], str)
    assert "<div" in out["html"] and "Plotly.newPlot" in out["html"]
    assert f"viva_{cls}" in out["html"]
    spec = inst.figure(s)
    assert spec["data"] and "layout" in spec
    import plotly.graph_objects as go
    go.Figure(spec)  # valid plotly spec


def _xs(spec, i=0):
    return list(spec["data"][i]["x"])


def test_runtime_has_six_bars():
    spec = _spec("runtime_comparison", "RuntimeComparison")
    assert len(spec["data"][0]["y"]) == 6


def test_breakdown_skips_null_total():
    spec = _spec("overhead_breakdown", "OverheadBreakdown")
    assert len(spec["data"]) == 2
    assert len(spec["data"][0]["x"]) == 4


def test_accuracy_excludes_exempt_arms():
    spec = _spec("accuracy_divergence", "AccuracyDivergence")
    bars = [t for t in spec["data"] if t.get("type") == "bar"]
    plotted = sum(len(t["x"]) for t in bars)
    assert plotted == 3  # mechanical split, modular seam, native (SBML-in-Tellurium is a floor marker)
    assert spec["layout"]["yaxis"]["type"] == "log"
    text = " ".join(a["text"] for a in spec["layout"]["annotations"])
    assert text.count("N/A") >= 2


def test_convergence_loglog_two_lines():
    spec = _spec("convergence_curves", "ConvergenceCurves")
    assert len(spec["data"]) == 2
    assert spec["layout"]["xaxis"]["type"] == spec["layout"]["yaxis"]["type"] == "log"


def test_ensemble_speedup_annotated():
    spec = _spec("ensemble_scaling", "EnsembleScaling")
    assert "7.5" in " ".join(a["text"] for a in spec["layout"]["annotations"])


def test_scorecard_heatmap_3x6():
    spec = _spec("composability_scorecard", "ComposabilityScorecard")
    z = spec["data"][0]["z"]
    assert len(z) == 3 and len(z[0]) == 6


def test_tradeoff_cross_engine_off_axis():
    spec = _spec("tradeoff_map", "TradeoffMap")
    main = spec["data"][0]
    assert len(main["x"]) == 4 and all(x > 0 for x in main["x"])
    names = " ".join(str(t.get("name", "")) for t in spec["data"]) + " ".join(
        a["text"] for a in spec["layout"].get("annotations", []))
    assert "Cross-engine" in names and "arm6" not in names and "no accuracy axis" in names


def test_species_dynamics_uses_trajectory():
    spec = _spec("species_dynamics", "SpeciesDynamics")
    assert 4 <= len(spec["data"]) <= 7
    assert len(spec["data"][0]["x"]) == 301


def test_decision_table_six_rows():
    spec = _spec("decision_table", "DecisionTable")
    t = spec["data"][0]
    assert t["type"] == "table" and len(t["cells"]["values"][0]) == 6


def test_standalone_report_embeds_all_figures(tmp_path):
    """The assembled investigation report embeds every figure's interactive html."""
    from viva_sbml_decomposition.visualizations.report import build_report, _FIGURES
    out = build_report(str(tmp_path / "report.html"))
    html = open(out).read()
    assert html.count("Plotly.newPlot") == len(_FIGURES) == 13
    # narrative + decision content present
    assert "Verdict" in html and "composab" in html.lower()


def test_per_study_reports_embed_hero_figures():
    """Each member study gets a committed report page with its hero figures + the model."""
    from viva_sbml_decomposition.visualizations.report import build_study_reports, _STUDY_FIGURES
    paths = build_study_reports()
    assert len(paths) == len(_STUDY_FIGURES) == 7
    for p in paths:
        html = open(p).read()
        assert "Plotly.newPlot" in html          # at least one interactive figure
        assert "Antimony" in html and "Findings" in html
