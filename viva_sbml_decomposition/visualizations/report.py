"""Assemble a self-contained investigation report embedding all nine figures.

The workbench's study/report renderer is run-coupled — it renders visualizations
that map to a single run's emitted observables. The decomposition figures are
cross-arm AGGREGATES (they read the committed metrics artifact, not one run), so
they don't render through that path. This builder renders every figure's
``update()`` html and lays them out with the investigation narrative + the
decision table into one standalone, shareable HTML page.

Run: ``python -m viva_sbml_decomposition.visualizations.report <out.html>``
(default out: the investigation's ``report.html``).
"""
from __future__ import annotations

import os

import yaml
from process_bigraph import allocate_core

from .abm_meanfield import ABMMeanField
from .abm_extinction import ABMExtinction
from .accuracy_divergence import AccuracyDivergence
from .composability_scorecard import ComposabilityScorecard
from .convergence_curves import ConvergenceCurves
from .decision_table import DecisionTable
from .ensemble_scaling import EnsembleScaling
from .multirate_scaling import MultirateScaling
from .multiorgan_scaling import MultiorganScaling
from .overhead_breakdown import OverheadBreakdown
from .runtime_comparison import RuntimeComparison
from .species_dynamics import SpeciesDynamics
from .tradeoff_map import TradeoffMap

_WS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_INV = os.path.join(_WS, "workspace", "investigations", "sbml-decomposition",
                    "investigation.yaml")

# Figures in reading order, each with a section heading.
_FIGURES = [
    ("Model dynamics (the glycolysis trajectory every arm computes)", SpeciesDynamics),
    ("Runtime comparison — framework share per arm", RuntimeComparison),
    ("Where the time goes — process vs framework", OverheadBreakdown),
    ("Accuracy divergence from the monolithic reference", AccuracyDivergence),
    ("Operator-split / modular convergence vs coupling interval", ConvergenceCurves),
    ("Ensemble — serial vs Ray throughput", EnsembleScaling),
    ("Multi-rate scheduler — event-heap (#270) vs stock scan", MultirateScaling),
    ("Multi-organ scaling — where monolithic Tellurium strains", MultiorganScaling),
    ("Composability scorecard", ComposabilityScorecard),
    ("The 3-way tradeoff map (speed × accuracy × composability)", TradeoffMap),
    ("Adversarial arm — can an ABM even be an SBML model? (only in the large-population limit)", ABMMeanField),
    ("Extinction the mean-field ODE can never show (the SBML-applicability limit)", ABMExtinction),
    ("Decision guide", DecisionTable),
]


def _fig_html(cls) -> str:
    core = allocate_core()
    try:
        inst = cls(core=core)
    except TypeError:
        inst = cls()
    return inst.update(None)["html"]


# Per-study "hero" figures — each study page leads with the figures that capture
# ITS main point (a curated slice of the suite; matches the study.yaml viz blocks).
_STUDY_FIGURES = {
    "arm1-monolithic-tellurium": [
        ("The glycolysis trajectory this arm runs exactly", SpeciesDynamics),
        ("Runtime — framework share per arm", RuntimeComparison),
        ("Where the time goes — process vs framework", OverheadBreakdown)],
    "arm2-operator-split": [
        ("Splitting error shrinks with the coupling interval", ConvergenceCurves),
        ("Accuracy vs the monolithic reference", AccuracyDivergence)],
    "arm3-modular-submodels": [
        ("Convergence of the biological-seam decomposition", ConvergenceCurves),
        ("Composability — this is where it climbs", ComposabilityScorecard)],
    "arm4-native-pbg": [
        ("Reference fidelity of the native port (~1e-7)", AccuracyDivergence),
        ("Runtime — framework share per arm", RuntimeComparison)],
    "arm5-ensemble-ray": [
        ("The parallel win — serial vs Ray", EnsembleScaling),
        ("Runtime — framework share per arm", RuntimeComparison)],
    "arm6-coupling-demo": [
        ("Composability — the cross-engine arm tops the axis", ComposabilityScorecard),
        ("The 3-way tradeoff (arm 6 off the accuracy axis)", TradeoffMap),
        ("Decision guide", DecisionTable)],
    "arm7-multirate": [
        ("Event-heap scheduler (#270) vs stock scan on a many-rate composite", MultirateScaling),
        ("Runtime — framework share per arm (context)", RuntimeComparison)],
    "arm8-multiorgan": [
        ("The strain curve — monolithic compile explodes, decomposition stays cheap", MultiorganScaling),
        ("Composability — why decomposition is the way out at scale", ComposabilityScorecard)],
    "arm9-abm-meanfield": [
        ("Extinction the mean-field ODE can never show (the SBML-applicability limit)", ABMExtinction),
        ("The ABM vs its mean-field ODE reduction — valid at large N, breaks at small N", ABMMeanField),
        ("Where arm 9 sits on the 3-way tradeoff", TradeoffMap)],
}


def _esc(s: str) -> str:
    return (str(s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def build_report(out_path: str) -> str:
    inv = yaml.safe_load(open(_INV))
    ex = inv.get("executive") or {}
    sa = inv.get("scientific_argument") or {}

    def _ul(items):
        return "<ul>" + "".join(f"<li>{_esc(x)}</li>" for x in (items or [])) + "</ul>"

    parts = [f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(inv.get('title'))}</title>
<style>
 :root {{ color-scheme: light dark; --fg:#1a1a1a; --muted:#666; --bg:#fff; --card:#f7f7f9; --accent:#2563eb; }}
 @media (prefers-color-scheme: dark) {{ :root {{ --fg:#e8e8ea; --muted:#9a9aa2; --bg:#141417; --card:#1e1e24; --accent:#6ea8ff; }} }}
 body {{ font: 16px/1.6 -apple-system,Segoe UI,Roboto,sans-serif; color:var(--fg); background:var(--bg); max-width:1000px; margin:0 auto; padding:24px 16px; }}
 h1 {{ font-size:1.7rem; line-height:1.25; }} h2 {{ margin-top:2.2rem; border-bottom:1px solid var(--card); padding-bottom:4px; }}
 .verdict {{ background:var(--card); border-left:4px solid var(--accent); padding:14px 18px; border-radius:8px; margin:16px 0; }}
 .fig {{ background:var(--card); border-radius:10px; padding:12px; margin:12px 0 28px; }}
 .muted {{ color:var(--muted); }} li {{ margin:3px 0; }}
</style></head><body>
<h1>{_esc(inv.get('title'))}</h1>
<p class="muted">{_esc(inv.get('lead'))}</p>
<div class="verdict"><strong>Verdict ({_esc(ex.get('verdict_status'))}):</strong> {_esc(ex.get('verdict'))}</div>
<h2>Scientific argument</h2>
<p><strong>Main claim.</strong> {_esc(sa.get('main_claim'))}</p>
<p><strong>Evidence for</strong></p>{_ul(sa.get('evidence_for'))}
<p><strong>Evidence against / nuance</strong></p>{_ul(sa.get('evidence_against'))}
<p><strong>Caveats</strong></p>{_ul(sa.get('caveats'))}
<h2>Figures</h2>
"""]
    for heading, cls in _FIGURES:
        parts.append(f'<h3>{_esc(heading)}</h3><div class="fig">{_fig_html(cls)}</div>')
    parts.append("<h2>Biological story</h2><p>"
                 + _esc(inv.get("biological_story")).replace("\n\n", "</p><p>") + "</p>")
    parts.append("</body></html>")
    html = "\n".join(parts)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        f.write(html)
    os.replace(tmp, out_path)
    return out_path


def build_study_reports() -> list:
    """Emit a committed per-study report page (studies/<slug>/report.html) for each
    member study: its question/claim, hero figures, findings, and the Antimony model
    it runs. Durable (survives a fresh clone + the PR), unlike the run-coupled
    workbench gallery. Returns the list of written paths."""
    from ..models.glycolysis import GLYCOLYSIS_ANTIMONY, GLYCOLYSIS_META
    studies_dir = os.path.join(os.path.dirname(_INV), "studies")
    written = []
    for slug, figs in _STUDY_FIGURES.items():
        spath = os.path.join(studies_dir, slug, "study.yaml")
        if not os.path.isfile(spath):
            continue
        s = yaml.safe_load(open(spath))
        findings = "".join(
            f"<li>{_esc(f.get('summary'))}</li>" for f in (s.get("findings") or [])
            if isinstance(f, dict) and f.get("summary"))
        fig_html = "".join(
            f'<h3>{_esc(h)}</h3><div class="fig">{_fig_html(c)}</div>' for h, c in figs)
        # Per-study model section. Most arms run the Teusink glycolysis model;
        # arm 9 (adversarial) runs a Lotka-Volterra ABM whose SBML reduction is
        # the mean-field ODE — and whose ABM half is deliberately NOT SBML.
        if slug == "arm9-abm-meanfield":
            from ..composites.arm9_abm import LV_ANTIMONY
            model_html = (
                "<h2>Model — the mean-field SBML reduction (Lotka-Volterra, Antimony)</h2>"
                "<p class=\"muted\">This is the <strong>mean-field ODE reduction</strong> "
                "(the <code>predator_prey_meanfield</code> composite), integrated by "
                "libroadrunner/CVODE — a real, runnable SBML model. The agent-based model "
                "itself (<code>predator_prey_abm</code>, a Gillespie-SSA process over integer "
                "populations) is <strong>NOT an SBML model</strong>: it has no continuous RHS, "
                "and its stochasticity / discreteness / extinction cannot be written as ODE "
                "state. That gap is the finding.</p>"
                "<details><summary>Show the mean-field Lotka-Volterra Antimony</summary>"
                f"<pre>{_esc(LV_ANTIMONY)}</pre></details>")
        else:
            model_html = (
                f"<h2>Model (Teusink glycolysis, {_esc(GLYCOLYSIS_META.get('biomodels_id'))}) — inline Antimony</h2>"
                f"<details><summary>Show the Antimony source "
                f"({len(GLYCOLYSIS_META.get('species') or [])} floating species)</summary>"
                f"<pre>{_esc(GLYCOLYSIS_ANTIMONY)}</pre></details>")
        html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(s.get('title') or slug)}</title>
<style>
 :root {{ color-scheme: light dark; --fg:#1a1a1a; --muted:#666; --bg:#fff; --card:#f7f7f9; --accent:#2563eb; }}
 @media (prefers-color-scheme: dark) {{ :root {{ --fg:#e8e8ea; --muted:#9a9aa2; --bg:#141417; --card:#1e1e24; --accent:#6ea8ff; }} }}
 body {{ font:16px/1.6 -apple-system,Segoe UI,Roboto,sans-serif; color:var(--fg); background:var(--bg); max-width:980px; margin:0 auto; padding:24px 16px; }}
 h1 {{ font-size:1.5rem; }} .claim {{ background:var(--card); border-left:4px solid var(--accent); padding:12px 16px; border-radius:8px; }}
 .fig {{ background:var(--card); border-radius:10px; padding:12px; margin:10px 0 26px; }} .muted {{ color:var(--muted); }}
 pre {{ background:var(--card); padding:12px; border-radius:8px; overflow:auto; font-size:12px; }}
</style></head><body>
<p class="muted"><a href="../../report.html">&larr; investigation report</a></p>
<h1>{_esc(s.get('title') or slug)}</h1>
<p class="muted">{_esc(s.get('question'))}</p>
<div class="claim"><strong>Claim.</strong> {_esc(s.get('claim'))}</div>
<h2>What this arm shows</h2>{fig_html}
<h2>Findings</h2><ul>{findings}</ul>
{model_html}
</body></html>"""
        out = os.path.join(studies_dir, slug, "report.html")
        tmp = out + ".tmp"
        with open(tmp, "w") as f:
            f.write(html)
        os.replace(tmp, out)
        written.append(out)
    return written


if __name__ == "__main__":
    import sys
    default = os.path.join(os.path.dirname(_INV), "report.html")
    out = sys.argv[1] if len(sys.argv) > 1 else default
    print("wrote", build_report(out))
    for p in build_study_reports():
        print("wrote", p)
