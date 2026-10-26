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

from .accuracy_divergence import AccuracyDivergence
from .composability_scorecard import ComposabilityScorecard
from .convergence_curves import ConvergenceCurves
from .decision_table import DecisionTable
from .ensemble_scaling import EnsembleScaling
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
    ("Composability scorecard", ComposabilityScorecard),
    ("The 3-way tradeoff map (speed × accuracy × composability)", TradeoffMap),
    ("Decision guide", DecisionTable),
]


def _fig_html(cls) -> str:
    core = allocate_core()
    try:
        inst = cls(core=core)
    except TypeError:
        inst = cls()
    return inst.update(None)["html"]


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


if __name__ == "__main__":
    import sys
    default = os.path.join(os.path.dirname(_INV), "report.html")
    out = sys.argv[1] if len(sys.argv) > 1 else default
    print("wrote", build_report(out))
