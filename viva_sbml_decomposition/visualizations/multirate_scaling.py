"""MultirateScaling — arm 7 scan (O(P^2)) vs heap (#270, O(P log P)) wall-clock vs N.

The data is the measured scan-vs-heap timing of the arm-7 multi-rate composite
(independent monolithic glycolysis instances on geometrically-spaced coupling
intervals), carried in :meth:`generate_state` rather than ``metrics.json`` — arm 7
is a scheduler measurement, not one of the cross-arm metric aggregates. The
numbers are the real medians recorded at build time by
``viva_sbml_decomposition.composites.ensemble.multirate_scaling``; the regime
is documented, not fabricated: heap is ~5% SLOWER at tiny N (heap bookkeeping with
nothing to gain) and pulls ahead as P grows, crossing over near P~50-64 and
reaching ~1.6x by P=256 on real libroadrunner/CVODE instances.
"""
from __future__ import annotations

from ._common import axis, caption, layout
from ._plotly import PlotlyVisualization

# Measured medians (repeats=3, run() only, construction excluded), recorded from
# multirate_scaling(ns=(4,16,64,128,256)) on 2026-10-26. scan = stock O(P^2) tick
# loop; heap = process-bigraph #270 event-heap O(P log P). REAL numbers — do not
# hand-edit to inflate the speedup; re-run the helper to refresh.
_POINTS = [
    {"n": 4, "scan_s": 0.00816, "heap_s": 0.00834, "speedup": 0.979},
    {"n": 16, "scan_s": 0.04679, "heap_s": 0.04917, "speedup": 0.952},
    {"n": 64, "scan_s": 0.41929, "heap_s": 0.37145, "speedup": 1.129},
    {"n": 128, "scan_s": 1.55510, "heap_s": 1.27974, "speedup": 1.215},
    {"n": 256, "scan_s": 6.59791, "heap_s": 4.01931, "speedup": 1.642},
]


class MultirateScaling(PlotlyVisualization):
    def inputs(self):
        return {"points": "list"}

    @staticmethod
    def generate_state(config=None):
        return {"points": [dict(p) for p in _POINTS]}

    def figure(self, s):
        pts = s["points"]
        ns = [p["n"] for p in pts]
        scan = [p["scan_s"] for p in pts]
        heap = [p["heap_s"] for p in pts]
        speed = [p["speedup"] for p in pts]
        top = pts[-1]

        data = [
            {"type": "scatter", "mode": "lines+markers", "name": "scan (stock, O(P²))",
             "x": ns, "y": scan, "line": {"color": "#94a3b8", "width": 3},
             "marker": {"size": 8},
             "hovertemplate": "N=%{x}<br>scan %{y:.3f} s<extra></extra>"},
            {"type": "scatter", "mode": "lines+markers", "name": "heap (#270, O(P·logP))",
             "x": ns, "y": heap, "line": {"color": "#2563eb", "width": 3},
             "marker": {"size": 8},
             "hovertemplate": "N=%{x}<br>heap %{y:.3f} s<extra></extra>"},
            {"type": "scatter", "mode": "lines+markers", "name": "speedup (scan/heap)",
             "x": ns, "y": speed, "yaxis": "y2",
             "line": {"color": "#db2777", "width": 2, "dash": "dot"},
             "marker": {"size": 7, "symbol": "diamond"},
             "hovertemplate": "N=%{x}<br>%{y:.2f}×<extra></extra>"},
        ]
        ann = [
            {"xref": "paper", "yref": "paper", "x": 0.97, "y": 0.18, "xanchor": "right",
             "showarrow": False, "text": f"<b>{top['speedup']:.2f}×</b>",
             "font": {"size": 40, "color": "#db2777"}},
            {"xref": "paper", "yref": "paper", "x": 0.97, "y": 0.04, "xanchor": "right",
             "showarrow": False, "text": f"heap vs scan at N={top['n']}",
             "font": {"size": 13}},
            caption("Same math, different tick scheduler &rarr; results bit-identical; only "
                    "wall-clock differs. Real libroadrunner/CVODE instances: the per-process "
                    "integration dominates the per-tick scan at small P, so heap is ~5% slower "
                    "at tiny N (bookkeeping with nothing to gain) and crosses over near P~50-64. "
                    "The multiplier keeps growing with P (into the hundreds here); #270's "
                    "cheap-process micro-benchmark is 4.8× at P=100 rising to 46× at P=1000.", -0.30),
        ]
        return {"data": data, "layout": layout(
            "Multi-rate scheduler: event-heap (#270) vs stock scan",
            "N independent monolith instances on geometrically-spaced intervals",
            xaxis=axis("number of instances P", type="log",
                       tickvals=ns, ticktext=[str(n) for n in ns]),
            yaxis=axis("run wall-clock (s)", type="log"),
            yaxis2={"title": {"text": "speedup (×)"}, "overlaying": "y", "side": "right",
                    "showgrid": False, "rangemode": "tozero", "zeroline": False},
            legend={"orientation": "h", "x": 0, "y": 1.12, "xanchor": "left"},
            margin={"l": 70, "r": 70, "t": 90, "b": 110}, annotations=ann)}
