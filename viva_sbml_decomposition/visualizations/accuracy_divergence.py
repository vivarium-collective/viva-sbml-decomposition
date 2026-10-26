"""AccuracyDivergence — max relative error vs the monolithic reference (log)."""
from __future__ import annotations

from ._plotly import PlotlyVisualization

from ._common import FLOOR, PAL, axis, label, layout, load_metrics


class AccuracyDivergence(PlotlyVisualization):
    def inputs(self):
        return {"arms": "list[string]", "labels": "list[string]",
                "max_rel_err": "list[float]", "exempt": "list[string]"}

    @staticmethod
    def generate_state(config=None):
        m = load_metrics()
        scored = [a for a in m["arms"] if m["accuracy"].get(a) is not None
                  and not m["composability"][a]["accuracy_exempt"]]
        exempt = [a for a in m["arms"] if a not in scored]
        return {"arms": scored, "labels": [label(m, a) for a in scored],
                "max_rel_err": [m["accuracy"][a] for a in scored],
                "exempt": exempt, "exempt_labels": [label(m, a) for a in exempt]}

    def figure(self, state):
        arms, errs, labs = state["arms"], state["max_rel_err"], state["labels"]
        pos = [(a, l, e) for a, l, e in zip(arms, labs, errs) if e > 0]
        zero = [(a, l) for a, l, e in zip(arms, labs, errs) if e <= 0]
        data = [{"type": "bar", "x": [l for _, l, _ in pos], "y": [e for _, _, e in pos],
                 "customdata": [a for a, _, _ in pos],
                 "marker": {"color": [PAL[a] for a, _, _ in pos]},
                 "text": [f"{e:.2g}" for _, _, e in pos], "textposition": "outside",
                 "hovertemplate": "%{customdata}<br>max rel err: %{y:.3g}<extra></extra>",
                 "name": "max rel err"}]
        if zero:
            data.append({"type": "scatter", "mode": "markers+text", "x": [l for _, l in zero],
                         "y": [FLOOR] * len(zero), "text": ["0 (reference)"] * len(zero),
                         "textposition": "top center", "name": "exact (error = 0)",
                         "marker": {"symbol": "diamond-open", "size": 14, "color": "#2563eb",
                                    "line": {"width": 2}},
                         "hovertemplate": "reference: error is exactly 0<br>(drawn at the log floor)<extra></extra>"})
        ann = []
        for i, (a, l) in enumerate(zip(state["exempt"], state.get("exempt_labels", state["exempt"]))):
            ann.append({"xref": "paper", "yref": "paper", "x": 0.99, "xanchor": "right",
                        "y": 0.95 - 0.1 * i, "showarrow": False, "align": "right",
                        "text": f"{a} ({l.replace(chr(10), ' ').replace('<br>', ' ')}): N/A (accuracy-exempt)",
                        "font": {"size": 12}})
        return {"data": data, "layout": layout(
            "Accuracy divergence from the monolithic reference",
            "max relative error across species, log scale; exact-zero shown as a floor marker",
            yaxis=axis("max relative error", type="log", range=[-9.5, 0.5]),
            xaxis={"gridcolor": "rgba(0,0,0,0)"}, showlegend=False,
            margin={"l": 70, "r": 25, "t": 80, "b": 90}, annotations=ann)}
