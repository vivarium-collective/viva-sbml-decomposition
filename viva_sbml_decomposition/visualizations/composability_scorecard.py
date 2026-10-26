"""ComposabilityScorecard — 0-3 scores per study and dimension."""
from __future__ import annotations

from ._plotly import PlotlyVisualization

from ._common import label, layout, load_metrics

DIMS = ["substitutability", "coupling_readiness", "multiscale_fit"]


class ComposabilityScorecard(PlotlyVisualization):
    def inputs(self):
        return {"arms": "list[string]", "labels": "list[string]", "dims": "list[string]",
                "scores": "list[list[integer]]"}

    @staticmethod
    def generate_state(config=None):
        m = load_metrics()
        arms = m["arms"]
        return {"arms": arms, "labels": [label(m, a) for a in arms], "dims": DIMS,
                "scores": [[m["composability"][a][d] for a in arms] for d in DIMS]}

    def figure(self, s):
        names = [d.replace("_", " ") for d in s["dims"]]
        trace = {"type": "heatmap", "x": s["labels"], "y": names, "z": s["scores"],
                 "zmin": 0, "zmax": 3, "colorscale": "Viridis",
                 "text": s["scores"], "texttemplate": "%{text}",
                 "textfont": {"size": 16, "color": "white"},
                 "xgap": 3, "ygap": 3, "colorbar": {"title": {"text": "score (0-3)"}, "tickvals": [0, 1, 2, 3]},
                 "customdata": [s["labels"]] * len(names),
                 "hovertemplate": "%{customdata}<br>%{y}: %{z}/3<extra></extra>"}
        return {"data": [trace], "layout": layout(
            "Composability scorecard",
            "0 = none, 3 = full; rises from the monolith to the cross-engine coupling demo",
            xaxis={"side": "bottom"}, yaxis={"autorange": "reversed"},
            margin={"l": 140, "r": 25, "t": 80, "b": 80})}
