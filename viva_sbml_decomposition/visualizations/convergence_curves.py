"""ConvergenceCurves — error shrinks as the coupling interval shrinks."""
from __future__ import annotations

from ._plotly import PlotlyVisualization

from ._common import PAL, axis, caption, label, layout, load_metrics


class ConvergenceCurves(PlotlyVisualization):
    def inputs(self):
        return {"curves": "map[list[list[float]]]", "labels": "map[string]"}

    @staticmethod
    def generate_state(config=None):
        m = load_metrics()
        return {"curves": m["convergence"],
                "labels": {a: label(m, a, " ") for a in m["convergence"]},
                "note": m["notes"]["mismatched_interval"]}

    def figure(self, state):
        data = []
        for arm, pts in state["curves"].items():
            pts = sorted(pts)
            data.append({"type": "scatter", "mode": "lines+markers",
                         "x": [p[0] for p in pts], "y": [p[1] for p in pts],
                         "name": state['labels'].get(arm, arm),
                         "line": {"color": PAL.get(arm), "width": 3},
                         "marker": {"size": 10},
                         "hovertemplate": state['labels'].get(arm, arm) + "<br>interval %{x:g}<br>max rel err %{y:.3g}<extra></extra>"})
        return {"data": data, "layout": layout(
            "Convergence: smaller coupling interval, smaller error",
            "max relative error vs coupling interval, log-log",
            xaxis=axis("coupling interval", type="log"),
            yaxis=axis("max relative error", type="log"),
            legend={"orientation": "h", "y": 1.02, "x": 1, "xanchor": "right"},
            margin={"l": 70, "r": 25, "t": 80, "b": 110},
            annotations=[caption(state.get("note", ""), -0.25)])}
