"""OverheadBreakdown — process vs framework seconds, arms with a measured total."""
from __future__ import annotations

from ._plotly import PlotlyVisualization

from ._common import axis, caption, label, layout, load_metrics


class OverheadBreakdown(PlotlyVisualization):
    def inputs(self):
        return {"arms": "list[string]", "labels": "list[string]",
                "process_s": "list[float]", "framework_s": "list[float]",
                "overhead_fraction": "list[float]"}

    @staticmethod
    def generate_state(config=None):
        m = load_metrics()
        arms = [a for a in m["arms"] if m["overhead"].get(a, {}).get("total_s") is not None]
        o = [m["overhead"][a] for a in arms]
        return {"arms": arms, "labels": [label(m, a) for a in arms],
                "process_s": [x["process_s"] for x in o],
                "framework_s": [x["framework_s"] for x in o],
                "overhead_fraction": [x["overhead_fraction"] for x in o],
                "note": m["notes"]["overhead_fraction"]}

    def figure(self, state):
        arms = state["arms"]
        common = {"type": "bar", "x": state["labels"], "customdata": arms}
        data = [
            {**common, "name": "process (compute)", "y": state["process_s"],
             "marker": {"color": "#2563eb"},
             "hovertemplate": "%{customdata}<br>process: %{y:.3f} s<extra></extra>"},
            {**common, "name": "framework", "y": state["framework_s"],
             "marker": {"color": "#f59e0b"},
             "hovertemplate": "%{customdata}<br>framework: %{y:.3f} s<extra></extra>"},
        ]
        tops = [p + f for p, f in zip(state["process_s"], state["framework_s"])]
        ann = [{"x": lab, "y": t, "yshift": 22, "showarrow": False,
                "text": f"{t:.2f} s<br>{o:.1%} framework", "font": {"size": 11}}
               for lab, t, o in zip(state["labels"], tops, state["overhead_fraction"])]
        ann.append(caption(state.get("note", ""), -0.32))
        return {"data": data, "layout": layout(
            "Where the time goes: process vs framework",
            "arms with a measured total (arm3, arm5 not re-timed live); linear seconds, so arm6 dominates",
            barmode="stack", yaxis=axis("wall-clock (s)"),
            xaxis={"gridcolor": "rgba(0,0,0,0)"},
            margin={"l": 70, "r": 25, "t": 80, "b": 110},
            legend={"orientation": "h", "y": 1.02, "x": 1, "xanchor": "right"},
            annotations=ann)}
