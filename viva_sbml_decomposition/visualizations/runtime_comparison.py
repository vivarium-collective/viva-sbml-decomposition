"""RuntimeComparison — framework-overhead share per arm."""
from __future__ import annotations

from process_bigraph.visualization import Visualization

from ._common import PAL, axis, caption, label, layout, load_metrics


class RuntimeComparison(Visualization):
    def inputs(self):
        return {"arms": "list[string]", "labels": "list[string]",
                "overhead_fraction": "list[maybe[float]]",
                "total_s": "list[maybe[float]]", "caption": "string"}

    @staticmethod
    def generate_state(config=None):
        m = load_metrics()
        arms = m["arms"]
        oh = [m["overhead"].get(a, {}) for a in arms]
        return {"arms": arms, "labels": [label(m, a) for a in arms],
                "overhead_fraction": [o.get("overhead_fraction") for o in oh],
                "total_s": [o.get("total_s") for o in oh],
                "caption": m["notes"]["overhead_fraction"]}

    def update(self, state):
        arms, ys, ts = state["arms"], state["overhead_fraction"], state["total_s"]
        text = []
        for y, t in zip(ys, ts):
            if y is None:
                text.append("not recorded")
            else:
                text.append(f"{y:.1%}" + (f"<br>{t:.2f} s total" if t is not None else "<br>total n/a"))
        trace = {"type": "bar", "x": state["labels"], "y": ys, "text": text,
                 "textposition": "outside", "cliponaxis": False,
                 "marker": {"color": [PAL.get(a, "#64748b") for a in arms]},
                 "customdata": arms,
                 "hovertemplate": "%{customdata}<br>framework share: %{y:.1%}<extra></extra>"}
        return {"data": [trace], "layout": layout(
            "Framework overhead share by arm",
            "share of wall-clock spent in the framework (not process compute)",
            yaxis=axis("overhead fraction", tickformat=".0%", range=[0, max([y for y in ys if y] + [0.1]) * 1.3]),
            xaxis={"gridcolor": "rgba(0,0,0,0)"}, showlegend=False,
            margin={"l": 70, "r": 25, "t": 80, "b": 110},
            annotations=[caption(state["caption"], -0.3)])}
