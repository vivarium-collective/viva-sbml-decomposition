"""SpeciesDynamics — the reference glycolysis trajectory every arm computes."""
from __future__ import annotations

from process_bigraph.visualization import Visualization

from ._common import axis, layout, load_metrics

N = 6
_COLORS = ["#2563eb", "#d97706", "#7c3aed", "#059669", "#db2777", "#0891b2"]


class SpeciesDynamics(Visualization):
    def inputs(self):
        return {"time": "list[float]", "species": "map[list[float]]"}

    @staticmethod
    def generate_state(config=None):
        t = load_metrics()["reference_trajectory"]
        sp = t["species"]
        top = sorted(sp, key=lambda k: max(sp[k]) - min(sp[k]), reverse=True)[:N]
        return {"time": t["time"], "species": {k: sp[k] for k in top}}

    def update(self, s):
        data = [{"type": "scatter", "mode": "lines", "x": s["time"], "y": y, "name": k,
                 "line": {"width": 2.5, "color": _COLORS[i % len(_COLORS)]},
                 "hovertemplate": k + "<br>t=%{x:.2f}<br>%{y:.3f}<extra></extra>"}
                for i, (k, y) in enumerate(s["species"].items())]
        return {"data": data, "layout": layout(
            "Glycolysis dynamics (reference trajectory)",
            f"the {len(data)} species with the largest dynamic range; all arms compute this",
            xaxis=axis("time"), yaxis=axis("concentration"), hovermode="x unified")}
