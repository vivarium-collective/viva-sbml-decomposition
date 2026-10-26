"""EnsembleScaling — arm5 serial vs Ray-parallel wall-clock."""
from __future__ import annotations

from ._plotly import PlotlyVisualization

from ._common import axis, caption, layout, load_metrics


class EnsembleScaling(PlotlyVisualization):
    def inputs(self):
        return {"n_instances": "integer", "n_cpus": "integer", "serial_s": "float",
                "parallel_s": "float", "speedup": "float", "backend": "string",
                "gil_note": "string"}

    @staticmethod
    def generate_state(config=None):
        e = load_metrics()["ensemble"]
        return {k: e[k] for k in ("n_instances", "n_cpus", "serial_s", "parallel_s",
                                  "speedup", "backend", "gil_note")}

    def figure(self, s):
        data = [{"type": "bar", "x": ["serial", f"parallel ({s['backend']})"],
                 "y": [s["serial_s"], s["parallel_s"]],
                 "marker": {"color": ["#94a3b8", "#db2777"]},
                 "text": [f"{s['serial_s']:.3f} s", f"{s['parallel_s']:.3f} s"],
                 "textposition": "outside",
                 "hovertemplate": "%{x}: %{y:.3f} s<extra></extra>"}]
        ann = [{"xref": "paper", "yref": "paper", "x": 0.97, "y": 0.9, "xanchor": "right",
                "showarrow": False, "text": f"<b>{s['speedup']:.2f}x</b>",
                "font": {"size": 44, "color": "#db2777"}},
               {"xref": "paper", "yref": "paper", "x": 0.97, "y": 0.78, "xanchor": "right",
                "showarrow": False, "text": f"on {s['n_cpus']} cores, {s['n_instances']} instances",
                "font": {"size": 13}},
               caption("GIL note: " + s["gil_note"], -0.2)]
        return {"data": data, "layout": layout(
            "Ensemble scaling: Ray beats the GIL",
            f"{s['n_instances']} independent instances, serial vs parallel",
            yaxis=axis("wall-clock (s)", range=[0, s["serial_s"] * 1.2]),
            xaxis={"gridcolor": "rgba(0,0,0,0)"}, showlegend=False,
            margin={"l": 70, "r": 25, "t": 80, "b": 100}, annotations=ann)}
