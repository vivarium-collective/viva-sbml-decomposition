"""SensitivityIndices — Morris parameter-importance ranking for glycolysis.

A horizontal bar of `mu*` (mean |elementary effect| = importance) per Vmax
parameter, most important at the top, with `sigma` drawn as an interaction /
nonlinearity band. The band is Morris `sigma` (spread of effects across the input
space), NOT a sampling-error confidence interval — a large sigma means the
parameter's effect depends on the others (interaction) or is nonlinear. Reads the
committed machine-tagged baseline so the report renders deterministically;
degrades on empty data.
"""
from __future__ import annotations

import json
import pathlib

from ._common import axis, layout
from ._plotly import PlotlyVisualization

_DATA = pathlib.Path(__file__).with_name("sensitivity_indices.json")
_BAR = "#2563eb"
_BAND = "#9333ea"


class SensitivityIndices(PlotlyVisualization):
    def inputs(self):
        return {"params": "list[string]", "mu_star": "list[float]",
                "sigma": "list[float]", "ranking": "list[string]",
                "qoi": "string", "machine": "string"}

    @staticmethod
    def generate_state(config=None):
        try:
            return json.loads(_DATA.read_text())
        except (FileNotFoundError, ValueError):
            return {"params": [], "mu_star": [], "sigma": [], "ranking": [],
                    "qoi": "ATP", "machine": ""}

    def figure(self, state):
        params = state.get("params") or []
        mu = state.get("mu_star") or []
        sig = state.get("sigma") or []
        by = {p: (m, s) for p, m, s in zip(params, mu, sig)}
        order = [p for p in (state.get("ranking") or params) if p in by]
        # Plotly draws the first y entry at the bottom, so reverse for "top = most important".
        order = list(reversed(order))
        ys = order
        xs = [by[p][0] for p in order]
        es = [by[p][1] for p in order]

        data = []
        if ys:
            data.append({
                "type": "bar", "orientation": "h", "x": xs, "y": ys,
                "marker": {"color": _BAR},
                "error_x": {"type": "data", "array": es, "visible": True,
                            "color": _BAND, "thickness": 1.5},
                "text": [f"{v:.2g}" for v in xs], "textposition": "outside",
                "name": "mu* (importance)",
                "hovertemplate": "%{y}<br>mu* %{x:.3g}<br>sigma (interaction) %{error_x.array:.3g}<extra></extra>"})

        qoi = state.get("qoi", "ATP")
        ann = [{"xref": "paper", "yref": "paper", "x": 0.99, "xanchor": "right", "y": 1.04,
                "showarrow": False, "font": {"size": 12},
                "text": f"QoI: {qoi}; bar = mu* (importance), band = sigma (interaction/nonlinearity)"}]
        return {"data": data, "layout": layout(
            "Global sensitivity: which Vmax parameters drive steady-state ATP",
            "Morris mu* per parameter (importance, most important on top); the band is sigma (interaction/nonlinearity), not a confidence interval",
            xaxis=axis("mu* (mean |elementary effect|)"),
            yaxis={"automargin": True}, showlegend=False,
            margin={"l": 160, "r": 40, "t": 80, "b": 60}, annotations=ann)}
