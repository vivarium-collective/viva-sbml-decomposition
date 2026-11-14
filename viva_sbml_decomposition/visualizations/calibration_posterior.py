"""CalibrationPosterior — the grid posterior over the mass-action rate constants.

Left: the three marginal posteriors over (k1, k2, k2r), each with its true value
marked (a dashed vertical line) — the posterior mass concentrates on the truth.
Right: the posterior-predictive check — the shaded 95% predictive band in
observation space with the noisy data overlaid, so the calibrated model reproduces
the data it was fit to. Reads the committed machine-tagged posterior so the report
renders deterministically; degrades on empty data.
"""
from __future__ import annotations

import json
import pathlib

from ._common import axis, layout
from ._plotly import PlotlyVisualization

_DATA = pathlib.Path(__file__).with_name("calibration_posterior.json")
_PARAM_COLORS = ["#2563eb", "#dc2626", "#059669"]
_SPECIES_COLORS = {"A": "#2563eb", "B": "#dc2626", "C": "#059669"}


class CalibrationPosterior(PlotlyVisualization):
    def inputs(self):
        return {"params": "list[string]", "true": "map[float]", "grid": "map[list]",
                "marginals": "map[list]", "ci95": "map[list]", "predictive": "map", "machine": "string"}

    @staticmethod
    def generate_state(config=None):
        try:
            return json.loads(_DATA.read_text())
        except (FileNotFoundError, ValueError):
            return {"params": [], "true": {}, "grid": {}, "marginals": {}, "ci95": {},
                    "predictive": {"time": [], "observed": {}}, "machine": ""}

    def figure(self, state):
        params = state.get("params") or []
        grid = state.get("grid") or {}
        marg = state.get("marginals") or {}
        true = state.get("true") or {}
        data = []
        shapes = []

        # Left panel: marginal posteriors, each with a dashed true-value line.
        for i, p in enumerate(params):
            color = _PARAM_COLORS[i % len(_PARAM_COLORS)]
            if grid.get(p) and marg.get(p):
                data.append({"type": "scatter", "mode": "lines", "x": grid[p], "y": marg[p],
                             "name": f"{p} posterior", "line": {"color": color},
                             "xaxis": "x", "yaxis": "y",
                             "hovertemplate": p + "=%{x:.3g}<br>posterior %{y:.3g}<extra></extra>"})
            if p in true:
                shapes.append({"type": "line", "xref": "x", "yref": "paper",
                               "x0": true[p], "x1": true[p], "y0": 0.0, "y1": 1.0,
                               "line": {"color": color, "dash": "dash", "width": 1}})

        # Right panel: posterior-predictive band + data in observation space.
        pc = state.get("predictive") or {}
        t = pc.get("time") or []
        for sp, band in (pc.get("observed") or {}).items():
            c = _SPECIES_COLORS.get(sp, "#64748b")
            lo, hi, dat = band.get("lo") or [], band.get("hi") or [], band.get("data") or []
            if t and lo and hi:
                data.append({"type": "scatter", "mode": "lines", "x": t + t[::-1],
                             "y": list(hi) + list(lo)[::-1], "fill": "toself",
                             "fillcolor": c, "opacity": 0.18, "line": {"width": 0},
                             "name": f"{sp} 95% predictive", "xaxis": "x2", "yaxis": "y2",
                             "hoverinfo": "skip", "showlegend": False})
            if t and dat:
                data.append({"type": "scatter", "mode": "markers", "x": t, "y": dat,
                             "name": f"{sp} data", "marker": {"color": c, "size": 4},
                             "xaxis": "x2", "yaxis": "y2",
                             "hovertemplate": sp + " data t=%{x:.2g} %{y:.3g}<extra></extra>"})

        lay = layout(
            "Bayesian calibration: posterior over the rate constants (left) + predictive check (right)",
            "grid posterior; dashed lines mark the true values; the right band is the 95% posterior-predictive over the data",
            showlegend=True, margin={"l": 60, "r": 30, "t": 90, "b": 70}, shapes=shapes)
        lay["xaxis"] = {"domain": [0.0, 0.46], "title": {"text": "rate constant value"}}
        lay["yaxis"] = {"title": {"text": "marginal posterior"}}
        lay["xaxis2"] = {"domain": [0.56, 1.0], "title": {"text": "time"}}
        lay["yaxis2"] = {"anchor": "x2", "title": {"text": "concentration"}}
        return {"data": data, "layout": lay}
