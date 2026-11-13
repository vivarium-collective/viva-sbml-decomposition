"""CrossEngineAgreement — engines agree to ~1e-7; the split costs ~1e-1.

One log-scale error axis with two groups, so a reviewer sees the thesis at a
glance: tellurium / COPASI / native agree on the same monolithic glycolysis model
to ~1e-7, while the decomposition additive split (mechanical-split-cost) costs a
first-order error of ~0.86 → ~0.007 as the coupling interval shrinks — four to
six orders of magnitude larger. Engine choice is not the decomposition cost; the
split is. COPASI is not bit-reproducible and may abort; a missing arm degrades to
an annotation rather than breaking the figure.
"""
from __future__ import annotations

import json
import pathlib

from ._common import FLOOR, axis, layout
from ._plotly import PlotlyVisualization

_DATA = pathlib.Path(__file__).with_name("cross_engine_data.json")
_ENGINE_COLOR = {"tellurium": "#2563eb", "copasi": "#0891b2", "native": "#7c3aed"}
_SPLIT_COLOR = "#dc2626"


class CrossEngineAgreement(PlotlyVisualization):
    def inputs(self):
        return {"reference": "string", "agreement": "map[float]",
                "aborted": "list[string]", "convergence": "map[list]"}

    @staticmethod
    def generate_state(config=None):
        try:
            data = json.loads(_DATA.read_text())
        except (FileNotFoundError, ValueError):
            data = {"reference": "tellurium", "agreement": {}, "aborted": []}
        from ._common import load_metrics
        split = (load_metrics().get("convergence") or {}).get("arm2") or []
        return {"reference": data.get("reference", "tellurium"),
                "agreement": data.get("agreement", {}),
                "aborted": data.get("aborted", []),
                "convergence": {"arm2": split}}

    def figure(self, state):
        ref = state["reference"]
        agreement = state.get("agreement") or {}
        data = []

        # Cross-engine agreement: non-reference engines that ran, on the log axis.
        eng_x, eng_y, eng_c, eng_t = [], [], [], []
        for eng, v in sorted(agreement.items()):
            if eng == ref or v is None or v <= 0:
                continue
            eng_x.append(eng)
            eng_y.append(v)
            eng_c.append(_ENGINE_COLOR.get(eng, "#64748b"))
            eng_t.append(f"{v:.2g}")
        if eng_x:
            data.append({"type": "bar", "x": eng_x, "y": eng_y, "name": "cross-engine (vs tellurium)",
                         "marker": {"color": eng_c}, "text": eng_t, "textposition": "outside",
                         "hovertemplate": "%{x} vs tellurium<br>max rel err %{y:.3g}<extra></extra>"})
        # The tellurium reference is exactly 0 — show it at the log floor.
        if ref in agreement and (agreement[ref] or 0) <= 0:
            data.append({"type": "scatter", "mode": "markers+text", "x": [ref], "y": [FLOOR],
                         "text": ["0 (reference)"], "textposition": "top center",
                         "name": "tellurium (reference)",
                         "marker": {"symbol": "diamond-open", "size": 14, "color": "#2563eb",
                                    "line": {"width": 2}},
                         "hovertemplate": "reference: error exactly 0 (log floor)<extra></extra>"})

        # Decomposition split error vs coupling interval (the real decomposition cost).
        split = (state.get("convergence") or {}).get("arm2") or []
        if split:
            xs = [f"split @{iv:g}" for iv, _ in split]
            ys = [err for _, err in split]
            data.append({"type": "bar", "x": xs, "y": ys, "name": "decomposition split (mechanical)",
                         "marker": {"color": _SPLIT_COLOR},
                         "text": [f"{e:.2g}" for e in ys], "textposition": "outside",
                         "hovertemplate": "%{x}<br>max rel err %{y:.3g}<extra></extra>"})

        ann = []
        if state.get("aborted"):
            ann.append({"xref": "paper", "yref": "paper", "x": 0.99, "xanchor": "right",
                        "y": 0.95, "showarrow": False, "align": "right",
                        "text": f"aborted (COPASI not bit-reproducible): {', '.join(state['aborted'])}",
                        "font": {"size": 12}})

        return {"data": data, "layout": layout(
            "Engines agree (~1e-7); the decomposition split is the cost (~1e-1)",
            "max relative error, log scale: cross-engine monolithic agreement vs the additive-split error",
            yaxis=axis("max relative error", type="log", range=[-9.5, 0.5]),
            xaxis={"gridcolor": "rgba(0,0,0,0)"}, showlegend=True,
            margin={"l": 70, "r": 25, "t": 80, "b": 90}, annotations=ann)}
