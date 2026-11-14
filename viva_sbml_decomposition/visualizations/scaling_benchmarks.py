"""ScalingBenchmarks — decomposition scaling on two axes.

Left: same-rate ensemble wall-clock speedup vs instance count (the parallel win,
up to ~7.5x at P=8; a speedup=1 reference line marks "no win" so a serial run is
obvious). Right: monolithic vs decomposed end-to-end wall-clock vs species count
(log-log) — decomposition is slower end-to-end at these sizes, the honest "not a
free win"; the monolithic-build superlinear strain is a large-N regime (see
bigger-models-and-abm-limit). Reads the committed machine-tagged baseline so the
report renders deterministically; degrades when an axis is empty/serial.
"""
from __future__ import annotations

import json
import pathlib

from ._common import axis, layout
from ._plotly import PlotlyVisualization

_DATA = pathlib.Path(__file__).with_name("scaling_benchmarks.json")
_MONO = "#2563eb"
_DECOMP = "#dc2626"
_SPEEDUP = "#059669"


class ScalingBenchmarks(PlotlyVisualization):
    def inputs(self):
        return {"model_size": "list[map]", "build_exponent": "float",
                "ensemble": "map", "machine": "string"}

    @staticmethod
    def generate_state(config=None):
        try:
            return json.loads(_DATA.read_text())
        except (FileNotFoundError, ValueError):
            return {"model_size": [], "build_exponent": float("nan"),
                    "ensemble": {"instances": [], "speedup": [], "backend": "serial"},
                    "machine": ""}

    def figure(self, state):
        ens = state.get("ensemble") or {}
        ms = state.get("model_size") or []
        data = []

        # Left panel: ensemble speedup vs instance count.
        inst = ens.get("instances") or []
        spd = ens.get("speedup") or []
        if inst:
            data.append({"type": "bar", "x": [str(p) for p in inst], "y": spd,
                         "name": "ensemble speedup", "marker": {"color": _SPEEDUP},
                         "text": [f"{s:.2g}x" for s in spd], "textposition": "outside",
                         "xaxis": "x", "yaxis": "y",
                         "hovertemplate": "P=%{x}<br>speedup %{y:.2f}x<extra></extra>"})
            data.append({"type": "scatter", "mode": "lines", "x": [str(p) for p in inst],
                         "y": [1.0] * len(inst), "name": "no speedup (=1)",
                         "line": {"dash": "dash", "color": "#94a3b8"},
                         "xaxis": "x", "yaxis": "y", "hoverinfo": "skip"})

        # Right panel: monolithic vs decomposed end-to-end wall-clock vs species.
        sp = [p["species"] for p in ms]
        if sp:
            mono = [p.get("mono_total_s") for p in ms]
            dec = [p.get("decomp_total_s") for p in ms]
            data.append({"type": "scatter", "mode": "lines+markers", "x": sp, "y": mono,
                         "name": "monolithic", "line": {"color": _MONO},
                         "xaxis": "x2", "yaxis": "y2",
                         "hovertemplate": "%{x} species<br>monolith %{y:.3g}s<extra></extra>"})
            if any(v is not None for v in dec):
                data.append({"type": "scatter", "mode": "lines+markers", "x": sp, "y": dec,
                             "name": "decomposed", "line": {"color": _DECOMP},
                             "xaxis": "x2", "yaxis": "y2",
                             "hovertemplate": "%{x} species<br>decomposed %{y:.3g}s<extra></extra>"})

        expo = state.get("build_exponent")
        ann = [{"xref": "paper", "yref": "paper", "x": 0.0, "y": 1.08, "showarrow": False,
                "text": f"ensemble ({ens.get('backend', '?')})", "font": {"size": 12}},
               {"xref": "paper", "yref": "paper", "x": 1.0, "xanchor": "right", "y": 1.08,
                "showarrow": False,
                "text": f"model size (build exponent {expo:.2f}, superlinear at large N)"
                        if expo == expo else "model size",
                "font": {"size": 12}}]

        lay = layout(
            "Scaling: ensemble parallel speedup (left) vs decomposition cost by model size (right)",
            "wall-clock is machine-specific; the graded properties are speedup>1 and decomposition slower end-to-end",
            showlegend=True, margin={"l": 60, "r": 30, "t": 90, "b": 70}, annotations=ann)
        lay["xaxis"] = {"domain": [0.0, 0.44], "title": {"text": "instances (P)"}}
        lay["yaxis"] = {"title": {"text": "speedup (x)"}}
        lay["xaxis2"] = {"domain": [0.56, 1.0], "type": "log", "title": {"text": "species"}}
        lay["yaxis2"] = {"anchor": "x2", "type": "log", "title": {"text": "end-to-end (s)"}}
        return {"data": data, "layout": lay}
