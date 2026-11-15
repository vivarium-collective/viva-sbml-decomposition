"""RustEngineFrontier — where a Rust path pays off, and where it does not.

A horizontal frontier of speedup factors: bigraph *dispatch* (viva-core ~1.27x
scalar, cited) and exact-SSA (viva-hybrid ~2x over numba, a rigorous negative
result) against a ">=5x to pay off" threshold — with the ODE-integration arm shown
as blocked, because viva-core rejects our native composite ("requires a Rust-side
process port"). Reads the committed machine-tagged spike result; degrades on empty.
"""
from __future__ import annotations

import json
import pathlib

from ._common import layout
from ._plotly import PlotlyVisualization

_DATA = pathlib.Path(__file__).with_name("rust_frontier.json")
_VIVA_HYBRID_SPEEDUP = 2.0   # cited: viva-hybrid ~2x over numba (negative result)
_PAYOFF = 5.0                # a Rust path has to clear this to be worth it


class RustEngineFrontier(PlotlyVisualization):
    def inputs(self):
        return {"viva_core_available": "boolean", "native_composite_runs": "boolean",
                "ode_speedup": "float", "dispatch_speedup": "float",
                "requires_port": "boolean", "ode_note": "string",
                "dispatch_note": "string", "viva_hybrid_note": "string",
                "caveat": "string", "machine": "string"}

    @staticmethod
    def generate_state(config=None):
        try:
            return json.loads(_DATA.read_text())
        except (FileNotFoundError, ValueError):
            return {}

    def figure(self, state):
        state = state or {}
        dispatch = state.get("dispatch_speedup")
        ode = state.get("ode_speedup")
        requires_port = bool(state.get("requires_port", True))

        labels, values, colors, texts = [], [], [], []
        # Dispatch — Rust helps, modestly (cited).
        if isinstance(dispatch, (int, float)):
            labels.append("Bigraph dispatch<br>(viva-core, scalar)")
            values.append(float(dispatch))
            colors.append("#f59e0b")
            texts.append(f"{dispatch:.2f}x (cited)")
        # Exact SSA — viva-hybrid negative result (cited).
        labels.append("Exact SSA<br>(viva-hybrid)")
        values.append(_VIVA_HYBRID_SPEEDUP)
        colors.append("#dc2626")
        texts.append(f"~{_VIVA_HYBRID_SPEEDUP:.1f}x, not faithful (cited)")
        # ODE integration — our composite: measured, or blocked.
        if isinstance(ode, (int, float)) and not requires_port:
            labels.append("ODE integration<br>(our composite)")
            values.append(float(ode))
            colors.append("#059669" if ode >= _PAYOFF else "#f59e0b")
            texts.append(f"{ode:.2f}x (measured)")
        else:
            labels.append("ODE integration<br>(our composite)")
            values.append(0.0)
            colors.append("#9ca3af")
            texts.append("requires Rust port — cannot run")

        data = [{
            "type": "bar", "orientation": "h", "y": labels, "x": values,
            "marker": {"color": colors}, "text": texts, "textposition": "outside",
            "hovertemplate": "%{y}: %{x:.2f}x<extra></extra>", "cliponaxis": False,
        }]

        shapes = [
            {"type": "line", "x0": 1.0, "x1": 1.0, "yref": "paper", "y0": 0, "y1": 1,
             "line": {"color": "#64748b", "dash": "dot", "width": 1}},
            {"type": "line", "x0": _PAYOFF, "x1": _PAYOFF, "yref": "paper", "y0": 0, "y1": 1,
             "line": {"color": "#059669", "dash": "dash", "width": 1.5}},
        ]
        annotations = [
            {"x": 1.0, "y": 1.04, "yref": "paper", "text": "parity", "showarrow": False,
             "font": {"size": 10, "color": "#64748b"}},
            {"x": _PAYOFF, "y": 1.04, "yref": "paper", "text": ">=5x to pay off",
             "showarrow": False, "font": {"size": 10, "color": "#059669"}},
        ]
        caveat = state.get("caveat")
        if caveat:
            annotations.append({"x": 0, "y": -0.22, "xref": "paper", "yref": "paper",
                                "xanchor": "left", "align": "left", "showarrow": False,
                                "font": {"size": 10, "color": "#64748b"},
                                "text": _wrap(caveat, 95)})

        lay = layout(
            "Rust-engine frontier: Rust accelerates dispatch, not ODE integration",
            "viva-core rejects our composite without a Rust-side port; viva-hybrid is a rigorous negative result",
            showlegend=False, margin={"l": 170, "r": 40, "t": 90, "b": 110},
            shapes=shapes, annotations=annotations)
        lay["xaxis"] = {"title": {"text": "speedup factor (x) — higher is faster"},
                        "range": [0, max(6.0, max(values) * 1.3 if values else 6.0)]}
        lay["yaxis"] = {"automargin": True}
        return {"data": data, "layout": lay}


def _wrap(text: str, width: int) -> str:
    import textwrap
    return "<br>".join(textwrap.wrap(text, width)) or text
