"""ABMExtinction — the extinction the mean-field ODE can NEVER show.

At small population the stochastic ABM collapses (predators, and sometimes the
whole system, go extinct); the deterministic Lotka-Volterra ODE cycles forever
and shows exactly zero extinction of any species at any population scale. This
figure puts the measured extinction fractions (predator / total / prey) at a
large vs a small population next to that structural zero — the headline reason an
ABM does not reduce to SBML outside its large-population limit.
"""
from __future__ import annotations

import json
import os

from ._plotly import PlotlyVisualization

_DATA = os.path.join(os.path.dirname(__file__), "abm_meanfield_data.json")

_PREDATOR = "#dc2626"  # red — predator collapse (the headline)
_TOTAL = "#7c3aed"     # violet — both species gone
_PREY = "#f59e0b"      # amber — prey gone


class ABMExtinction(PlotlyVisualization):
    """Extinction fractions (predator / total / prey) at large vs small population,
    against the ODE's structural zero."""

    def inputs(self):
        return {"large_scale": "float", "small_scale": "float",
                "large_predator_extinct_fraction": "float",
                "small_predator_extinct_fraction": "float"}

    @staticmethod
    def generate_state(config=None):
        return json.load(open(_DATA))

    def figure(self, state):
        d = state or {}
        K_lo = d.get("small_scale", 8)
        K_hi = d.get("large_scale", 400)
        cats = [f"K={int(K_hi)}<br>(large pop)", f"K={int(K_lo)}<br>(small pop)"]
        series = [
            ("predators extinct", _PREDATOR,
             [d.get("large_predator_extinct_fraction", 0.0), d.get("small_predator_extinct_fraction", 0.0)]),
            ("both species extinct", _TOTAL,
             [d.get("large_total_extinct_fraction", 0.0), d.get("small_total_extinct_fraction", 0.0)]),
            ("prey extinct", _PREY,
             [d.get("large_prey_extinct_fraction", 0.0), d.get("small_prey_extinct_fraction", 0.0)]),
        ]
        traces = [{
            "type": "bar", "name": name, "x": cats, "y": ys,
            "marker": {"color": color},
            "text": [f"{v*100:.0f}%" for v in ys], "textposition": "outside",
            "hovertemplate": name + ": %{y:.1%}<extra></extra>",
        } for name, color, ys in series]
        # The ODE's structural zero — a flat reference line at 0.
        traces.append({
            "type": "scatter", "mode": "lines", "x": cats, "y": [0, 0],
            "name": "mean-field ODE (any species, any K)",
            "line": {"color": "#111827", "width": 2, "dash": "dot"},
            "hovertemplate": "ODE extinction: 0% (structural)<extra></extra>",
        })
        layout = {
            "title": {"text": "Extinction the mean-field ODE can never show<br>"
                      "<sub>stochastic ABM collapse vs the ODE's structural zero</sub>"},
            "barmode": "group",
            "yaxis": {"title": "fraction of runs extinct", "tickformat": ".0%", "range": [0, 1.0]},
            "xaxis": {"title": "population scale"},
            "legend": {"orientation": "h", "y": -0.2},
            "margin": {"t": 70, "b": 80},
            "annotations": [{
                "x": cats[1], "y": 0.92, "xref": "x", "yref": "y", "showarrow": False,
                "text": "at small population the reduction is INVALID", "font": {"color": _PREDATOR},
            }],
        }
        return {"data": traces, "layout": layout}
