"""ABMMeanField — the adversarial study (bigger-models-and-abm-limit): does the ABM reduce to the mean-field ODE?

Two stacked panels, both in DENSITY units (count / population-scale K) so the one
scale-free Lotka-Volterra ODE trajectory is the shared reference in each:

* TOP — LARGE population (K = 400): the Gillespie-SSA ABM ensemble mean (dashed)
  tracks the libroadrunner/CVODE mean-field ODE (solid) to a few percent; the
  predator ±1σ spread is tight and NO run goes extinct. The reduction is VALID.
* BOTTOM — SMALL population (K = 8): the predators COLLAPSE — the dashed ABM
  predator mean decays toward 0 and the predator sample paths crash into the
  extinction floor — in the great majority of runs, while the neutrally-stable
  ODE predator simply orbits forever. The reduction is INVALID.

Extinction is reported honestly as two numbers, because only prey extinction is
truly absorbing while predator extinction leaves the prey to grow: the headline
is the PREDATOR-extinction fraction (the Lotka-Volterra collapse the ODE can
never show) with the TOTAL-extinction fraction (both species zero) alongside.

All numbers are measured (committed to ``abm_meanfield_data.json`` by the ABM
ensemble helper; re-run to refresh). Nothing is tuned to make the ABM match the
ODE — the small-population collapse is the finding. Prey that escape after their
predators die are bounded by a documented population cap (POP_CAP_DENSITY).
"""
from __future__ import annotations

import json
import os

from ._common import GRID, INK, axis, caption, layout
from ._plotly import PlotlyVisualization

_DATA = os.path.join(os.path.dirname(__file__), "abm_meanfield_data.json")

_PREY = "#2563eb"   # blue
_PRED = "#d97706"   # amber
_SAMPLE = "rgba(217,119,6,0.40)"   # faded amber — predator sample paths
_BAND = "rgba(217,119,6,0.15)"


class ABMMeanField(PlotlyVisualization):
    def inputs(self):
        return {"data": "tree"}

    @staticmethod
    def generate_state(config=None):
        with open(_DATA) as f:
            return {"data": json.load(f)}

    def _panel(self, d, tag, xaxis, yaxis, show_legend):
        t = d["t"]
        mean_pred = d[f"{tag}_mean_pred"]
        std = d[f"{tag}_std_pred"]
        upper = [m + s for m, s in zip(mean_pred, std)]
        lower = [max(m - s, 0.0) for m, s in zip(mean_pred, std)]
        traces = []
        # ±1σ band around the predator ensemble mean.
        traces.append({"type": "scatter", "x": t + t[::-1], "y": upper + lower[::-1],
                       "fill": "toself", "fillcolor": _BAND, "line": {"width": 0},
                       "hoverinfo": "skip", "showlegend": False,
                       "xaxis": xaxis, "yaxis": yaxis})
        # A few individual stochastic predator sample paths (the collapse story).
        for i, path in enumerate(d[f"{tag}_sample_pred"]):
            traces.append({"type": "scatter", "mode": "lines", "x": t, "y": path,
                           "line": {"color": _SAMPLE, "width": 1},
                           "name": "ABM predator sample paths", "legendgroup": "sample",
                           "showlegend": show_legend and i == 0,
                           "hovertemplate": "sample pred<br>t=%{x:.1f}<br>%{y:.2f}<extra></extra>",
                           "xaxis": xaxis, "yaxis": yaxis})
        # Mean-field ODE (solid) — prey + predator.
        traces.append({"type": "scatter", "mode": "lines", "x": t, "y": d["ode_prey"],
                       "line": {"color": _PREY, "width": 3}, "name": "ODE prey (mean-field)",
                       "legendgroup": "odeprey", "showlegend": show_legend,
                       "hovertemplate": "ODE prey<br>t=%{x:.1f}<br>%{y:.2f}<extra></extra>",
                       "xaxis": xaxis, "yaxis": yaxis})
        traces.append({"type": "scatter", "mode": "lines", "x": t, "y": d["ode_pred"],
                       "line": {"color": _PRED, "width": 3}, "name": "ODE predator (mean-field)",
                       "legendgroup": "odepred", "showlegend": show_legend,
                       "hovertemplate": "ODE pred<br>t=%{x:.1f}<br>%{y:.2f}<extra></extra>",
                       "xaxis": xaxis, "yaxis": yaxis})
        # ABM ensemble means (dashed) — prey + predator.
        traces.append({"type": "scatter", "mode": "lines", "x": t, "y": d[f"{tag}_mean_prey"],
                       "line": {"color": _PREY, "width": 2.5, "dash": "dot"},
                       "name": "ABM mean prey (SSA ensemble)", "legendgroup": "abmprey",
                       "showlegend": show_legend,
                       "hovertemplate": "ABM prey<br>t=%{x:.1f}<br>%{y:.2f}<extra></extra>",
                       "xaxis": xaxis, "yaxis": yaxis})
        traces.append({"type": "scatter", "mode": "lines", "x": t, "y": mean_pred,
                       "line": {"color": _PRED, "width": 2.5, "dash": "dot"},
                       "name": "ABM mean predator (SSA ensemble)", "legendgroup": "abmpred",
                       "showlegend": show_legend,
                       "hovertemplate": "ABM pred<br>t=%{x:.1f}<br>%{y:.2f}<extra></extra>",
                       "xaxis": xaxis, "yaxis": yaxis})
        return traces

    def figure(self, s):
        d = s["data"]
        data = (self._panel(d, "large", "x", "y", show_legend=True)
                + self._panel(d, "small", "x2", "y2", show_legend=False))

        lk, sk = d["large_scale"], d["small_scale"]
        lpe = d["large_predator_extinct_fraction"]
        spe = d["small_predator_extinct_fraction"]
        ste = d["small_total_extinct_fraction"]
        lr = d["large_max_rel"]

        ann = [
            {"xref": "paper", "yref": "paper", "x": 0.0, "y": 1.0, "xanchor": "left",
             "yanchor": "bottom", "showarrow": False,
             "text": f"<b>LARGE population (K = {lk})</b> — reduction VALID: "
                     f"ABM mean ≈ ODE (max rel err {lr:.1%}), predator extinction {lpe:.0%}",
             "font": {"size": 13, "color": _PREY}},
            {"xref": "paper", "yref": "paper", "x": 0.0, "y": 0.46, "xanchor": "left",
             "yanchor": "bottom", "showarrow": False,
             "text": f"<b>SMALL population (K = {sk})</b> — reduction INVALID: "
                     f"<b>predators collapse in {spe:.0%} of runs</b> "
                     f"(total extinction {ste:.0%}); the ODE orbits forever, 0% either species",
             "font": {"size": 13, "color": "#dc2626"}},
            # Extinction-floor marker on the small panel.
            {"xref": "x2", "yref": "y2", "x": d["t"][-1], "y": 0.0, "xanchor": "right",
             "yanchor": "bottom", "showarrow": False, "text": "predator extinction floor (0)",
             "font": {"size": 10, "color": "#dc2626"}},
            caption(
                f"Same Lotka-Volterra system two ways: a Gillespie-SSA agent-based model "
                f"(discrete individuals, stochastic events) vs its deterministic mean-field "
                f"ODE run in libroadrunner/CVODE — the SBML reduction. Plotted in density "
                f"units (count / K) so the one scale-free ODE is the shared reference. The "
                f"ensemble mean converges to the ODE only as K grows (Kurtz limit): at K={lk} "
                f"the mean-field ODE reproduces the ABM ensemble mean to {lr:.1%} with no "
                f"extinction, but at K={sk} the predators go extinct in {spe:.0%} of runs and "
                f"both species in {ste:.0%} — the absorbing collapse the ODE structurally "
                f"cannot represent (it has no state for discreteness, demographic noise, or "
                f"an extinction probability). Only PREY extinction is absorbing; when predators "
                f"die the prey escape (pure birth), bounded by a documented population cap. The "
                f"ABM is NOT itself an SBML model; it reduces to one ONLY in the large-"
                f"population, well-mixed, deterministic limit. Measured, not tuned.",
                -0.14),
        ]
        # Bound the small panel view so a few escaped-prey runs don't flatten the
        # collapse story (data is unchanged; only the visible range is capped).
        return {"data": data, "layout": layout(
            "Can an agent-based model be an SBML/Tellurium model? Only in the large-population limit",
            "stochastic Gillespie ABM (ensemble mean, predator ±σ + sample paths) vs the mean-field LV ODE",
            height=720,
            xaxis={"gridcolor": GRID, "zeroline": False, "anchor": "y",
                   "matches": "x2", "showticklabels": False},
            xaxis2=dict(axis("time"), anchor="y2"),
            yaxis={"title": {"text": "prey / predator density (count / K)"},
                   "domain": [0.56, 1.0], "gridcolor": GRID, "zeroline": False},
            yaxis2={"title": {"text": "prey / predator density (count / K)"},
                    "domain": [0.0, 0.44], "gridcolor": GRID, "zeroline": True,
                    "zerolinecolor": "#dc2626", "range": [0, 6]},
            legend={"orientation": "h", "x": 0, "y": 1.14, "xanchor": "left",
                    "font": {"size": 10}},
            margin={"l": 70, "r": 25, "t": 120, "b": 110},
            font={"color": INK})}
