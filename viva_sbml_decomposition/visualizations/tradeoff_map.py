"""TradeoffMap — accuracy x framework-overhead, sized/colored by composability."""
from __future__ import annotations

from ._plotly import PlotlyVisualization

from ._common import FLOOR, axis, comp_total, layout, load_metrics


class TradeoffMap(PlotlyVisualization):
    def inputs(self):
        return {"points": "list[node]", "off_axis": "list[node]"}

    @staticmethod
    def generate_state(config=None):
        m = load_metrics()
        pts, off = [], []
        for a in m["arms"]:
            oh = m["overhead"].get(a, {}).get("overhead_fraction")
            row = {"arm": a, "label": m["labels"][a].replace("\n", " "), "overhead": oh,
                   "composability": comp_total(m, a), "accuracy": m["accuracy"].get(a),
                   "total_s": m["overhead"].get(a, {}).get("total_s")}
            if m["accuracy"].get(a) is None or m["composability"][a]["accuracy_exempt"]:
                off.append(row)
            else:
                pts.append(row)
        return {"points": pts, "off_axis": off, "ensemble": m["ensemble"]}

    def figure(self, s):
        pts = s["points"]
        xs = [max(p["accuracy"], FLOOR) for p in pts]
        data = [{"type": "scatter", "mode": "markers+text", "name": "accuracy-scored arms",
                 "x": xs, "y": [p["overhead"] for p in pts],
                 "text": [p["arm"] for p in pts], "textposition": "top center",
                 "customdata": [[p["arm"], p["label"], p["accuracy"], p["overhead"], p["composability"]] for p in pts],
                 "marker": {"size": [14 + 9 * p["composability"] for p in pts],
                            "color": [p["composability"] for p in pts], "cmin": 0, "cmax": 9,
                            "colorscale": "Viridis",
                            "colorbar": {"title": {"text": "composability<br>(sum, 0-9)"}, "x": 0.72, "len": 0.8},
                            "line": {"width": 1.5, "color": "#64748b"}},
                 "hovertemplate": "<b>%{customdata[0]}</b> %{customdata[1]}<br>"
                                  "max rel err: %{customdata[2]:.3g}<br>"
                                  "framework share: %{customdata[3]:.1%}<br>"
                                  "composability: %{customdata[4]}/9<extra></extra>"}]
        ann = [{"xref": "paper", "yref": "paper", "x": 0.0, "y": -0.2, "xanchor": "left",
                "showarrow": False, "font": {"size": 11},
                "text": "arm1 error is exactly 0, drawn at the 1e-9 floor. Lower-left is faster-share and more accurate; bigger/brighter is more composable."}]
        # side panel: off-axis arms, stacked callouts
        notes = {"arm5": "cross-instance ensemble: scales via Ray ({:.1f}x), accuracy-exempt — no accuracy axis",
                 "arm6": "cross-engine composition — no accuracy axis"}
        for i, p in enumerate(s["off_axis"]):
            txt = notes.get(p["arm"], "no accuracy axis")
            if p["arm"] == "arm5":
                txt = txt.format(s["ensemble"]["speedup"])
            txt = f"<b>{p['arm']}</b> ({p['label']}): {txt}<br>composability {p['composability']}/9"
            if p["overhead"] is not None:
                txt += f", framework share {p['overhead']:.1%}"
            data.append({"type": "scatter", "mode": "markers", "name": f"{p['arm']} (off accuracy axis)",
                         "x": [0.5], "y": [1 - 0.28 * i - 0.15], "xaxis": "x2", "yaxis": "y2",
                         "marker": {"symbol": "star", "size": 14 + 9 * p["composability"] * 0.5,
                                    "color": p["composability"], "cmin": 0, "cmax": 9,
                                    "colorscale": "Viridis", "line": {"width": 1.5, "color": "#64748b"}},
                         "hovertemplate": txt + "<extra></extra>", "showlegend": False})
            ann.append({"xref": "x2", "yref": "y2", "x": 0.5, "y": 1 - 0.28 * i - 0.15,
                        "yshift": -22, "showarrow": False, "align": "center", "text": txt,
                        "font": {"size": 11}})
        return {"data": data, "layout": layout(
            "Speed x accuracy x composability",
            "each decomposition trades framework share and error for composability",
            xaxis=axis("max relative error vs reference (log)", type="log", range=[-9.5, 0], domain=[0, 0.66]),
            yaxis=axis("framework overhead share", tickformat=".0%", rangemode="tozero"),
            xaxis2={"domain": [0.76, 1], "range": [0, 1], "visible": False},
            yaxis2={"range": [0, 1], "visible": False, "anchor": "x2"},
            showlegend=False, margin={"l": 70, "r": 25, "t": 80, "b": 100},
            annotations=ann)}
