"""DecisionTable — the bottom line: when to use or avoid each arm."""
from __future__ import annotations

from ._plotly import PlotlyVisualization

from ._common import comp_total, layout, load_metrics

_ROLE = {
    "arm1": "USE as the reference and when you only need one fast, exact model. AVOID when parts must be swapped or coupled: nothing is substitutable.",
    "arm2": "USE for a first decomposition with equal coupling intervals. AVOID coarse intervals: error is {acc} and falls only as the interval shrinks.",
    "arm3": "USE when submodels must be swappable. AVOID mismatched intervals: {mi}",
    "arm4": "USE as the default composable build. Scalar vs array: {sv}",
    "arm5": "USE for many independent instances: Ray gives {sp:.1f}x on {cpu} cores. AVOID threads: {gil}",
    "arm6": "USE to couple across engines (capability, no accuracy axis). Highest composability; cost is wall-clock.",
}


class DecisionTable(PlotlyVisualization):
    def inputs(self):
        return {"rows": "list[node]"}

    @staticmethod
    def generate_state(config=None):
        m = load_metrics()
        e, n = m["ensemble"], m["notes"]
        rows = []
        for a in m["arms"]:
            o = m["overhead"].get(a, {})
            acc = m["accuracy"].get(a)
            exempt = acc is None or m["composability"][a]["accuracy_exempt"]
            frac = o.get("overhead_fraction")
            speed = "n/a" if frac is None else f"{frac:.1%} framework" + (
                f", {o['total_s']:.2f} s" if o.get("total_s") is not None else "")
            if a == "arm5":
                speed = f"{e['speedup']:.2f}x parallel ({e['serial_s']:.2f} s to {e['parallel_s']:.2f} s)"
            rows.append({
                "arm": f"{a}: " + m["labels"][a].replace("\n", " "),
                "speed": speed,
                "accuracy": "N/A (accuracy-exempt)" if exempt else ("exact (reference)" if acc == 0 else f"{acc:.2g} max rel err"),
                "composability": f"{comp_total(m, a)}/9",
                "why": _ROLE[a].format(acc=f"{acc:.2g}" if acc else "", mi=n["mismatched_interval"],
                                       sv=n["scalar_vs_array"], sp=e["speedup"], cpu=e["n_cpus"],
                                       gil=e["gil_note"]),
            })
        return {"rows": rows}

    def figure(self, s):
        rows = s["rows"]
        cols = ["Arm", "Speed (overhead / ratio)", "Accuracy", "Composability", "Use / Avoid / Why"]
        keys = ["arm", "speed", "accuracy", "composability", "why"]
        trace = {"type": "table", "columnwidth": [1.3, 1.6, 1.3, 0.9, 5],
                 "header": {"values": [f"<b>{c}</b>" for c in cols], "align": "left",
                            "fill": {"color": "#334155"}, "font": {"color": "white", "size": 13}},
                 "cells": {"values": [[r[k] for r in rows] for k in keys], "align": "left",
                           "height": 44, "fill": {"color": "rgba(148,163,184,0.15)"},
                           "font": {"size": 12}}}
        return {"data": [trace], "layout": layout(
            "Decision table: which arm when",
            "speed x accuracy x composability (arm1/2/4/6 measured live; arm3/arm5 from their studies)",
            margin={"l": 20, "r": 20, "t": 80, "b": 20}, height=560)}
