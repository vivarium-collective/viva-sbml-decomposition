"""Shared loading + styling for the visualization suite (data comes only from metrics.json)."""
from __future__ import annotations

import json
import os

_PATH = os.path.join(os.path.dirname(__file__), "metrics.json")
FLOOR = 1e-9  # log-axis floor used to place an exact-zero error (arm1)
GRID = "rgba(148,163,184,0.22)"
INK = "#64748b"  # mid-grey: legible on light and dark
PAL = {"arm1": "#2563eb", "arm2": "#d97706", "arm3": "#7c3aed",
       "arm4": "#059669", "arm5": "#db2777", "arm6": "#0891b2"}


def load_metrics():
    with open(_PATH) as f:
        return json.load(f)


def arm_overhead(m, arm):
    return m["overhead"].get(arm) or {}


def comp_total(m, arm):
    c = m["composability"][arm]
    return c["substitutability"] + c["coupling_readiness"] + c["multiscale_fit"]


def label(m, arm, sep="<br>"):
    return m["labels"][arm].replace("\n", sep)


def layout(title, subtitle=None, **kw):
    t = title + (f"<br><span style='font-size:12px;color:{INK}'>{subtitle}</span>" if subtitle else "")
    base = {
        "title": {"text": t, "x": 0.02, "xanchor": "left", "font": {"size": 17}},
        "margin": {"l": 70, "r": 25, "t": 80, "b": 70},
        "plot_bgcolor": "rgba(0,0,0,0)", "paper_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": INK},
    }
    base.update(kw)
    return base


def axis(title, **kw):
    d = {"title": {"text": title}, "gridcolor": GRID, "zeroline": False}
    d.update(kw)
    return d


def caption(text, y=-0.28):
    return {"xref": "paper", "yref": "paper", "x": 0, "y": y, "xanchor": "left",
            "yanchor": "top", "text": text, "showarrow": False,
            "font": {"size": 11, "color": INK}}
