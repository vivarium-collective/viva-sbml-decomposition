"""MultiorganScaling — arm 8 strain curve: monolithic Tellurium vs vivarium decomposition as N grows.

The synthetic multi-organ body (arm 8) grows species=3N+1, reactions=4N as the
organ count N climbs. This figure plots, against the species count on a log-log
axis, the monolithic libroadrunner BUILD time (the compile — where Tellurium
strains, superlinear in reactions), the monolithic raw-CVODE integration time, and
the decomposed vivarium composite run time, plus the decomposed-vs-monolithic
accuracy on a second axis.

The numbers are the real medians recorded by
``viva_sbml_decomposition.composites.arm8_multiorgan.multiorgan_scaling`` — carried
here (not in metrics.json) because arm 8 is a standalone scaling study, not one of
the cross-arm metric aggregates. Do not hand-edit to flatter the curve; re-run the
helper to refresh.
"""
from __future__ import annotations

from ._common import axis, caption, layout
from ._plotly import PlotlyVisualization

# Measured medians recorded 2026-10-25 (M-series laptop; repeats=2, decomposed run
# construction excluded, coupling_interval=0.05). mono_build = te.loada compile;
# mono_run = raw CVODE integrate over T_END=10; decomp_run = vivarium composite run;
# accuracy = decomposed-vs-monolithic max_rel_err over all species. REAL numbers.
_POINTS = [
    {"n": 2, "species": 7, "mono_build_s": 0.0172, "mono_run_s": 0.00012,
     "decomp_run_s": 0.0181, "accuracy": 1.99e-03},
    {"n": 8, "species": 25, "mono_build_s": 0.0402, "mono_run_s": 0.00029,
     "decomp_run_s": 0.0595, "accuracy": 1.23e-02},
    {"n": 32, "species": 97, "mono_build_s": 0.2000, "mono_run_s": 0.0024,
     "decomp_run_s": 0.2228, "accuracy": 5.24e-02},
    {"n": 128, "species": 385, "mono_build_s": 1.7870, "mono_run_s": 0.0584,
     "decomp_run_s": 0.8981, "accuracy": 3.09e-01},
]

# Larger-N monolithic measurements (build/run only — a full decomposed pbg run at
# these N is too slow for the build harness; see the study). Shows the strain
# continuing: the monolithic compile keeps climbing superlinearly.
_MONO_TAIL = [
    {"n": 256, "species": 769, "mono_build_s": 5.5100, "mono_run_s": 0.3950},
    {"n": 512, "species": 1537, "mono_build_s": 18.8200, "mono_run_s": 2.5600},
]


class MultiorganScaling(PlotlyVisualization):
    def inputs(self):
        return {"points": "list"}

    @staticmethod
    def generate_state(config=None):
        return {"points": [dict(p) for p in _POINTS],
                "mono_tail": [dict(p) for p in _MONO_TAIL]}

    def figure(self, s):
        pts = s["points"]
        tail = s.get("mono_tail") or []
        all_mono = pts + tail
        sp_all = [p["species"] for p in all_mono]
        mb = [p["mono_build_s"] for p in all_mono]
        mr = [p["mono_run_s"] for p in all_mono]
        sp = [p["species"] for p in pts]
        dr = [p["decomp_run_s"] for p in pts]
        acc = [p["accuracy"] for p in pts]

        top = all_mono[-1]
        # Build speedup of decomposed over monolithic at the largest COMMON N
        # (decomposed build = N small compiles; recorded in the study).
        data = [
            {"type": "scatter", "mode": "lines+markers",
             "name": "monolithic BUILD (libroadrunner compile)",
             "x": sp_all, "y": mb, "line": {"color": "#dc2626", "width": 3},
             "marker": {"size": 8},
             "hovertemplate": "%{x} species<br>build %{y:.3f} s<extra></extra>"},
            {"type": "scatter", "mode": "lines+markers",
             "name": "monolithic run (raw CVODE)",
             "x": sp_all, "y": mr, "line": {"color": "#d97706", "width": 2, "dash": "dash"},
             "marker": {"size": 7},
             "hovertemplate": "%{x} species<br>mono run %{y:.3f} s<extra></extra>"},
            {"type": "scatter", "mode": "lines+markers",
             "name": "decomposed run (vivarium, N processes)",
             "x": sp, "y": dr, "line": {"color": "#2563eb", "width": 3},
             "marker": {"size": 8},
             "hovertemplate": "%{x} species<br>decomp run %{y:.3f} s<extra></extra>"},
            {"type": "scatter", "mode": "lines+markers",
             "name": "accuracy (decomp vs mono max rel err)",
             "x": sp, "y": acc, "yaxis": "y2",
             "line": {"color": "#7c3aed", "width": 2, "dash": "dot"},
             "marker": {"size": 7, "symbol": "diamond"},
             "hovertemplate": "%{x} species<br>max rel err %{y:.1e}<extra></extra>"},
        ]
        ann = [
            {"xref": "paper", "yref": "paper", "x": 0.97, "y": 0.20, "xanchor": "right",
             "showarrow": False, "text": f"<b>{top['mono_build_s']:.0f} s</b>",
             "font": {"size": 38, "color": "#dc2626"}},
            {"xref": "paper", "yref": "paper", "x": 0.97, "y": 0.05, "xanchor": "right",
             "showarrow": False,
             "text": f"monolithic compile at {top['species']} species",
             "font": {"size": 13}},
            caption("The scaling knob is N organs (species=3N+1, reactions=4N). The monolithic "
                    "libroadrunner BUILD is superlinear in reactions (0.21 s &rarr; 18.8 s from "
                    "97 to 1537 species, ~O(N^1.7)) and the dense CVODE step blows up in step; "
                    "the decomposed body compiles N SMALL models (linear in N) so it builds far "
                    "faster at scale. The additive-split accuracy is first-order in the coupling "
                    "interval and degrades with coupling density (more organs sharing one pool). "
                    "Real medians &mdash; re-run multiorgan_scaling to refresh.", -0.30),
        ]
        return {"data": data, "layout": layout(
            "Multi-organ scaling: monolithic Tellurium strains, decomposition stays cheap to build",
            "N-organ synthetic body, shared blood pool &mdash; wall-clock vs species count",
            xaxis=axis("species count (3N + 1)", type="log",
                       tickvals=sp_all, ticktext=[str(x) for x in sp_all]),
            yaxis=axis("wall-clock (s)", type="log"),
            yaxis2={"title": {"text": "max rel err"}, "overlaying": "y", "side": "right",
                    "showgrid": False, "type": "log"},
            legend={"orientation": "h", "x": 0, "y": 1.14, "xanchor": "left"},
            margin={"l": 70, "r": 70, "t": 100, "b": 115}, annotations=ann)}
