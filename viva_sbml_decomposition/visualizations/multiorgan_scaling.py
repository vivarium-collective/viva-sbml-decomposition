"""MultiorganScaling — arm 8 end-to-end strain curve: monolithic Tellurium vs vivarium decomposition.

The synthetic multi-organ body (arm 8) grows species=3N+1, reactions=4N as the
organ count N climbs. This figure plots, against the species count on a log-log
axis, FOUR measured wall-clocks — the monolithic libroadrunner BUILD (compile) and
raw-CVODE RUN, and the decomposed BUILD (N small compiles) and composite RUN — plus
the decomposed-vs-monolithic accuracy on a second axis. Plotting both halves for
both implementations is the point: the honest story is end-to-end, not build-only.

Headline, read honestly: the monolithic BUILD is superlinear in reactions
(~O(N^1.64)) and is the real strain point; but the decomposed build is SLOWER than
the one monolithic compile up to ~100 species (N separate libroadrunner setups),
reaches rough parity by ~385 species and is a clear win only at ~1500 species; and
the decomposed per-tick RUN overhead is ~16x the monolith's CVODE run, so
decomposition is SLOWER END-TO-END at every N measured here. The decomposed RUN is
only measured through N=128; beyond that the figure shows BUILD-only anchors
(decomposed run left honestly unmeasured, so no end-to-end crossover is claimed).

Numbers are the real medians recorded by
``viva_sbml_decomposition.composites.arm8_multiorgan.multiorgan_scaling`` — carried
here (not in metrics.json) because arm 8 is a standalone scaling study. Do not
hand-edit; re-run the helper to refresh.
"""
from __future__ import annotations

from ._common import axis, caption, layout
from ._plotly import PlotlyVisualization

# Measured medians recorded 2026-10-25 (M-series laptop; repeats=3,
# coupling_interval=0.05). mono_build / decomp_build = te.loada compile(s);
# mono_run = raw CVODE integrate over T_END=10; decomp_run = vivarium composite run
# (construction excluded); accuracy = decomposed-vs-monolithic max_rel_err. decomp
# run+accuracy measured through N=128; N>=256 are BUILD-only anchors (decomposed
# run left honestly unmeasured — too slow for the harness). REAL numbers.
_POINTS = [
    {"n": 2, "species": 7, "mono_build_s": 0.0169, "mono_run_s": 0.00013,
     "decomp_build_s": 0.0260, "decomp_run_s": 0.0178, "accuracy": 1.99e-03},
    {"n": 8, "species": 25, "mono_build_s": 0.0391, "mono_run_s": 0.00027,
     "decomp_build_s": 0.0829, "decomp_run_s": 0.0603, "accuracy": 1.23e-02},
    {"n": 32, "species": 97, "mono_build_s": 0.1974, "mono_run_s": 0.0023,
     "decomp_build_s": 0.3371, "decomp_run_s": 0.2221, "accuracy": 5.24e-02},
    {"n": 128, "species": 385, "mono_build_s": 1.7550, "mono_run_s": 0.0584,
     "decomp_build_s": 1.4001, "decomp_run_s": 0.9452, "accuracy": 3.09e-01},
    {"n": 256, "species": 769, "mono_build_s": 5.5405, "mono_run_s": 0.3953,
     "decomp_build_s": 3.6877, "decomp_run_s": None, "accuracy": None},
    {"n": 512, "species": 1537, "mono_build_s": 18.5304, "mono_run_s": 2.5567,
     "decomp_build_s": 7.7517, "decomp_run_s": None, "accuracy": None},
]


class MultiorganScaling(PlotlyVisualization):
    def inputs(self):
        return {"points": "list"}

    @staticmethod
    def generate_state(config=None):
        return {"points": [dict(p) for p in _POINTS]}

    def figure(self, s):
        pts = s["points"]
        sp = [p["species"] for p in pts]
        mb = [p["mono_build_s"] for p in pts]
        mr = [p["mono_run_s"] for p in pts]
        db = [p["decomp_build_s"] for p in pts]
        # Decomposed run + accuracy only where measured.
        sp_run = [p["species"] for p in pts if p["decomp_run_s"] is not None]
        dr = [p["decomp_run_s"] for p in pts if p["decomp_run_s"] is not None]
        sp_acc = [p["species"] for p in pts if p["accuracy"] is not None]
        acc = [p["accuracy"] for p in pts if p["accuracy"] is not None]

        top = pts[-1]
        data = [
            {"type": "scatter", "mode": "lines+markers",
             "name": "monolithic BUILD (one big compile)",
             "x": sp, "y": mb, "line": {"color": "#dc2626", "width": 3},
             "marker": {"size": 8},
             "hovertemplate": "%{x} species<br>mono build %{y:.3f} s<extra></extra>"},
            {"type": "scatter", "mode": "lines+markers",
             "name": "monolithic run (raw CVODE)",
             "x": sp, "y": mr, "line": {"color": "#dc2626", "width": 2, "dash": "dot"},
             "marker": {"size": 6},
             "hovertemplate": "%{x} species<br>mono run %{y:.3f} s<extra></extra>"},
            {"type": "scatter", "mode": "lines+markers",
             "name": "decomposed BUILD (N small compiles)",
             "x": sp, "y": db, "line": {"color": "#2563eb", "width": 3},
             "marker": {"size": 8},
             "hovertemplate": "%{x} species<br>decomp build %{y:.3f} s<extra></extra>"},
            {"type": "scatter", "mode": "lines+markers",
             "name": "decomposed run (vivarium, N procs; to N=128)",
             "x": sp_run, "y": dr, "line": {"color": "#2563eb", "width": 2, "dash": "dot"},
             "marker": {"size": 6},
             "hovertemplate": "%{x} species<br>decomp run %{y:.3f} s<extra></extra>"},
            {"type": "scatter", "mode": "lines+markers",
             "name": "accuracy (decomp vs mono max rel err)",
             "x": sp_acc, "y": acc, "yaxis": "y2",
             "line": {"color": "#7c3aed", "width": 2, "dash": "dash"},
             "marker": {"size": 7, "symbol": "diamond"},
             "hovertemplate": "%{x} species<br>max rel err %{y:.1e}<extra></extra>"},
        ]
        ann = [
            {"xref": "paper", "yref": "paper", "x": 0.97, "y": 0.20, "xanchor": "right",
             "showarrow": False, "text": f"<b>{top['mono_build_s']:.0f} s</b>",
             "font": {"size": 36, "color": "#dc2626"}},
            {"xref": "paper", "yref": "paper", "x": 0.97, "y": 0.05, "xanchor": "right",
             "showarrow": False,
             "text": f"monolithic compile at {top['species']} species",
             "font": {"size": 13}},
            caption("Scaling knob = N organs (species=3N+1, reactions=4N). The monolithic BUILD is "
                    "superlinear (~O(N^1.64)): 0.20 s &rarr; 18.5 s from 97 to 1537 species &mdash; the "
                    "real strain point. But the decomposed build is N separate libroadrunner setups, so it "
                    "is SLOWER than the one monolithic compile up to ~100 species (1.7x at 97 sp), reaches "
                    "rough parity around 385 sp (1.40 vs 1.76 s) and only becomes a clear win at ~1500 sp "
                    "(2.4x). And the decomposed per-tick RUN overhead dwarfs the monolith's CVODE run (16x "
                    "at 385 sp), so decomposition is SLOWER END-TO-END at every N measured (decomp 2.35 s "
                    "vs mono 1.81 s total at 385 sp); the gap narrows with N but the decomposed run is "
                    "UNMEASURED past N=128, so no end-to-end crossover is claimed. Accuracy is good only at "
                    "small N / fine interval (0.31 at 385 sp) &mdash; the speed and accuracy regimes do NOT "
                    "co-occur. Real medians &mdash; re-run multiorgan_scaling to refresh.",
                    -0.36),
        ]
        return {"data": data, "layout": layout(
            "Multi-organ scaling: the monolithic BUILD strains, but decomposition is not a free win",
            "N-organ synthetic body, shared blood pool &mdash; build AND run, both implementations",
            xaxis=axis("species count (3N + 1)", type="log",
                       tickvals=sp, ticktext=[str(x) for x in sp]),
            yaxis=axis("wall-clock (s)", type="log"),
            yaxis2={"title": {"text": "max rel err"}, "overlaying": "y", "side": "right",
                    "showgrid": False, "type": "log"},
            legend={"orientation": "h", "x": 0, "y": 1.16, "xanchor": "left"},
            margin={"l": 70, "r": 70, "t": 110, "b": 130}, annotations=ann)}
