"""Plotly HTML-port helper: Visualization.update must return {"html": str}."""
from __future__ import annotations

import itertools
import json

from process_bigraph.visualization import Visualization

_CDN = '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'
_counter = itertools.count()


def plotly_html(traces, layout, div_id=None, height=None) -> str:
    div_id = div_id or f"viva_fig_{next(_counter)}"
    h = height or (layout or {}).get("height") or 420
    return (f'<div id="{div_id}" style="width:100%;height:{h}px"></div>' + _CDN
            + f'<script>Plotly.newPlot("{div_id}",' + json.dumps(traces) + ","
            + json.dumps(layout) + ", {responsive:true, displayModeBar:false});</script>")


class PlotlyVisualization(Visualization):
    """Subclasses implement figure(state) -> {"data", "layout"}; update wraps it as html."""

    def figure(self, state):
        raise NotImplementedError

    def update(self, state=None):
        # These are self-contained comparative figures: all data comes from the
        # committed metrics artifact via generate_state(). We ignore any state the
        # workbench passes (its demo/preview path synthesizes an EMPTY state from
        # inputs(), which would blank the figure) so the figure renders identically
        # in preview, study, and investigation contexts.
        spec = self.figure(type(self).generate_state())
        return {"html": plotly_html(spec["data"], spec["layout"],
                                    div_id=f"viva_{type(self).__name__}")}
