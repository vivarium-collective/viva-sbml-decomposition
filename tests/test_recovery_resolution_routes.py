"""The recovery follow-up studies' ``resolution_route`` blocks must conform to the
canonical engine enum (``discovery_engine.resolution``).

The 4-route gap-resolution taxonomy is owned engine-side (DE 2.0): the enum and
the ``{routes, mode}`` shape live in ``discovery_engine.resolution``; this
workspace's study YAML is a *consumer* that must stay in sync. This test points
the sbml recovery cards at that canonical module — if the engine enum changes or
a study drifts (a bad route name, a missing ``mode``, an extra field), the sbml
CI catches it here instead of the data silently diverging.

Guarded with ``importorskip``: when the pinned ``discovery_engine`` predates the
taxonomy module the test skips cleanly; it activates (and enforces) the moment
the dependency carries ``resolution``. The three follow-up studies were verified
to pass ``normalize_resolution_route`` unchanged (already canonical) when wired.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

resolution = pytest.importorskip(
    "discovery_engine.resolution",
    reason="discovery_engine.resolution (DE 2.0 gap-resolution taxonomy) not in the pinned engine yet",
)

_STUDIES = Path(__file__).parent.parent / "workspace" / "investigations" / "discovery-sbml-recovery" / "studies"

# The follow-up / synthesis studies that carry a resolution_route (spawned from a
# refuted/unmet rung). Kept explicit so a new follow-up without a route is noticed.
_EXPECTED_ROUTED = {
    "recover-enzyme-moiety-aware",
    "recover-oscillator-data-richness",
    "recover-gene-switch-rational-laws",
}


def _routed_studies() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for d in sorted(_STUDIES.iterdir()) if _STUDIES.is_dir() else []:
        sy = d / "study.yaml"
        if not sy.is_file():
            continue
        spec = yaml.safe_load(sy.read_text(encoding="utf-8")) or {}
        if "resolution_route" in spec:
            out[d.name] = spec["resolution_route"]
    return out


def test_expected_followups_carry_a_resolution_route():
    """Each spawned follow-up names how its gap would be closed."""
    routed = set(_routed_studies())
    missing = _EXPECTED_ROUTED - routed
    assert not missing, f"follow-up studies missing a resolution_route: {sorted(missing)}"


@pytest.mark.parametrize("slug", sorted(_EXPECTED_ROUTED))
def test_resolution_route_is_canonical(slug):
    """Each resolution_route validates against the engine enum and is already in
    canonical ``{routes, mode}`` form (normalize is a no-op)."""
    rr = _routed_studies()[slug]
    normalized = resolution.normalize_resolution_route(rr)

    assert set(normalized) == {"routes", "mode"}, f"{slug}: block must be strictly {{routes, mode}}, got {sorted(normalized)}"
    assert normalized["mode"] in resolution.RESOLUTION_MODES, f"{slug}: bad mode {normalized['mode']!r}"
    assert normalized["routes"], f"{slug}: routes must be non-empty"
    for route in normalized["routes"]:
        assert route in resolution.RESOLUTION_ROUTES, f"{slug}: unknown route {route!r} (not in {resolution.RESOLUTION_ROUTES})"

    # Authored YAML should already be canonical — normalize must not have to change it.
    assert normalized == resolution.normalize_resolution_route(normalized), f"{slug}: not idempotent under normalize"
