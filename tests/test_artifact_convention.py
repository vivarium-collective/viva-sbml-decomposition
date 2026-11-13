"""Enforce the two-class sim-artifact convention (see docs/CONVENTIONS.md).

Every study commits a reproducible, seeded artifact — but of one of two classes
depending on how it is graded:

- **Emitter-backed studies** (the decomposition investigation) run real
  simulations through an emitter and commit a seeded ``runs.db``.
- **Code-graded studies** (the recovery investigation) are graded by the
  evaluators against a hidden oracle with no emitter, so their reproducible
  artifact is the seeded ``computed_outcomes`` on the canonical run (plus the
  rendered ``report.html``), not a ``runs.db``.

This test pins the invariant so a committed artifact cannot silently vanish.
"""
import pathlib

import yaml

INV = pathlib.Path("workspace/investigations")

# Synthesis studies aggregate their sibling studies into a landscape/frontier and
# run no new emitter-backed simulation of their own, so they commit no runs.db —
# their artifact is the rendered synthesis. The test still verifies the exemption
# is legitimate (the study genuinely has no own graded/emitted run), so a real
# study cannot hide a missing runs.db by being listed here. See docs/CONVENTIONS.md.
SYNTHESIS_STUDIES = {"decomposition-synthesis", "recoverability-frontier"}


def test_recovery_studies_commit_computed_outcomes():
    studies = INV / "discovery-sbml-recovery" / "studies"
    for p in sorted(studies.glob("*/study.yaml")):
        d = yaml.safe_load(p.read_text()) or {}
        runs = d.get("runs") or []
        if not runs:
            continue  # a scaffolded future-direction study with no run yet
        canonical = next((r for r in runs if r.get("canonical")), runs[0])
        assert canonical.get("computed_outcomes"), (
            f"{p.parent.name}: canonical run is missing computed_outcomes"
        )


def test_decomposition_studies_commit_runs_db():
    studies = INV / "sbml-decomposition" / "studies"
    missing = []
    for d in sorted(studies.iterdir()):
        if not d.is_dir() or (d / "runs.db").exists():
            continue
        spec = yaml.safe_load((d / "study.yaml").read_text()) or {}
        canonical = next((r for r in (spec.get("runs") or []) if r.get("canonical")), None)
        # A missing runs.db is allowed ONLY for a synthesis study with no own
        # graded/emitted run (it aggregates siblings). Anything else is a gap.
        is_synthesis = d.name in SYNTHESIS_STUDIES and not (canonical or {}).get("computed_outcomes")
        if not is_synthesis:
            missing.append(d.name)
    assert not missing, f"emitter-backed studies missing runs.db: {missing}"
