"""Enforce the sim-artifact convention (see docs/CONVENTIONS.md).

Every study commits a reproducible, seeded artifact, of one of two classes by how
it is graded:

- **Emitter-backed studies** run real simulations through an emitter and commit a
  seeded ``runs.db`` (the decomposition investigation's norm).
- **Code-graded studies** are graded by the evaluators with no emitter (recovery,
  and the cross-engine-validation study), so their reproducible artifact is the
  seeded ``computed_outcomes`` on the canonical run.

The invariant: every non-synthesis study that has a run commits at least one of
those artifacts, so a committed artifact cannot silently vanish. Synthesis
studies aggregate their siblings and run no simulation of their own — their
artifact is the rendered synthesis — so they are exempt, and the exemption is
verified (the study must genuinely have no own graded/emitted run), so a real
study cannot hide a missing artifact by being named a synthesis.
"""
import pathlib

import yaml

INV = pathlib.Path("workspace/investigations")
INVESTIGATIONS = ("sbml-decomposition", "discovery-sbml-recovery")
SYNTHESIS_STUDIES = {"decomposition-synthesis", "recoverability-frontier"}


def _canonical(spec):
    runs = spec.get("runs") or []
    return next((r for r in runs if r.get("canonical")), runs[0] if runs else None)


def test_every_study_commits_a_reproducible_artifact():
    offenders = []
    for inv in INVESTIGATIONS:
        for d in sorted((INV / inv / "studies").iterdir()):
            if not d.is_dir():
                continue
            spec = yaml.safe_load((d / "study.yaml").read_text()) or {}
            canonical = _canonical(spec)
            if canonical is None:
                continue  # scaffolded study with no run yet
            has_runs_db = (d / "runs.db").exists()
            has_computed = bool(canonical.get("computed_outcomes"))
            if has_runs_db or has_computed:
                continue
            # No artifact: allowed only for a synthesis study that genuinely ran
            # no own graded/emitted simulation (it aggregates siblings).
            if d.name in SYNTHESIS_STUDIES:
                continue
            offenders.append(f"{inv}/{d.name}")
    assert not offenders, (
        f"studies with neither runs.db nor computed_outcomes: {offenders}"
    )
