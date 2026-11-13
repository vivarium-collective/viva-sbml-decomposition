"""Enforce the sim-artifact convention (see docs/CONVENTIONS.md).

Every study commits a reproducible, seeded artifact, of one of two classes by how
it is graded:

- **Code-graded studies** — all of whose behavior tests use this workspace's
  registered custom evaluators (``recovery_*``, ``cross_engine_agreement``) — are
  graded with no emitter, so their reproducible artifact is the seeded
  ``computed_outcomes`` on the canonical run (recovery, and cross-engine-validation).
- **Emitter-backed studies** — anything else (engine-native ``accuracy`` /
  ``composability`` tests that run real simulations) — commit a seeded ``runs.db``.

The class is derived from the study's ``behavior_tests[].measure.kind`` against
the evaluator registry, so the ``runs.db`` guard stays specific to emitter-backed
studies while code-graded studies are correctly checked for ``computed_outcomes``.
Synthesis studies aggregate siblings and run no simulation of their own, so they
are exempt — and the exemption is verified (no own graded/emitted run), so a real
study cannot hide a missing artifact by being named a synthesis.
"""
import pathlib

import yaml

from viva_sbml_decomposition.evaluators import register_evaluators

INV = pathlib.Path("workspace/investigations")
INVESTIGATIONS = ("sbml-decomposition", "discovery-sbml-recovery")
SYNTHESIS_STUDIES = {"decomposition-synthesis", "recoverability-frontier"}
CODE_GRADED_KINDS = set(register_evaluators({}))


def _canonical(spec):
    runs = spec.get("runs") or []
    return next((r for r in runs if r.get("canonical")), runs[0] if runs else None)


def _is_code_graded(spec):
    kinds = {
        (bt.get("measure") or {}).get("kind")
        for bt in (spec.get("behavior_tests") or [])
    }
    kinds.discard(None)
    return bool(kinds) and kinds <= CODE_GRADED_KINDS


def test_every_study_commits_its_reproducible_artifact():
    offenders = []
    for inv in INVESTIGATIONS:
        for d in sorted((INV / inv / "studies").iterdir()):
            if not d.is_dir():
                continue
            spec = yaml.safe_load((d / "study.yaml").read_text()) or {}
            canonical = _canonical(spec)
            if canonical is None:
                continue  # scaffolded study with no run yet
            if d.name in SYNTHESIS_STUDIES and not canonical.get("computed_outcomes") \
                    and not (d / "runs.db").exists():
                continue  # synthesis: aggregates siblings, no own artifact
            if _is_code_graded(spec):
                if not canonical.get("computed_outcomes"):
                    offenders.append(f"{inv}/{d.name}: code-graded, missing computed_outcomes")
            else:
                if not (d / "runs.db").exists():
                    offenders.append(f"{inv}/{d.name}: emitter-backed, missing runs.db")
    assert not offenders, f"studies missing their reproducible artifact: {offenders}"
