"""Recovery grades are deterministic and reproducible.

Re-running the recovery pipeline (``scripts/recover.py`` then
``scripts/grade_recovery.py``) for each graded recovery study must reproduce
the committed grades exactly. The grade is the substantive outcome of each
behavior test (result / measured_value / operator / detail / reconcile); the
per-grade ``_grader`` block is bookkeeping (a wall-clock ``graded_at`` and an
inputs hash) and is excluded from the comparison, which would otherwise fail on
the timestamp alone.
"""
import copy
import os
import pathlib
import subprocess
import sys

import yaml

# The interpreter running the tests — portable across machines and CI, unlike a
# hardcoded venv path. It has the workspace deps installed (same env pytest runs
# in); the recover/grade subprocesses import viva_sbml_decomposition from
# PYTHONPATH=. (the worktree) below.
VENV = sys.executable
INV = "discovery-sbml-recovery"
# The five studies that carry acceptance criteria and have a recovery target in
# RECOVERY_TARGETS. The other dirs under studies/ are non-graded future-direction
# scaffolds and the frontier synthesis; they are not re-gradeable here.
STUDIES = [
    "recover-mass-action",
    "recover-enzyme-kinetics",
    "recover-gene-switch",
    "recover-oscillator",
    "recover-stochastic-abm",
]

_ENV = {**os.environ, "PYTHONPATH": "."}


def _study_path(slug):
    return pathlib.Path(f"workspace/investigations/{INV}/studies/{slug}/study.yaml")


def _grades(slug):
    """Computed grades per run, with the volatile `_grader` block removed."""
    runs = yaml.safe_load(_study_path(slug).read_text()).get("runs", []) or []
    out = []
    for r in runs:
        co = r.get("computed_outcomes") or {}
        out.append({name: grade for name, grade in co.items() if name != "_grader"})
    return out


def test_recovery_grades_are_deterministic():
    for slug in STUDIES:
        before = copy.deepcopy(_grades(slug))
        subprocess.run([VENV, "scripts/recover.py", slug], check=True, env=_ENV)
        subprocess.run([VENV, "scripts/grade_recovery.py", slug], check=True, env=_ENV)
        assert _grades(slug) == before, (
            f"{slug}: re-grade changed computed grades (non-deterministic or stale commit)"
        )
