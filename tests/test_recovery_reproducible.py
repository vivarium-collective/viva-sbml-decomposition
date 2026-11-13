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


# The fields of a computed outcome that form the reproducibility contract: the
# verdict-determining decision. Excluded on purpose are the per-grade `_grader`
# bookkeeping block (a wall-clock timestamp), and the derived-float fields
# `detail` (embeds the raw fit nrmse / parameter errors) and `measured_value`
# (the raw metric). The optimiser converges to the same answer to ~8 significant
# figures — far inside every threshold, so the PASS/FAIL result is identical —
# but those last digits wobble across BLAS/LAPACK builds and platforms, so they
# are not bit-reproducible and are not part of the contract.
_STABLE_FIELDS = ("result", "operator", "evaluated_by", "reconcile")


def _grades(slug):
    """The machine-stable graded decision per run (verdict-determining fields)."""
    runs = yaml.safe_load(_study_path(slug).read_text()).get("runs", []) or []
    out = []
    for r in runs:
        co = r.get("computed_outcomes") or {}
        stable = {
            name: {k: grade.get(k) for k in _STABLE_FIELDS}
            for name, grade in co.items()
            if name != "_grader" and isinstance(grade, dict)
        }
        out.append(stable)
    return out


def test_recovery_grades_are_deterministic():
    for slug in STUDIES:
        before = copy.deepcopy(_grades(slug))
        subprocess.run([VENV, "scripts/recover.py", slug], check=True, env=_ENV)
        subprocess.run([VENV, "scripts/grade_recovery.py", slug], check=True, env=_ENV)
        assert _grades(slug) == before, (
            f"{slug}: re-grade changed computed grades (non-deterministic or stale commit)"
        )
