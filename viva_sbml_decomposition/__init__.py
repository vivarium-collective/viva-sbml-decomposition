"""viva_sbml_decomposition — workspace Python package."""
# Import the composites subpackage so every arm's @composite_generator registers
# at package load — this is what lets the workbench discover the arms.
from . import composites  # noqa: F401

