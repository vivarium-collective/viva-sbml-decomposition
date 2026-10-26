"""Decomposition-spectrum composites: the generic composite set.

Importing this package registers every ``@composite_generator`` (the decorators
fire on import) so the workbench discovers them:

* ``engine_monolithic`` / ``native_composite`` - a whole model in ONE engine process
* ``reaction_split`` - two reaction groups, two engines, one additive species store
* ``ensemble`` - N independent instances, same-rate (Ray) or multi-rate (#270 heap)
* ``abm`` - stochastic predator/prey ABM + its mean-field SBML reduction
  (``predator_prey_abm`` / ``predator_prey_meanfield``)

``ensemble`` is a runner as well as a generator; ``run_ensemble`` /
``run_multirate`` return per-instance results.
"""
from . import abm  # noqa: F401  predator_prey_abm / predator_prey_meanfield
from . import engine_monolithic  # noqa: F401  engine_monolithic / native_composite
from . import ensemble  # noqa: F401  ensemble (same-rate + multi-rate #270)
from . import reaction_split  # noqa: F401  reaction_split (two-engine reaction partition)
