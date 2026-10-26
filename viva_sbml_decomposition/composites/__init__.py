"""Decomposition-spectrum composites.

Importing this package registers every arm's ``@composite_generator`` (the
decorators fire on import) so the workbench discovers them. Arm 5 is an ensemble
*runner* (``glycolysis_ensemble`` returns per-instance results), not a single
composite document, so it exposes no generator here.
"""
from . import arm1_monolithic  # noqa: F401  glycolysis_monolithic
from . import arm2_operator_split  # noqa: F401  glycolysis_operator_split
from . import arm3_modular  # noqa: F401  glycolysis_modular
from . import arm4_native  # noqa: F401  glycolysis_native_scalar / _array
from . import arm5_ensemble  # noqa: F401  glycolysis_ensemble (runner, no generator)
from . import arm6_coupling  # noqa: F401  glycolysis_coupled
from . import arm7_multirate  # noqa: F401  glycolysis_multirate
from . import arm8_multiorgan  # noqa: F401  glycolysis_multiorgan_mono / _decomposed
from . import arm9_abm  # noqa: F401  predator_prey_abm / predator_prey_meanfield (adversarial arm)
from . import arm10_copasi  # noqa: F401  glycolysis_copasi_mono / glycolysis_cross_engine (COPASI second engine)
from . import engine_monolithic  # noqa: F401  engine_monolithic / native_composite (generic, model-as-data)
from . import reaction_split  # noqa: F401  reaction_split (generic, two-engine reaction partition)
from . import ensemble  # noqa: F401  ensemble (generic, same-rate + multi-rate #270)
# Stable names for the consolidated composite set: the ABM arm is re-exported
# unchanged (thin aliases of the arm-9 generators; arm 9 is not reimplemented).
from .arm9_abm import predator_prey_abm as abm  # noqa: F401,E402
from .arm9_abm import predator_prey_meanfield as abm_meanfield  # noqa: F401,E402
