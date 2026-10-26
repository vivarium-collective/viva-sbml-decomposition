"""Native process-bigraph processes for the SBML decomposition arms."""
from .copasi_glycolysis import CopasiProcess, register_copasi
from .native_glycolysis import (
    NativeGlycolysisProcess,
    register_native_glycolysis,
)

__all__ = [
    "NativeGlycolysisProcess",
    "register_native_glycolysis",
    "CopasiProcess",
    "register_copasi",
]
