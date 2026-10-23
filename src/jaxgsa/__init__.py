"""Global sensitivity analysis in JAX."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from . import (
    benchmarks,
    borgonovo,
    config,
    dgsm,
    efast,
    hdmr,
    hsic,
    kucherenko,
    morris,
    optimal_transport,
    pawn,
    pce,
    rank,
    sampling,
    shapley,
    sobol,
    vkoga,
)
from ._core.invalid import InvalidReport, InvalidUnit, OnInvalid
from ._core.warning_types import JaxgsaWarning
from .problem import (
    BetaInputSpec,
    BetaSpec,
    CategoricalInputSpec,
    CategoricalSpec,
    GaussianInputSpec,
    GaussianSpec,
    InputSpec,
    LognormalInputSpec,
    LognormalSpec,
    Problem,
    TriangularInputSpec,
    TriangularSpec,
    UniformInputSpec,
    UniformSpec,
    WeibullInputSpec,
    WeibullSpec,
)

try:
    __version__ = _pkg_version("jaxgsa")
except PackageNotFoundError:  # running from a source tree without the package
    __version__ = "unknown"

__all__ = [
    "BetaInputSpec",
    "BetaSpec",
    "CategoricalInputSpec",
    "CategoricalSpec",
    "GaussianInputSpec",
    "GaussianSpec",
    "InputSpec",
    "InvalidReport",
    "InvalidUnit",
    "JaxgsaWarning",
    "LognormalInputSpec",
    "LognormalSpec",
    "OnInvalid",
    "Problem",
    "TriangularInputSpec",
    "TriangularSpec",
    "UniformInputSpec",
    "UniformSpec",
    "WeibullInputSpec",
    "WeibullSpec",
    "__version__",
    "benchmarks",
    "borgonovo",
    "config",
    "dgsm",
    "efast",
    "hdmr",
    "hsic",
    "kucherenko",
    "morris",
    "optimal_transport",
    "pawn",
    "pce",
    "rank",
    "sampling",
    "shapley",
    "sobol",
    "vkoga",
]
