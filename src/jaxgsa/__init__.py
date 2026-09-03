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
    sampling,
    shapley,
    sobol,
    vkoga,
)
from ._core.invalid import InvalidReport, InvalidUnit, OnInvalid
from ._core.sampling import Theta
from ._core.warning_types import JaxgsaWarning
from .problem import (
    CategoricalInputSpec,
    CategoricalSpec,
    GaussianInputSpec,
    GaussianSpec,
    InputSpec,
    Problem,
    UniformInputSpec,
    UniformSpec,
)

try:
    __version__ = _pkg_version("jaxgsa")
except PackageNotFoundError:  # running from a source tree without the package
    __version__ = "unknown"

__all__ = [
    "CategoricalInputSpec",
    "CategoricalSpec",
    "GaussianInputSpec",
    "GaussianSpec",
    "InputSpec",
    "InvalidReport",
    "InvalidUnit",
    "JaxgsaWarning",
    "OnInvalid",
    "Problem",
    "Theta",
    "UniformInputSpec",
    "UniformSpec",
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
    "sampling",
    "shapley",
    "sobol",
    "vkoga",
]
