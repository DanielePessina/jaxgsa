"""gsax: global sensitivity analysis in JAX.

Typical workflow:

1. Define a :class:`Problem` naming each model input and its distribution.
2. Draw input samples — :func:`sample` for Sobol/Saltelli designs,
   :func:`sample_morris` for Morris screening, :func:`sample_mc` for plain
   Monte Carlo. Given-data methods (PAWN, HSIC, Borgonovo, ...) accept any
   ``(N, D)`` input matrix, so no dedicated sampler is needed.
3. Evaluate your model at the returned samples to obtain outputs ``Y``.
4. Call the matching analysis function: :func:`analyze` for Sobol indices,
   or one of the ``analyze_*`` entry points for the other methods.

Every analysis returns a result dataclass (``SAResult``, ``MorrisResult``,
...) with a ``to_dataset()`` method for labeled xarray export.
"""

from gsax._config import enable_compilation_cache
from gsax.borgonovo import DeltaResult
from gsax.borgonovo import analyze as analyze_borgonovo
from gsax.dgsm import DGSMResult
from gsax.dgsm import analyze as analyze_dgsm
from gsax.efast import EFASTResult
from gsax.efast import analyze as analyze_efast
from gsax.efast import sample as sample_efast
from gsax.hdmr import HDMREmulator, HDMRResult
from gsax.hdmr import analyze as analyze_hdmr
from gsax.hdmr import emulate as emulate_hdmr
from gsax.hsic import HSICResult
from gsax.hsic import analyze as analyze_hsic
from gsax.morris import MorrisResult, MorrisSamplingResult
from gsax.morris import analyze as analyze_morris
from gsax.morris import sample as sample_morris
from gsax.pawn import PAWNResult
from gsax.pawn import analyze as analyze_pawn
from gsax.pce import PCEResult
from gsax.pce import analyze as analyze_pce
from gsax.pce import emulate as emulate_pce
from gsax.problem import GaussianInputSpec, Problem, UniformInputSpec
from gsax.sampling import SamplingResult, downsample, load, sample, sample_mc, verify_prefix
from gsax.shapley import ShapleyResult
from gsax.shapley import analyze as analyze_shapley
from gsax.sobol import SAResult, analyze

__all__ = [
    "DGSMResult",
    "DeltaResult",
    "EFASTResult",
    "GaussianInputSpec",
    "HDMREmulator",
    "HDMRResult",
    "HSICResult",
    "MorrisResult",
    "MorrisSamplingResult",
    "PAWNResult",
    "PCEResult",
    "Problem",
    "SAResult",
    "SamplingResult",
    "ShapleyResult",
    "UniformInputSpec",
    "analyze",
    "analyze_borgonovo",
    "analyze_dgsm",
    "analyze_efast",
    "analyze_hdmr",
    "analyze_hsic",
    "analyze_morris",
    "analyze_pawn",
    "analyze_pce",
    "analyze_shapley",
    "downsample",
    "emulate_hdmr",
    "emulate_pce",
    "enable_compilation_cache",
    "load",
    "sample",
    "sample_efast",
    "sample_mc",
    "sample_morris",
    "verify_prefix",
]
