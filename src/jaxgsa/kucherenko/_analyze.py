"""Kucherenko index estimators over the conditional-copula design.

Write ``y = X_i`` and ``z = X_{~i}``. The design supplies, for each base point
``k``: a joint row ``(y_k, z_k)``, a first-order row ``(y_k, z'_k)`` with
``z' ~ p(z | y_k)``, and a total row ``(y'_k, z_k)`` with ``y' ~ p(y | z_k)``.
The implemented estimators are the single-loop forms of Kucherenko, Tarantola
& Annoni (2012), per output slice with ``N`` base points:

.. code-block:: text

    f0_A   = (1/N) sum_k f(y_k, z_k)
    f0_B,i = (1/N) sum_k f(y_k, z'_k)
    V      = (1/N) sum_k f(y_k, z_k)^2  -  f0_A^2

    S1_i = [ (1/N) sum_k f(y_k, z_k) f(y_k, z'_k)  -  f0_A f0_B,i ] / V
    ST_i = [ (1/(2N)) sum_k ( f(y_k, z_k) - f(y'_k, z_k) )^2 ] / V

``S1_i`` estimates ``V(E(Y|X_i)) / V(Y)``. The product pairs two draws that
share ``y_k``, so its expectation is ``E[E(Y|y)^2]``. Subtracting the mean
product then leaves the variance of the conditional mean. The mean correction
uses the product of the two block means, since each block estimates ``f0``.
That cancels the shared Monte-Carlo error of the two blocks. With ``f0_A^2``
the estimator carries a strictly larger O(1/N) bias. The implementation
subtracts one shared shift (the joint-block mean) from both factors before the
product. The estimator is algebraically identical, but the arithmetic stays
centered when ``Y`` carries a large mean.

``ST_i`` estimates ``E(V(Y|X_{~i})) / V(Y)``. The squared difference pairs two
draws of ``y`` under the same ``z_k``, which is the Jansen form of the
conditional variance.

Under an identity correlation the conditional draws are plain independent
draws and both formulas reduce exactly to the classic Saltelli column-swap
estimators: the Sobol'/Homma-Saltelli form for ``S1`` and the Jansen form for
``ST``.

References:
    Kucherenko, Tarantola & Annoni (2012). Comput. Phys. Commun. 183:937-946.
    Jansen (1999). Comput. Phys. Commun. 117:35-43.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxgsa._core.validation import (
    _prepare_Y,
    _squeeze_output_axes,
    _validate_output,
    _warn_zero_variance_slices,
)
from jaxgsa.kucherenko._result import KucherenkoResult
from jaxgsa.kucherenko._sampling import KucherenkoSamples


def analyze(sampling_result: KucherenkoSamples, Y: Array) -> KucherenkoResult:
    """Compute Kucherenko first-order and total indices from model outputs.

    The estimators run on the actual model outputs. No surrogate is fitted.
    This is the design-based counterpart to :func:`jaxgsa.vkoga.analyze`. It
    estimates the same two conditional-variance quantities, but it evaluates
    the model on a conditional-copula design instead of querying a fitted
    kernel surrogate. Under independent inputs the indices are the classic
    Sobol' ``S1`` and ``ST``.

    Args:
        sampling_result: Design from :func:`jaxgsa.kucherenko.sample`.
        Y: Model outputs evaluated at each row of
            ``sampling_result.samples``, in the same row order. Accepted
            shapes: ``(n_runs,)``, ``(n_runs, K)``, or ``(n_runs, T, K)``.
            Indices are computed independently for every output slice.

    Returns:
        A :class:`KucherenkoResult` with ``S1`` and ``ST`` shaped ``(D,)``,
        ``(K, D)``, or ``(T, K, D)`` to mirror ``Y``.

    Raises:
        ValueError: If ``Y``'s shape violates the output contract.

    Warns:
        UserWarning: If any output slice has zero variance (its indices are
            NaN), or if non-finite outputs force base points to be dropped.
    """
    problem = sampling_result.problem
    N = sampling_result.base_n
    D = sampling_result.n_params
    Y = _validate_output(jnp.asarray(Y), sampling_result.n_runs, problem)

    Y_canonical, squeeze_time, squeeze_output = _prepare_Y(Y)
    n_time, n_out = Y_canonical.shape[1], Y_canonical.shape[2]
    n_slices = n_time * n_out
    # (2D+1, N, S) — block-major, matching the design's stacking order.
    F = np.asarray(Y_canonical, dtype=np.float64).reshape(2 * D + 1, N, n_slices)

    # A base point k feeds the joint row and every conditional row with the
    # same k, so one non-finite output anywhere invalidates all of them.
    finite = np.isfinite(F).all(axis=(0, 2))
    n_dropped = N - int(finite.sum())
    if n_dropped:
        warnings.warn(
            f"jaxgsa: dropped {n_dropped} of {N} base points because a model output "
            "in their group is non-finite",
            stacklevel=2,
        )
        F = F[:, finite, :]
        if F.shape[1] < 2:
            raise ValueError("Fewer than 2 base points remain after dropping non-finite outputs")

    f_joint = F[0]  # (N, S)
    f_first = F[1 : D + 1]  # (D, N, S) — z redrawn given x_i
    f_total = F[D + 1 :]  # (D, N, S) — x_i redrawn given z

    _warn_zero_variance_slices(
        f_joint.reshape(-1, n_time, n_out), output_names=problem.output_names
    )
    variance = f_joint.var(axis=0)  # (S,)
    safe_variance = np.where(variance > 0.0, variance, np.nan)

    # Shift both S1 factors by one shared constant (the joint-block mean)
    # before the product. The estimator is algebraically unchanged: for any
    # shift c, mean((a-c)(b-c)) - mean(a-c) mean(b-c) == mean(ab) - mean(a)
    # mean(b). Numerically it is decisive: the uncentered product form loses
    # the O(1) covariance in rounding when Y carries a large mean, because
    # mean(f_joint f_first) and f0_joint f0_first are both O(mean^2). The ST
    # estimator is a squared difference and the variance is a central moment,
    # so both are already shift-safe.
    f0_joint = f_joint.mean(axis=0)  # (S,)
    g_joint = f_joint - f0_joint  # (N, S), centered
    g_first = f_first - f0_joint[None, None]  # (D, N, S), same shift
    S1 = (
        (g_joint[None] * g_first).mean(axis=1) - g_joint.mean(axis=0)[None] * g_first.mean(axis=1)
    ) / safe_variance
    ST = 0.5 * ((f_joint[None] - f_total) ** 2).mean(axis=1) / safe_variance

    def _shape(index: np.ndarray) -> Array:
        """Reshape a ``(D, S)`` block to the output contract, params last."""
        arr = jnp.asarray(index.T.reshape(n_time, n_out, D))
        return _squeeze_output_axes(arr, squeeze_time, squeeze_output, n_trailing=1)

    variance_shaped = _squeeze_output_axes(
        jnp.asarray(variance.reshape(n_time, n_out)), squeeze_time, squeeze_output, n_trailing=0
    )
    return KucherenkoResult(
        S1=_shape(S1),
        ST=_shape(ST),
        problem=problem,
        variance=variance_shaped,
    )
