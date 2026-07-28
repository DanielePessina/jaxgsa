"""Generic input sampling utilities."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from jaxgsa._core.copula import (
    _safe_cholesky,
    correlation_from_covariance,
    fit_gaussian_copula,
    latent_to_physical,
)
from jaxgsa._core.sampling import _transform_samples
from jaxgsa.problem import Problem


def monte_carlo(
    problem: Problem,
    n: int,
    *,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Draw Monte Carlo samples from a problem's marginals.

    Each column follows the corresponding parameter's declared input
    distribution (uniform, Gaussian, or truncated Gaussian) via inverse-CDF
    transformation of pseudo-random draws. Unlike :func:`jaxgsa.sobol.sample`,
    this uses plain pseudo-random draws with no low-discrepancy structure and
    no Saltelli layout.

    When ``problem.correlation`` declares a (non-identity) Gaussian-copula
    correlation matrix, it is honored transparently: correlated standard
    normals are drawn on the latent scale and pushed through each marginal's
    inverse CDF (the NORTA construction), so every column keeps its declared
    marginal exactly while the joint sample carries the declared dependence
    structure. Independent problems keep the plain uniform-draw path
    bit-for-bit, so existing seeds reproduce existing samples.

    Args:
        problem: Problem definition with parameter names and distributions;
            its marginals determine how each column is transformed, and its
            optional ``correlation`` couples the columns.
        n: Number of samples (rows) to draw. Must be at least 1.
        seed: Seed for NumPy's default RNG, or an existing
            ``np.random.Generator`` to draw from. ``None`` (default) uses
            fresh OS entropy, so repeated calls give different samples;
            pass an int for reproducible draws.

    Returns:
        Array of shape ``(n, D)``, where ``D`` is the number of parameters,
        in physical units (each column transformed through the problem's
        declared marginal distribution).

    Raises:
        ValueError: If ``n`` is less than 1.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    rng = np.random.default_rng(seed)
    if not problem.has_correlated_inputs:
        return _transform_samples(problem, rng.random((n, problem.num_vars)))

    R = problem.correlation
    assert R is not None  # has_correlated_inputs implies a stored matrix
    Z = rng.standard_normal((n, problem.num_vars)) @ _safe_cholesky(R).T
    return latent_to_physical(problem, Z)


def correlate(
    X: npt.ArrayLike,
    problem: Problem,
    *,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Impose ``problem.correlation`` on an existing sample by rank re-pairing.

    Iman-Conover-style retrofit: draws a correlated latent score matrix from
    the problem's Gaussian-copula correlation, then reorders each column of
    ``X`` so its ranks match the score column's ranks. Each output column is
    an exact permutation of the corresponding input column, so the marginal
    sample values — including any structure a low-discrepancy design put into
    them — are preserved; only the pairing across columns changes.

    Use it when the per-column samples already exist (an evaluated design, an
    observational data set) and only the dependence structure is missing.
    For fresh samples prefer :func:`monte_carlo`, which draws the coupling
    directly.

    Args:
        X: ``(N, D)`` sample matrix in physical units, one column per
            parameter in ``problem.names`` order.
        problem: Problem whose ``correlation`` matrix defines the target
            dependence structure. Must have one declared (may be identity,
            which reduces to a random re-pairing).
        seed: Seed for NumPy's default RNG, or an existing
            ``np.random.Generator``; controls the latent score draw.

    Returns:
        ``(N, D)`` array with the same values per column as ``X`` (each
        column permuted) and ranks following the declared correlation.

    Raises:
        ValueError: If ``problem`` has no correlation set, or ``X`` is not
            ``(N, D)`` with ``D == problem.num_vars``.
    """
    R = problem.correlation
    if R is None:
        raise ValueError(
            "correlate() requires problem.correlation to be set; declare one at "
            "construction or attach it with problem.with_correlation(R)"
        )
    X = np.asarray(X, dtype=np.float64)
    D = problem.num_vars
    if X.ndim != 2 or X.shape[1] != D:
        raise ValueError(f"X must be (N, {D}) to match the problem, got {X.shape}")

    rng = np.random.default_rng(seed)
    scores = rng.standard_normal((X.shape[0], D)) @ _safe_cholesky(R).T
    # Rank of each score within its column (argsort twice); row i of the
    # output takes the X value holding the same within-column rank.
    ranks = np.argsort(np.argsort(scores, axis=0), axis=0)
    return np.take_along_axis(np.sort(X, axis=0), ranks, axis=0)


def fit_correlation(problem: Problem, X: npt.ArrayLike) -> np.ndarray:
    """Estimate a Gaussian-copula correlation matrix from observed samples.

    Computes the Spearman rank correlation of ``X`` and converts it to the
    latent-normal Pearson correlation via ``2 sin(pi rho_s / 6)``, repaired
    to positive definite if the rank estimate was not. Working from ranks
    keeps the estimate invariant to the declared marginals, so a heavily
    skewed parameter does not distort the dependence structure.

    The intended workflow attaches the fit to the (frozen) problem:

    .. code-block:: python

        problem = problem.with_correlation(
            jaxgsa.sampling.fit_correlation(problem, X_observed)
        )

    Heavy ties break the rank estimate: for discrete or heavily rounded
    columns the Spearman-based fit is biased toward zero, and a polychoric
    correlation would be needed instead (future work, alongside categorical
    marginals). The conversion is exact for continuous marginals only.

    Args:
        problem: Problem the samples were drawn for. Only its parameter
            count is used, but it is required so callers cannot silently
            pass a transposed matrix.
        X: ``(N, D)`` samples in physical units, at least 3 rows.

    Returns:
        ``(D, D)`` latent correlation matrix with unit diagonal, ready for
        ``problem.with_correlation``.

    Raises:
        ValueError: If ``X`` is not ``(N, D)`` with ``D == problem.num_vars``,
            or holds fewer than three rows.
    """
    return fit_gaussian_copula(problem, np.asarray(X, dtype=np.float64))


__all__ = ["correlate", "correlation_from_covariance", "fit_correlation", "monte_carlo"]
