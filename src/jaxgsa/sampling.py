"""Generic input sampling utilities."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

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

    A ``problem.correlation`` that declares a non-identity Gaussian-copula
    correlation matrix is honored transparently. Correlated standard normals
    are drawn on the latent scale, then pushed through each marginal's
    inverse CDF. This is the NORTA construction: every column keeps its
    declared marginal exactly, and the joint sample carries the declared
    dependence structure. Independent problems keep the plain uniform-draw
    path bit-for-bit, so existing seeds reproduce existing samples.

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

    The Iman-Conover method (Iman & Conover 1982). It builds a score matrix
    ``M`` from van der Waerden scores, ``Phi^{-1}(i / (N + 1))``, independently
    permuted per column. It then maps ``M`` through
    ``chol(R) chol(corr(M))^{-1}`` so the score matrix carries the target
    correlation ``R`` exactly, not just in expectation. Finally it reorders
    each column of ``X`` so its ranks match the score column's ranks.

    The ``corr(M)^{-1/2}`` step is what separates the method from a plain
    correlated normal draw. It removes the sampling noise of the finite score
    matrix. Without it the achieved rank correlation of the output scatters
    around the target about 2.5 times as widely, and it carries a small
    negative bias. This matters because ``correlate`` exists to retrofit a
    target onto a finite design, which is exactly where that variance
    reduction pays.

    Each output column is an exact permutation of the corresponding input
    column. The marginal sample values are therefore preserved, including any
    structure a low-discrepancy design put into them. Only the pairing across
    columns changes.

    Use it when the per-column samples already exist (an evaluated design, an
    observational data set) and only the dependence structure is missing.
    For fresh samples prefer :func:`monte_carlo`, which draws the coupling
    directly.

    References:
        Iman & Conover (1982). Commun. Stat. Simul. Comput. 11(3):311-334.

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
    scores = _iman_conover_scores(X.shape[0], R, rng)
    # Rank of each score within its column (argsort twice); row i of the
    # output takes the X value holding the same within-column rank.
    ranks = np.argsort(np.argsort(scores, axis=0), axis=0)
    return np.take_along_axis(np.sort(X, axis=0), ranks, axis=0)


def _iman_conover_scores(n: int, R: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Build the Iman-Conover score matrix whose sample correlation is ``R``.

    Args:
        n: Number of rows.
        R: ``(D, D)`` target latent correlation matrix, positive definite.
        rng: Generator supplying the column permutations.

    Returns:
        ``(n, D)`` score matrix. Every column is a permutation of the same van
        der Waerden scores, so every column has the identical marginal; the
        Pearson correlation of the columns equals ``R`` up to the rounding of
        the linear map.
    """
    D = R.shape[0]
    # Van der Waerden scores: the expected order statistics of a standard
    # normal, up to the usual i / (n + 1) plotting position. Using a fixed
    # score set rather than a fresh normal draw removes one source of noise.
    scores = norm.ppf(np.arange(1, n + 1) / (n + 1))
    M = np.empty((n, D), dtype=np.float64)
    for j in range(D):
        M[:, j] = rng.permutation(scores)

    # corr(M) is singular when there are fewer rows than columns, and undefined
    # for a single row. Those designs carry no usable rank structure anyway, so
    # skip the de-correlation and fall back to the raw score matrix.
    if n <= D or n < 3:
        return M

    T = np.atleast_2d(np.corrcoef(M, rowvar=False))
    # M @ (chol(R) chol(T)^{-1}).T == M @ chol(T)^{-T} chol(R).T.
    return M @ np.linalg.solve(_safe_cholesky(T).T, _safe_cholesky(R).T)


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

    Tied values get average ranks, which is the Spearman convention
    (``scipy.stats.spearmanr``). Average ranks reduce the tie problem but
    do not remove it. For heavily discrete or rounded columns the fit is
    still biased toward zero. Such data needs a polychoric correlation
    instead (future work). The conversion is exact for continuous
    marginals only.

    Categorical parameters are excluded from the fit: their level codes
    carry no order, so a rank correlation over them would depend on the
    arbitrary code assignment. Their rows and columns come back as exact
    identity (independent), with one ``JaxgsaWarning`` naming them.
    Polychoric estimation is future work.

    Args:
        problem: Problem the samples were drawn for. Its parameter count
            shapes the fit, and its categorical parameters are excluded
            from it.
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
