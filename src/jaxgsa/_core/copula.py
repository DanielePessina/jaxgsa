"""Gaussian-copula dependency structure for correlated-input analyses.

Every sampling design in jaxgsa draws independent marginals, so dependence has
to be introduced explicitly. A Gaussian copula does that with the least
commitment: it keeps each parameter's declared marginal exactly as the user
wrote it and adds a rank-correlation structure on top, which is what Hilhorst
et al. (2024) use to map independent nodes into the correlated input space.

The latent space is standard normal. Writing ``R`` for the copula correlation
matrix, ``Z ~ N(0, R)``, ``U = Phi(Z)`` and ``X_i = F_i^{-1}(U_i)``. All
conditioning is done on ``Z`` where the conditionals are Gaussian and available
in closed form; the marginal transform is applied only at the very end.

References:
    Hilhorst, Quicken, van de Vosse & Huberts (2024). Int. J. Numer. Meth.
        Biomed. Engng. 40(2):e3797.
    Li, Rabitz, Yelvington et al. (2010). J. Phys. Chem. A 114:6022-6032.
"""

from __future__ import annotations

import warnings
from typing import Literal, NamedTuple

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from scipy.stats import norm, qmc

from jaxgsa._core.sampling import _transform_samples
from jaxgsa._core.transforms import cdf_to_unit_interval
from jaxgsa.problem import Problem

# Eigenvalues below this are lifted when repairing a non-PD correlation matrix.
# Large enough to survive a float32 Cholesky downstream, small enough not to
# distort a matrix that was only marginally indefinite.
_MIN_EIGENVALUE = 1e-8


class _ConditionalPlan(NamedTuple):
    """Closed-form Gaussian conditionals used by the correlated estimators.

    All fields are indexed by the conditioning parameter ``i`` and expressed in
    the latent standard-normal space, where ``R[i, i] == 1``.

    Attributes:
        others: ``(D, D - 1)`` integer index of the parameters other than ``i``,
            in ascending order.
        beta_rest: ``(D, D - 1)`` regression of ``Z_-i`` on ``Z_i``, i.e.
            ``R[-i, i]``. Used to draw ``Z_-i | Z_i``.
        chol_rest: ``(D, D - 1, D - 1)`` Cholesky factor of the conditional
            covariance ``R[-i, -i] - R[-i, i] R[i, -i]``.
        beta_self: ``(D, D - 1)`` regression of ``Z_i`` on ``Z_-i``, i.e.
            ``R[i, -i] R[-i, -i]^{-1}``. Used to draw ``Z_i | Z_-i``.
        std_self: ``(D,)`` conditional standard deviation of ``Z_i | Z_-i``.
        chol_marginal: ``(D, D - 1, D - 1)`` Cholesky factor of ``R[-i, -i]``,
            for drawing the ``Z_-i`` outer sample of the total-uncorrelated
            index.
    """

    others: np.ndarray
    beta_rest: np.ndarray
    chol_rest: np.ndarray
    beta_self: np.ndarray
    std_self: np.ndarray
    chol_marginal: np.ndarray


def independent_correlation(n_params: int) -> np.ndarray:
    """Return the identity correlation matrix for ``n_params`` parameters."""
    return np.eye(n_params, dtype=np.float64)


def _spearman_to_latent(R: np.ndarray) -> np.ndarray:
    """Convert a Spearman rank-correlation matrix to the latent Pearson one.

    Under a Gaussian copula the Pearson correlation of the latent normals that
    reproduces a Spearman rank correlation ``rho_s`` is ``2 sin(pi rho_s / 6)``
    (Kruskal 1958). The conversion is exact for continuous marginals and maps
    ``[-1, 1]`` onto itself, so validity checks may run on either scale; it is
    not guaranteed to preserve positive definiteness, so callers must
    re-validate the result.

    Args:
        R: ``(D, D)`` Spearman rank-correlation matrix.

    Returns:
        The equivalent latent-normal Pearson correlation matrix, with the
        diagonal pinned to exactly 1.
    """
    latent = 2.0 * np.sin(np.pi * np.asarray(R, dtype=np.float64) / 6.0)
    np.fill_diagonal(latent, 1.0)
    return latent


def canonicalize_correlation(
    correlation: npt.ArrayLike,
    n_params: int,
    *,
    kind: Literal["latent", "spearman"] = "latent",
    warn_on_repair: bool = False,
) -> np.ndarray:
    """Normalize a user-supplied correlation matrix to the latent scale.

    Single entry point behind every surface that accepts a correlation
    matrix: checks the shape against the problem dimension, converts a
    Spearman matrix to the latent scale, and runs the full structural and
    positive-definiteness validation. The Spearman conversion maps
    ``[-1, 1]`` onto itself monotonically and preserves symmetry and the
    unit diagonal, so validating after conversion catches exactly the same
    structural errors as validating before it.

    Args:
        correlation: ``(D, D)`` candidate correlation matrix.
        n_params: Expected ``D``.
        kind: Scale the matrix is expressed on. ``"latent"`` (default) is the
            Pearson correlation of the latent normals; ``"spearman"`` is a
            rank correlation, converted via ``2 sin(pi rho_s / 6)``.
        warn_on_repair: Forwarded to :func:`validate_correlation`.

    Returns:
        ``(D, D)`` validated latent correlation matrix.

    Raises:
        ValueError: If ``kind`` is unknown or the matrix fails validation.
    """
    if kind not in ("latent", "spearman"):
        raise ValueError(f"correlation_kind must be 'latent' or 'spearman', got {kind!r}")
    R = np.asarray(correlation, dtype=np.float64)
    if R.shape != (n_params, n_params):
        raise ValueError(f"correlation must be ({n_params}, {n_params}), got {R.shape}")
    if kind == "spearman":
        # The conversion is not guaranteed to preserve positive definiteness,
        # so the full validation (including PD repair) runs on the result.
        R = _spearman_to_latent(R)
    return validate_correlation(R, n_params, warn_on_repair=warn_on_repair)


def correlation_from_covariance(cov: npt.ArrayLike) -> np.ndarray:
    """Extract the correlation matrix from a published covariance matrix.

    The common literature case supplies a full covariance ``Sigma`` rather
    than a correlation matrix. This rescales it to unit diagonal,
    ``R = D^{-1} Sigma D^{-1}`` with ``D = diag(sqrt(diag(Sigma)))``, so the
    result can be handed to ``Problem.with_correlation`` or the ``correlation``
    argument of the :class:`~jaxgsa.problem.Problem` constructors.

    Two caveats the caller must own:

    - The result is the **Pearson** correlation of the physical variables.
      Under a Gaussian copula that equals the latent correlation only when
      every marginal is Gaussian; for non-Gaussian marginals prefer a rank
      correlation with ``correlation_kind="spearman"``, which is exactly
      invertible. Pearson matching for arbitrary marginals is the NORTA
      correlation-matching problem and is not attempted here.
    - Variances always come from the **declared marginals** on the
      ``Problem``, never from the covariance diagonal, so nothing is
      specified twice. If the supplied diagonal disagrees with a declared
      ``GaussianInputSpec.variance``, the diagonal is silently discarded —
      this function cannot see the ``Problem``, so reconciling the two is
      the caller's responsibility.

    Args:
        cov: ``(D, D)`` covariance matrix — square, symmetric, with a
            strictly positive diagonal.

    Returns:
        ``(D, D)`` correlation matrix with the diagonal pinned to exactly 1
        (so it survives ``validate_correlation``'s unit-diagonal check).

    Raises:
        ValueError: If ``cov`` is not square, not symmetric, or its diagonal
            is not strictly positive.
    """
    cov = np.asarray(cov, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"covariance must be a square (D, D) matrix, got shape {cov.shape}")
    if not np.allclose(cov, cov.T, atol=1e-10):
        raise ValueError("covariance must be symmetric")
    diagonal = np.diag(cov)
    if np.any(diagonal <= 0):
        raise ValueError(f"covariance diagonal must be strictly positive, got {diagonal!r}")

    d = 1.0 / np.sqrt(diagonal)
    R = d[:, None] * cov * d[None, :]
    np.fill_diagonal(R, 1.0)
    return R


def fit_gaussian_copula(problem: Problem, X: np.ndarray) -> np.ndarray:
    """Estimate a Gaussian-copula correlation matrix from a design matrix.

    Uses Spearman rank correlation converted to the equivalent Pearson
    correlation of the latent normals, ``rho = 2 sin(pi rho_s / 6)``. Working
    from ranks rather than raw values keeps the estimate invariant to the
    declared marginals, so a heavily skewed parameter does not distort the
    dependency structure.

    Args:
        problem: Problem the samples were drawn for. Only its parameter count
            is used, but it is required so callers cannot silently pass a
            transposed matrix.
        X: ``(N, D)`` samples in physical units.

    Returns:
        ``(D, D)`` correlation matrix with unit diagonal, repaired to be
        positive definite if the rank estimate was not.

    Raises:
        ValueError: If ``X`` is not ``(N, D)`` with ``D == problem.num_vars``,
            or holds fewer than three rows.
    """
    X = np.asarray(X, dtype=np.float64)
    D = problem.num_vars
    if X.ndim != 2 or X.shape[1] != D:
        raise ValueError(f"X must be (N, {D}) to fit a copula for this problem, got {X.shape}")
    if X.shape[0] < 3:
        raise ValueError(f"Fitting a copula needs at least 3 samples, got {X.shape[0]}")

    # Rank-transform each column independently, then correlate. np.argsort twice
    # gives ranks; ties are broken arbitrarily, which is harmless for continuous
    # marginals and is what a copula fit assumes anyway.
    ranks = np.argsort(np.argsort(X, axis=0), axis=0).astype(np.float64)
    spearman = np.corrcoef(ranks, rowvar=False)
    if D == 1:  # np.corrcoef collapses to a scalar for a single column
        spearman = np.atleast_2d(spearman)

    # Pearson correlation of the latent normals that reproduces this Spearman
    # rank correlation under a Gaussian copula.
    return _project_to_correlation(_spearman_to_latent(spearman))


def validate_correlation(
    R: np.ndarray,
    n_params: int,
    *,
    warn_on_repair: bool = False,
) -> np.ndarray:
    """Validate a user-supplied correlation matrix and make it usable.

    Args:
        R: ``(D, D)`` candidate correlation matrix.
        n_params: Expected ``D``.
        warn_on_repair: Emit a ``UserWarning`` when the positive-definiteness
            repair actually alters the matrix. A materially indefinite ``R``
            usually signals a modelling error (inconsistent pairwise
            correlations or a redundant column), so surfaces that accept
            user-declared matrices should opt in; internal fitting paths stay
            silent.

    Returns:
        A symmetric, positive-definite copy with unit diagonal.

    Raises:
        ValueError: If the shape is wrong, the matrix is not symmetric, the
            diagonal is not unit, or any entry lies outside ``[-1, 1]``.
    """
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (n_params, n_params):
        raise ValueError(f"correlation must be ({n_params}, {n_params}), got {R.shape}")
    if not np.allclose(R, R.T, atol=1e-10):
        raise ValueError("correlation must be symmetric")
    if not np.allclose(np.diag(R), 1.0, atol=1e-10):
        raise ValueError("correlation must have a unit diagonal")
    if np.abs(R).max() > 1.0 + 1e-10:
        raise ValueError("correlation entries must lie in [-1, 1]")
    return _project_to_correlation(R, warn_on_repair=warn_on_repair)


def _project_to_correlation(R: np.ndarray, *, warn_on_repair: bool = False) -> np.ndarray:
    """Return the nearest positive-definite matrix with a unit diagonal.

    A rank-correlation estimate transformed entry-by-entry through
    ``2 sin(pi rho_s / 6)`` is not guaranteed to stay positive definite, and a
    user-declared matrix need not be either. Clipping the eigenvalues and
    renormalising the diagonal is the standard repair; it is a no-op when the
    input is already valid, so the common case pays only one eigendecomposition.

    Args:
        R: ``(D, D)`` candidate correlation matrix (structurally valid).
        warn_on_repair: Emit a ``UserWarning`` reporting the minimum
            eigenvalue and the maximum entrywise change when the repair
            actually engages.
    """
    R = 0.5 * (R + R.T)  # kill any asymmetry from floating-point accumulation
    eigenvalues, eigenvectors = np.linalg.eigh(R)
    if eigenvalues.min() >= _MIN_EIGENVALUE:
        return R

    clipped = eigenvectors @ np.diag(np.maximum(eigenvalues, _MIN_EIGENVALUE)) @ eigenvectors.T
    # Renormalise so the diagonal is exactly 1 again -- eigenvalue clipping
    # perturbs it, and every conditional formula below assumes R[i, i] == 1.
    scale = np.sqrt(np.diag(clipped))
    repaired = clipped / np.outer(scale, scale)
    np.fill_diagonal(repaired, 1.0)
    if warn_on_repair:
        warnings.warn(
            "jaxgsa: the declared correlation matrix is not positive definite "
            f"(minimum eigenvalue {eigenvalues.min():.3e}) and was repaired by "
            "eigenvalue clipping; the largest entrywise change is "
            f"{np.abs(repaired - R).max():.3e}. Samples will follow the repaired "
            "dependence structure, not the declared one — check the matrix for "
            "inconsistent pairwise correlations or redundant parameters.",
            stacklevel=2,
        )
    return repaired


def is_independent(R: np.ndarray) -> bool:
    """Return ``True`` when ``R`` is the identity to floating-point tolerance."""
    return bool(np.allclose(R, np.eye(R.shape[0]), atol=1e-12))


def build_conditional_plan(R: np.ndarray) -> _ConditionalPlan:
    """Precompute every Gaussian conditional the correlated indices need.

    Both conditioning directions are needed once per parameter and reused for
    every Monte-Carlo draw, so they are built up front in float64 rather than
    re-derived inside the estimator loop.

    Args:
        R: ``(D, D)`` positive-definite correlation matrix with unit diagonal.

    Returns:
        A :class:`_ConditionalPlan` with all quantities stacked over parameters.

    Raises:
        ValueError: If ``D < 2``; conditioning on "the other parameters" is
            meaningless for a single-parameter problem.
    """
    D = R.shape[0]
    if D < 2:
        raise ValueError("Correlated sensitivity indices need at least 2 parameters")

    others = np.empty((D, D - 1), dtype=np.int64)
    beta_rest = np.empty((D, D - 1), dtype=np.float64)
    chol_rest = np.empty((D, D - 1, D - 1), dtype=np.float64)
    beta_self = np.empty((D, D - 1), dtype=np.float64)
    std_self = np.empty(D, dtype=np.float64)
    chol_marginal = np.empty((D, D - 1, D - 1), dtype=np.float64)

    for i in range(D):
        rest = np.array([j for j in range(D) if j != i], dtype=np.int64)
        others[i] = rest
        R_rest = R[np.ix_(rest, rest)]
        r_cross = R[rest, i]

        # Z_-i | Z_i = z: mean r_cross * z (R[i, i] == 1), covariance the Schur
        # complement. Jitter guards the Cholesky when a parameter is nearly a
        # deterministic function of the others.
        beta_rest[i] = r_cross
        cov_rest = R_rest - np.outer(r_cross, r_cross)
        chol_rest[i] = _safe_cholesky(cov_rest)

        # Z_i | Z_-i: ordinary least squares of Z_i on the others.
        solved = np.linalg.solve(R_rest, r_cross)
        beta_self[i] = solved
        residual_var = max(1.0 - float(r_cross @ solved), _MIN_EIGENVALUE)
        std_self[i] = np.sqrt(residual_var)

        chol_marginal[i] = _safe_cholesky(R_rest)

    return _ConditionalPlan(
        others=others,
        beta_rest=beta_rest,
        chol_rest=chol_rest,
        beta_self=beta_self,
        std_self=std_self,
        chol_marginal=chol_marginal,
    )


def _safe_cholesky(cov: np.ndarray) -> np.ndarray:
    """Cholesky factor of a covariance, repaired if it is not quite PD."""
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        # Conditional covariances lose rank as correlations approach +-1. Lift
        # the spectrum rather than failing: the caller is sampling, not solving.
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (cov + cov.T))
        lifted = eigenvectors @ np.diag(np.maximum(eigenvalues, _MIN_EIGENVALUE)) @ eigenvectors.T
        return np.linalg.cholesky(lifted)


def latent_normal_sample(n: int, dim: int, *, seed: int, scramble: bool = True) -> np.ndarray:
    """Draw ``(n, dim)`` standard normal variates from a scrambled Sobol' set.

    Quasi-random draws are used throughout because every integrand here is
    evaluated on a cheap surrogate, so the sample size is limited by memory
    rather than model cost and the better equidistribution is free.

    Args:
        n: Number of points. Rounded up to the next power of two internally by
            the Sobol' engine's balance properties; passed through as-is.
        dim: Dimension of each point.
        seed: Seed for the scrambling.
        scramble: Whether to scramble the sequence. Unscrambled Sobol' starts
            at the origin, which maps to an infinite normal deviate.

    Returns:
        ``(n, dim)`` array of standard normal variates.
    """
    engine = qmc.Sobol(d=dim, scramble=scramble, seed=seed)
    unit = engine.random(n)
    # Sobol' points can land on exactly 0 or 1 when unscrambled; clip before the
    # probit so the deviates stay finite.
    return norm.ppf(np.clip(unit, 1e-12, 1.0 - 1e-12))


def latent_to_physical(problem: Problem, Z: np.ndarray) -> np.ndarray:
    """Map latent normal variates to physical units through the marginals.

    Args:
        problem: Problem supplying the marginal distributions.
        Z: ``(N, D)`` latent standard-normal variates with the copula's
            correlation structure already applied.

    Returns:
        ``(N, D)`` samples in physical units.
    """
    unit = norm.cdf(np.asarray(Z, dtype=np.float64))
    # The marginal inverse CDFs reject exact 0/1; the same clip is applied on
    # the forward path in _core.sampling.
    unit = np.clip(unit, 1e-12, 1.0 - 1e-12)
    return _transform_samples(problem, unit)


def physical_to_latent(problem: Problem, X: np.ndarray) -> np.ndarray:
    """Map physical samples back to the latent normal space.

    Inverse of :func:`latent_to_physical`, used to place a user-supplied design
    matrix in the space where the copula conditionals are defined.

    Args:
        problem: Problem supplying the marginal distributions.
        X: ``(N, D)`` samples in physical units.

    Returns:
        ``(N, D)`` latent standard-normal variates.
    """
    unit = np.asarray(cdf_to_unit_interval(jnp.asarray(X), problem), dtype=np.float64)
    return norm.ppf(np.clip(unit, 1e-12, 1.0 - 1e-12))
