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

import numpy as np
import numpy.typing as npt
from scipy.stats import norm, qmc, rankdata

from jaxgsa._core.sampling import _transform_samples
from jaxgsa.problem import Problem

# Eigenvalues below this are lifted when repairing a non-PD correlation matrix.
# Everything on this path is float64. The floor only has to keep the downstream
# float64 Cholesky away from a zero pivot, so it stays small enough not to
# distort a matrix that was only marginally indefinite.
_MIN_EIGENVALUE = 1e-8

# Eigenvalues are lifted to this, not to the acceptance floor itself.
# Renormalising the diagonal afterwards perturbs the spectrum, and clipping to
# exactly _MIN_EIGENVALUE leaves the result sitting on the comparison boundary:
# whether it then reads as >= the floor comes down to one unit in the last
# place, which differs between LAPACK builds. Lifting to a small multiple puts
# the converged minimum at 1.6x-4x the floor on every platform, and cuts the
# loop from three or four passes to two.
_EIGENVALUE_CLIP_TARGET = 4.0 * _MIN_EIGENVALUE

# Clipping the eigenvalues and then renormalising the diagonal can push the
# smallest eigenvalue back under the floor, so the repair runs again. Two
# passes are enough in practice; the cap only bounds a pathological input.
_MAX_REPAIR_PASSES = 16

# The severity of a repair is the maximum entrywise change,
# ``max|R_repaired - R_declared|``, measured on the scale the caller declared.
# That number is scale-free and reads directly ("your 0.9 became 0.72"),
# unlike a minimum eigenvalue, which means very different things at D = 3 and
# at D = 50. The minimum eigenvalue stays as the numerical floor above and as
# a diagnostic in the message.
#
# Below this the repair only removed floating-point noise, so nothing is said.
# It has to clear the largest change a pure floor-lift can make, which is a
# little under _EIGENVALUE_CLIP_TARGET (measured 3.9e-8 for a near-singular
# 3x3), with room to spare. A correlation entry moving by 1e-6 is negligible
# by any reading.
_REPAIR_NOISE = 1e-6
# At or above this the declared matrix is structurally inconsistent, not merely
# rounded, and the two policies below diverge.
_REPAIR_MATERIAL = 0.05

# Policy applied when the repair actually engages.
#
# ``"declared"``  the matrix came from the user. A noise-level repair is
#     silent, a small repair warns, and a material repair raises: the user can
#     fix the matrix or fit one from data.
# ``"fitted"``    the matrix came from our own estimator. A small repair is
#     silent and a material repair warns. It never raises, because refusing to
#     fit helps nobody when it is the data that is inconsistent.
RepairPolicy = Literal["declared", "fitted"]


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
    ``[-1, 1]`` onto itself. It is not guaranteed to preserve positive
    definiteness, so callers must run the positive-definiteness check on the
    result. Structural checks must run on the input instead: the conversion
    rewrites every entry, so it would mask a bad declared matrix.

    Args:
        R: ``(D, D)`` Spearman rank-correlation matrix.

    Returns:
        The equivalent latent-normal Pearson correlation matrix, with the
        diagonal pinned to exactly 1.
    """
    latent = 2.0 * np.sin(np.pi * np.asarray(R, dtype=np.float64) / 6.0)
    np.fill_diagonal(latent, 1.0)
    return latent


def _latent_to_spearman(R: np.ndarray) -> np.ndarray:
    """Convert a latent Pearson correlation matrix back to the Spearman scale.

    Exact inverse of :func:`_spearman_to_latent`,
    ``rho_s = (6 / pi) arcsin(rho / 2)`` (Kruskal 1958). It exists so a repair
    of a Spearman-declared matrix can be reported in the units the user wrote.

    Args:
        R: ``(D, D)`` latent-normal Pearson correlation matrix.

    Returns:
        The equivalent Spearman rank-correlation matrix, with the diagonal
        pinned to exactly 1.
    """
    spearman = (6.0 / np.pi) * np.arcsin(np.clip(np.asarray(R, dtype=np.float64), -2.0, 2.0) / 2.0)
    np.fill_diagonal(spearman, 1.0)
    return spearman


def canonicalize_correlation(
    correlation: npt.ArrayLike,
    n_params: int,
    *,
    kind: Literal["latent", "spearman"] = "latent",
    policy: RepairPolicy = "fitted",
) -> np.ndarray:
    """Normalize a user-supplied correlation matrix to the latent scale.

    Single entry point behind every surface that accepts a correlation
    matrix. The structural checks (shape, symmetry, unit diagonal, entry
    range) run on the matrix as the user declared it, before any conversion.
    A Spearman matrix is only then converted to the latent scale. The
    positive-definiteness check and repair run last, on the latent matrix,
    because the conversion can destroy positive definiteness.

    The order matters. The conversion rewrites every entry and pins the
    diagonal to exactly 1, so a check made after it cannot see a wrong
    declared diagonal, and it would report any other error against numbers
    the user never wrote.

    Args:
        correlation: ``(D, D)`` candidate correlation matrix.
        n_params: Expected ``D``.
        kind: Scale the matrix is expressed on. ``"latent"`` (default) is the
            Pearson correlation of the latent normals; ``"spearman"`` is a
            rank correlation, converted via ``2 sin(pi rho_s / 6)``. The
            repair also reports its severity on this scale.
        policy: Repair policy, see :data:`RepairPolicy`.

    Returns:
        ``(D, D)`` validated latent correlation matrix.

    Raises:
        ValueError: If ``kind`` is unknown, the matrix fails validation, or
            the repair of a declared matrix is material.
    """
    if kind not in ("latent", "spearman"):
        raise ValueError(f"correlation_kind must be 'latent' or 'spearman', got {kind!r}")
    R = np.asarray(correlation, dtype=np.float64)
    _validate_structure(R, n_params)
    if kind == "spearman":
        # The conversion is not guaranteed to preserve positive definiteness,
        # so the repair still runs on the result.
        R = _spearman_to_latent(R)
    return _project_to_correlation(R, policy=policy, report_kind=kind)


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


def _force_categorical_identity(R: np.ndarray, cat_dims: "list[int]") -> np.ndarray:
    """Reset the categorical rows and columns of ``R`` to exact identity.

    Used after a fit or a positive-definiteness repair, both of which leave
    float-level noise in rows that must be exactly decoupled. The reset
    preserves positive definiteness: the result is block-diagonal, made of
    an identity block and a principal submatrix of the (positive-definite)
    input.

    Args:
        R: ``(D, D)`` correlation matrix.
        cat_dims: Indices of the categorical parameters.

    Returns:
        A copy of ``R`` with identity rows and columns at ``cat_dims``
        (``R`` itself when ``cat_dims`` is empty).
    """
    if not cat_dims:
        return R
    R = R.copy()
    R[cat_dims, :] = 0.0
    R[:, cat_dims] = 0.0
    R[cat_dims, cat_dims] = 1.0
    return R


def fit_gaussian_copula(problem: Problem, X: np.ndarray) -> np.ndarray:
    """Estimate a Gaussian-copula correlation matrix from a design matrix.

    Uses Spearman rank correlation converted to the equivalent Pearson
    correlation of the latent normals, ``rho = 2 sin(pi rho_s / 6)``. Working
    from ranks rather than raw values keeps the estimate invariant to the
    declared marginals, so a heavily skewed parameter does not distort the
    dependency structure. Tied values get average ranks, which is the
    Spearman convention (``scipy.stats.spearmanr``).

    Categorical parameters have unordered level codes, so a rank correlation
    over them would depend on the arbitrary code order. Their rows and
    columns are forced to exact identity (independent) and one
    ``UserWarning`` names them; polychoric estimation is future work. The
    continuous pairs are fitted normally.

    Args:
        problem: Problem the samples were drawn for. Its parameter count
            shapes the fit, and its categorical parameters are excluded
            from it.
        X: ``(N, D)`` samples in physical units.

    Returns:
        ``(D, D)`` correlation matrix with unit diagonal, repaired to be
        positive definite if the rank estimate was not.

    Raises:
        ValueError: If ``X`` is not ``(N, D)`` with ``D == problem.num_vars``,
            or holds fewer than three rows.
    """
    from jaxgsa.problem import _categorical_dims

    X = np.asarray(X, dtype=np.float64)
    D = problem.num_vars
    if X.ndim != 2 or X.shape[1] != D:
        raise ValueError(f"X must be (N, {D}) to fit a copula for this problem, got {X.shape}")
    if X.shape[0] < 3:
        raise ValueError(f"Fitting a copula needs at least 3 samples, got {X.shape[0]}")

    # Rank-transform each column independently, then correlate. Average ranks
    # follow the Spearman convention for tied values. Position-dependent tie
    # breaking would bias the estimate on discrete or quantized columns.
    ranks = rankdata(X, method="average", axis=0)
    spearman = np.corrcoef(ranks, rowvar=False)
    if D == 1:  # np.corrcoef collapses to a scalar for a single column
        spearman = np.atleast_2d(spearman)

    cat_dims = [d for d, _ in _categorical_dims(problem)]
    latent = _spearman_to_latent(spearman)
    if cat_dims:
        # Level codes carry no order, so their Spearman numbers are
        # artifacts of the code assignment (relabeling flips them).
        warnings.warn(
            f"jaxgsa: parameters {[problem.names[d] for d in cat_dims]} are "
            "categorical; a rank correlation over unordered level codes "
            "depends on the arbitrary code order, so their rows and columns "
            "are kept at identity (independent). Polychoric estimation is "
            "future work.",
            # The public entry point is jaxgsa.sampling.fit_correlation.
            stacklevel=3,
        )
        latent = _force_categorical_identity(latent, cat_dims)

    # Pearson correlation of the latent normals that reproduces this Spearman
    # rank correlation under a Gaussian copula. Zeroing before the repair
    # keeps the matrix block-diagonal (the repair preserves the blocks up to
    # float noise); zeroing again after it makes the zeros exact.
    return _force_categorical_identity(_project_to_correlation(latent), cat_dims)


def validate_correlation(
    R: np.ndarray,
    n_params: int,
    *,
    policy: RepairPolicy = "fitted",
) -> np.ndarray:
    """Validate a user-supplied correlation matrix and make it usable.

    Args:
        R: ``(D, D)`` candidate correlation matrix.
        n_params: Expected ``D``.
        policy: Repair policy, see :data:`RepairPolicy`. Surfaces that accept
            user-declared matrices pass ``"declared"``; internal fitting paths
            keep the default.

    Returns:
        A symmetric, positive-definite copy with unit diagonal.

    Raises:
        ValueError: If the shape is wrong, the matrix is not symmetric, the
            diagonal is not unit, any entry lies outside ``[-1, 1]``, or the
            repair of a declared matrix is material.
    """
    R = np.asarray(R, dtype=np.float64)
    _validate_structure(R, n_params)
    return _project_to_correlation(R, policy=policy)


def _validate_structure(R: np.ndarray, n_params: int) -> None:
    """Check the shape, symmetry, unit diagonal, and entry range of ``R``.

    Runs on the matrix exactly as the caller supplied it, on whichever scale
    that is, and before any conversion rewrites the entries.

    Args:
        R: Candidate correlation matrix.
        n_params: Expected ``D``.

    Raises:
        ValueError: If the shape is wrong, the matrix is not symmetric, the
            diagonal is not unit, or any entry lies outside ``[-1, 1]``.
    """
    if R.shape != (n_params, n_params):
        raise ValueError(f"correlation must be ({n_params}, {n_params}), got {R.shape}")
    if not np.allclose(R, R.T, atol=1e-10):
        raise ValueError("correlation must be symmetric")
    if not np.allclose(np.diag(R), 1.0, atol=1e-10):
        raise ValueError("correlation must have a unit diagonal")
    if np.abs(R).max() > 1.0 + 1e-10:
        raise ValueError("correlation entries must lie in [-1, 1]")


def _project_to_correlation(
    R: np.ndarray,
    *,
    policy: RepairPolicy = "fitted",
    report_kind: Literal["latent", "spearman"] = "latent",
) -> np.ndarray:
    """Return the nearest positive-definite matrix with a unit diagonal.

    A rank-correlation estimate transformed entry-by-entry through
    ``2 sin(pi rho_s / 6)`` is not guaranteed to stay positive definite, and a
    user-declared matrix need not be either. Clipping the eigenvalues and
    renormalising the diagonal is the standard repair; it is a no-op when the
    input is already valid, so the common case pays only one eigendecomposition.

    The repair is idempotent: ``_project_to_correlation`` of its own output
    returns that output bit for bit. One clip-then-renormalise pass does not
    guarantee this, because the renormalisation can push the smallest
    eigenvalue back under the floor, so the pass repeats until the floor
    holds. The result is symmetric by construction, so the leading
    symmetrisation is also a no-op on a second call.

    How loudly the repair reports itself depends on how far it had to move the
    matrix. The severity is the maximum entrywise change between the final
    repaired matrix and the matrix as it arrived, not a per-pass change, and it
    is measured on ``report_kind``. See :data:`RepairPolicy` for the bands.

    Args:
        R: ``(D, D)`` candidate correlation matrix (structurally valid), on
            the latent scale.
        policy: Repair policy, see :data:`RepairPolicy`.
        report_kind: Scale the caller declared the matrix on. With
            ``"spearman"`` both matrices go back through
            ``rho_s = (6 / pi) arcsin(rho / 2)`` before the change is measured,
            so the number the user is asked to judge is in the units the user
            wrote.

    Returns:
        A symmetric, positive-definite matrix with unit diagonal.

    Raises:
        ValueError: If ``policy`` is ``"declared"`` and the repair had to move
            an entry by ``_REPAIR_MATERIAL`` or more.
    """
    original = 0.5 * (R + R.T)  # kill any asymmetry from floating-point accumulation
    current = original
    declared_minimum = None

    for _ in range(_MAX_REPAIR_PASSES):
        eigenvalues, eigenvectors = np.linalg.eigh(current)
        smallest = float(eigenvalues.min())
        if declared_minimum is None:
            declared_minimum = smallest
        if smallest >= _MIN_EIGENVALUE:
            break
        lifted = np.maximum(eigenvalues, _EIGENVALUE_CLIP_TARGET)
        clipped = (eigenvectors * lifted) @ eigenvectors.T
        # Renormalise so the diagonal is exactly 1 again -- eigenvalue clipping
        # perturbs it, and every conditional formula below assumes R[i, i] == 1.
        scale = np.sqrt(np.diag(clipped))
        current = clipped / np.outer(scale, scale)
        current = 0.5 * (current + current.T)
        np.fill_diagonal(current, 1.0)
    else:
        # Never reached for any matrix we have seen; blending toward the
        # identity raises every eigenvalue and leaves the diagonal at 1, so it
        # closes the loop for certain rather than returning an unusable matrix.
        smallest = float(np.linalg.eigvalsh(current).min())
        weight = (2.0 * _EIGENVALUE_CLIP_TARGET - smallest) / (1.0 - smallest)
        current = (1.0 - weight) * current + weight * np.eye(current.shape[0])
        np.fill_diagonal(current, 1.0)

    if current is original:
        return original

    # Measure the whole repair, first pass to last, on the scale the caller
    # declared. A Spearman user never wrote the latent numbers, so a latent
    # change would ask them to judge a quantity they cannot recognise.
    if report_kind == "spearman":
        before, after = _latent_to_spearman(original), _latent_to_spearman(current)
    else:
        before, after = original, current
    change = float(np.abs(after - before).max())
    scale = "Spearman" if report_kind == "spearman" else "latent"
    detail = (
        f"is not positive definite (minimum eigenvalue {declared_minimum:.3e}); "
        f"repairing it by eigenvalue clipping moves an entry by up to {change:.3e} "
        f"on the {scale} scale"
    )

    if policy == "declared":
        if change >= _REPAIR_MATERIAL:
            raise ValueError(
                f"jaxgsa: the declared correlation matrix {detail}, which is too far "
                "to accept. A matrix that has "
                "to move this much is structurally inconsistent — the pairwise "
                "correlations cannot hold at the same time, or two parameters are "
                "redundant. Correct the matrix, or obtain a valid one from data with "
                "jaxgsa.sampling.fit_correlation. If you declared a rank correlation, "
                "pass correlation_kind='spearman' so it is converted instead of read "
                "as a latent matrix."
            )
        if change >= _REPAIR_NOISE:
            warnings.warn(
                f"jaxgsa: the declared correlation matrix {detail}. Samples will "
                "follow the repaired dependence structure, not the declared one — "
                "check the matrix for inconsistent pairwise correlations or redundant "
                "parameters.",
                stacklevel=2,
            )
    elif change >= _REPAIR_MATERIAL:
        # A fit is never refused: it is the data that is inconsistent, not the
        # user. A fit that moved this far is still worth reporting.
        warnings.warn(
            f"jaxgsa: the fitted correlation matrix {detail}. The fit is usable but "
            "the data it came from is close to rank deficient — check for duplicated "
            "or collinear columns.",
            stacklevel=2,
        )
    return current


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
        n: Number of points. Pass a power of two: the Sobol' engine loses its
            balance guarantees (and scipy warns) otherwise. jaxgsa callers
            round their sample sizes up before calling; the value is used
            as-is here.
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
