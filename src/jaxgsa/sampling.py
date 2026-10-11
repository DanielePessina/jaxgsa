"""Generic input sampling utilities."""

import warnings as _warnings

import numpy as _np
import numpy.typing as _npt
from scipy.stats import norm as _norm

from jaxgsa._core.copula import (
    _safe_cholesky,
    correlation_from_covariance,
)
from jaxgsa._core.copula import fit_gaussian_copula as _fit_gaussian_copula
from jaxgsa._core.copula import latent_to_physical as _latent_to_physical
from jaxgsa._core.dependence import InputSampler as InputSampler
from jaxgsa._core.dependence import transform_samples as _transform_sampler_points
from jaxgsa._core.sampling import _transform_samples
from jaxgsa._core.sampling import check_seed as _check_seed
from jaxgsa._core.warning_types import JaxgsaWarning as _JaxgsaWarning
from jaxgsa.problem import Problem as _Problem


def _require_problem(value: object, name: str, signature: str) -> None:
    """Reject a first argument that is not a Problem, with the right order.

    Args:
        value: What the caller passed as ``problem``.
        name: Public function name, such as ``"correlate"``.
        signature: The correct call order, such as ``"correlate(problem, X)"``.

    Raises:
        TypeError: If ``value`` is not a :class:`~jaxgsa.problem.Problem`.
    """
    if not isinstance(value, _Problem):
        raise TypeError(
            f"jaxgsa.sampling.{name}: the first argument must be a Problem; "
            f"the order is {signature}"
        )


def monte_carlo(
    problem: _Problem,
    n_samples: int,
    *,
    seed: int | _np.random.Generator | None = None,
) -> _np.ndarray:
    """Draw Monte Carlo samples from a problem's marginals.

    Each column follows the corresponding parameter's declared input
    distribution (uniform, Gaussian, truncated Gaussian, or categorical) via
    inverse-CDF transformation of pseudo-random draws; a categorical column
    comes back as level codes drawn with the declared probabilities. Unlike
    :func:`jaxgsa.sobol.sample`, this uses plain pseudo-random draws with no
    low-discrepancy structure and no Saltelli layout.

    A ``problem.correlation`` that declares a non-identity Gaussian-copula
    correlation matrix is honored transparently. Correlated standard normals
    are drawn on the latent scale, then pushed through each marginal's
    inverse CDF. This is the NORTA construction: every column keeps its
    declared marginal exactly, and the joint sample carries the declared
    dependence structure. Independent problems keep the plain uniform-draw
    path bit-for-bit, so existing seeds reproduce existing samples.

    A custom ``problem.sampler`` is honored too: pseudo-random uniform points
    of width ``problem.sampler_dimension`` go through its joint branch.

    To push your own unit-cube design (a Latin hypercube, say) through the
    same marginals, copula or sampler, use :func:`transform`.

    Args:
        problem: Problem definition with parameter names and distributions;
            its marginals determine how each column is transformed, and its
            optional ``correlation`` or ``sampler`` couples the columns.
        n_samples: Number of samples (rows) to draw. Must be at least 1.
        seed: Seed for NumPy's default RNG, or an existing
            ``np.random.Generator`` to draw from. ``None`` (default) uses
            fresh OS entropy, so repeated calls give different samples;
            pass an int for reproducible draws.

    Returns:
        Array of shape ``(n_samples, D)``, where ``D`` is the number of
        parameters, in physical units (each column transformed through the
        problem's declared marginal distribution).

    Raises:
        ValueError: If ``n_samples`` is less than 1.
        TypeError: If ``problem`` is not a ``Problem``, or if ``seed`` is a
            JAX key. Keys go to ``analyze(key=...)``.
    """
    _require_problem(problem, "monte_carlo", "monte_carlo(problem, n_samples)")
    _check_seed(seed, "jaxgsa.sampling.monte_carlo")
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    rng = _np.random.default_rng(seed)
    if problem.sampler is not None and problem.sampler_dimension is not None:
        return transform(problem, rng.random((n_samples, problem.sampler_dimension)))
    R = problem.correlation
    if problem.dependence == "independent" or R is None:
        return transform(problem, rng.random((n_samples, problem.n_params)))

    # The copula draws latent normals directly with standard_normal, which
    # keeps the random stream of earlier releases. The construction is the one
    # transform() uses, but the draws differ from transform(problem,
    # rng.random(...)) for the same seed.
    return _copula_to_physical(problem, rng.standard_normal((n_samples, problem.n_params)), R)


def transform(problem: _Problem, U: _npt.ArrayLike) -> _np.ndarray:
    """Map unit-cube points to physical rows through a problem's inputs.

    This is the step every jaxgsa sampler ends with, exposed so that you can
    bring your own design. Draw ``U`` from any low-discrepancy or space-filling
    generator, and ``transform`` applies the problem's marginals, its
    Gaussian copula, or its custom sampler, whichever the problem declares.

    A custom ``problem.sampler`` takes precedence and receives ``U`` as its
    uniform points. Otherwise a ``problem.correlation`` couples the columns:
    each column of ``U`` becomes a standard normal, the normals take the
    declared correlation, and each marginal's inverse CDF gives the physical
    value. :func:`monte_carlo` uses the same construction but draws its
    normals directly, so for a correlated problem it does not give the same
    rows as ``transform`` on uniforms from the same seed. Otherwise each
    column goes through its own marginal's inverse CDF.

    Example:
        A scrambled Latin hypercube design through a problem with a Gaussian
        input:

        .. code-block:: python

            from scipy.stats import qmc
            import jaxgsa

            problem = jaxgsa.Problem.from_dict(
                {"a": (0.0, 1.0), "b": {"dist": "gaussian", "mean": 0.0, "variance": 1.0}}
            )
            U = qmc.LatinHypercube(d=2, seed=0).random(256)
            X = jaxgsa.sampling.transform(problem, U)  # (256, 2), physical units

    Args:
        problem: Problem whose marginals, copula or sampler map the points.
        U: ``(n, D)`` points in the unit cube, one column per parameter, or
            ``(n, problem.sampler_dimension)`` when the problem has a custom
            sampler.

    Returns:
        ``(n, D)`` float64 array in physical units.

    Raises:
        TypeError: If ``problem`` is not a ``Problem``.
        ValueError: If ``U`` is not 2-D with the expected width, or holds
            values that are non-finite or outside ``[0, 1]``.
    """
    _require_problem(problem, "transform", "transform(problem, U)")
    u = _np.asarray(U, dtype=_np.float64)
    width = problem.n_params if problem.sampler_dimension is None else problem.sampler_dimension
    if u.ndim != 2 or u.shape[1] != width:
        raise ValueError(f"U must be (n, {width}) to match the problem, got {u.shape}")
    if not _np.all((u >= 0.0) & (u <= 1.0)):
        raise ValueError("U must hold finite values in [0, 1]")
    if problem.sampler is not None and problem.sampler_dimension is not None:
        return _transform_sampler_points(
            problem.sampler,
            u,
            problem.n_params,
            input_specs=problem.input_specs,
            names=problem.names,
        )
    R = problem.correlation
    if problem.dependence == "independent" or R is None:
        return _transform_samples(problem, u)
    # Exact 0 and 1 have no finite normal quantile.
    return _copula_to_physical(problem, _norm.ppf(_np.clip(u, 1e-12, 1.0 - 1e-12)), R)


def _copula_to_physical(problem: _Problem, Z: _np.ndarray, R: _np.ndarray) -> _np.ndarray:
    """Correlate independent standard normals and push them through the marginals.

    Args:
        problem: Problem supplying the marginals.
        Z: ``(n, D)`` independent standard-normal draws.
        R: ``(D, D)`` latent correlation matrix.

    Returns:
        ``(n, D)`` samples in physical units.
    """
    return _latent_to_physical(problem, Z @ _safe_cholesky(R).T)


def correlate(
    problem: _Problem,
    X: _npt.ArrayLike,
    *,
    seed: int | _np.random.Generator | None = None,
) -> _np.ndarray:
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

    With ``N <= D`` or ``N < 3`` the score matrix is singular or undefined
    (``corr(M)`` needs more rows than columns), so the de-correlation step is
    skipped and the achieved pairing is a plain random permutation that does
    not follow the declared matrix at this sample size. The fallback warns;
    draw more rows if the target dependence matters.

    Use it when the per-column samples already exist (an evaluated design, an
    observational data set) and only the dependence structure is missing.
    For fresh samples prefer :func:`monte_carlo`, which draws the coupling
    directly.

    References:
        Iman & Conover (1982). Commun. Stat. Simul. Comput. 11(3):311-334.

    Args:
        problem: Problem whose ``correlation`` matrix defines the target
            dependence structure. Must have one declared (may be identity,
            which reduces to a random re-pairing).
        X: ``(N, D)`` sample matrix in physical units, one column per
            parameter in ``problem.names`` order.
        seed: Seed for NumPy's default RNG, or an existing
            ``np.random.Generator``; controls the latent score draw.

    Returns:
        ``(N, D)`` array with the same values per column as ``X`` (each
        column permuted) and ranks following the declared correlation.

    Raises:
        TypeError: If ``problem`` is not a ``Problem``, for example when the
            arguments are in the order ``(X, problem)``.
        ValueError: If ``problem`` has no correlation set, ``X`` is not
            ``(N, D)`` with ``D == problem.n_params``, or ``X`` holds
            non-finite values (they would be paired deterministically with
            the extreme rank scores and bias the result).

    Warns:
        JaxgsaWarning: If ``N <= D`` or ``N < 3``, when the de-correlation
            step cannot run and the re-pairing does not follow the declared
            matrix.
    """
    _require_problem(problem, "correlate", "correlate(problem, X)")
    _check_seed(seed, "jaxgsa.sampling.correlate")
    R = problem.correlation
    if R is None:
        raise ValueError(
            "correlate() requires problem.correlation to be set; declare one at "
            "construction or attach it with problem.with_correlation(R)"
        )
    X = _np.asarray(X, dtype=_np.float64)
    D = problem.n_params
    if X.ndim != 2 or X.shape[1] != D:
        raise ValueError(f"X must be (N, {D}) to match the problem, got {X.shape}")
    if not _np.all(_np.isfinite(X)):
        # _np.sort puts NaN last, so every NaN would be deterministically
        # paired with the highest-score rows and bias the achieved
        # correlation of the finite rows — even if the NaNs are dropped later.
        n_bad = int(_np.count_nonzero(~_np.all(_np.isfinite(X), axis=1)))
        raise ValueError(
            f"X holds non-finite values (nan or inf) in {n_bad} row(s); rank "
            "re-pairing would pair them with the extreme scores and bias the "
            "correlation of the finite rows. Remove or impute the non-finite "
            "rows before calling correlate()."
        )

    rng = _np.random.default_rng(seed)
    scores = _iman_conover_scores(X.shape[0], R, rng)
    # Rank of each score within its column (argsort twice); row i of the
    # output takes the X value holding the same within-column rank.
    ranks = _np.argsort(_np.argsort(scores, axis=0), axis=0)
    return _np.take_along_axis(_np.sort(X, axis=0), ranks, axis=0)


def _iman_conover_scores(n: int, R: _np.ndarray, rng: _np.random.Generator) -> _np.ndarray:
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
    scores = _norm.ppf(_np.arange(1, n + 1) / (n + 1))
    M = _np.empty((n, D), dtype=_np.float64)
    for j in range(D):
        M[:, j] = rng.permutation(scores)

    # corr(M) is singular when there are fewer rows than columns, and undefined
    # for a single row. Those designs carry no usable rank structure anyway, so
    # skip the de-correlation and fall back to the raw score matrix. The
    # achieved pairing then does not follow the declared matrix, which the
    # caller must hear about.
    if n <= D or n < 3:
        _warnings.warn(
            f"jaxgsa.sampling.correlate: N={n} with D={D} leaves the score "
            "matrix singular (the de-correlation step needs N > D and N >= 3), "
            "so the re-pairing is a plain random permutation that does not "
            "follow the declared correlation matrix. Draw more rows if the "
            "target dependence matters.",
            stacklevel=3,
            category=_JaxgsaWarning,
        )
        return M

    T = _np.atleast_2d(_np.corrcoef(M, rowvar=False))
    # M @ (chol(R) chol(T)^{-1}).T == M @ chol(T)^{-T} chol(R).T.
    return M @ _np.linalg.solve(_safe_cholesky(T).T, _safe_cholesky(R).T)


def fit_correlation(problem: _Problem, X: _npt.ArrayLike) -> _np.ndarray:
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
        ValueError: If ``X`` is not ``(N, D)`` with ``D == problem.n_params``,
            or holds fewer than three rows.
    """
    return _fit_gaussian_copula(problem, _np.asarray(X, dtype=_np.float64))


__all__ = [
    "InputSampler",
    "correlate",
    "correlation_from_covariance",
    "fit_correlation",
    "monte_carlo",
    "transform",
]
