"""Variance-based sensitivity indices for correlated inputs (Li et al. 2010).

When inputs are dependent the Sobol' variance decomposition no longer holds.
The same conditional-variance quantities stay well defined, but they change
connotation (Hilhorst et al. 2024, Figure 1):

===============  ===================================  ==========================
index            definition                           reading
===============  ===================================  ==========================
``S_TC``         ``V(E(Y|X_i)) / V(Y)``               total correlated: what
                                                      ``X_i`` explains through
                                                      itself and through its
                                                      correlation. It is a
                                                      first-order quantity in
                                                      form. "Total" names the
                                                      pathways it counts, not
                                                      the interaction order
``S_TU``         ``E(V(Y|X_-i)) / V(Y)``              total uncorrelated: what
                                                      only ``X_i`` can explain
``S_U``          ``E(V(f_i|X_-i)) / V(Y)``            the independent part
``S_C``          ``S_TC - S_U``                       the correlation-borne part
``S_IU``         ``S_TU - S_U``                       independent interactions
===============  ===================================  ==========================

``f_i`` is the additive first-order component of ``Y``. It is fitted by least
squares under the correlated input measure (see
:func:`_fit_component_functions`). ``S_U`` therefore measures the part of
``f_i`` that ``X_-i`` cannot predict. This reading matches the decorrelated
first-order index of Mara & Tarantola (2012) only when ``f_i`` is linear;
their own estimator instead conditions on the Rosenblatt residual, a
different construction that this module does not implement. ``S_U`` is not
the structural index ``Var(f_i) / V(Y)`` of Li et al. (2010, Equation 25)
either. The two differ on a linear-Gaussian model: Li's keeps the correlated
share of ``Var(f_i)`` and this one removes it. The pair ``(S_U, S_C)`` splits
``S_TC`` along exactly the boundary this module's estimator draws.

``f_i`` is fitted in the basis each parameter's own marginal is orthonormal
under: probabilists' Hermite polynomials of the normal score ``Z_i`` for an
untruncated Gaussian marginal, shifted Legendre polynomials of the marginal
CDF value ``U_i`` otherwise (:func:`_component_basis_kinds`). A Legendre fit of
a polynomial or exponential response in ``Z_i`` converges slowly, because
``Z_i = Phi^{-1}(U_i)`` is steep near the tails; picking the basis the
marginal matches removes that gap.

Every expectation is taken under the problem's input distribution, reached
through a :class:`~jaxgsa._core.dependence.Draws` object: the closed-form
conditionals of the built-in Gaussian copula, or the joint and conditional
draws of a custom sampler declared on the problem. Both consume the same
uniform points, so one estimator serves both. Every model evaluation goes
through a fitted surrogate. That is the point of the two-stage approach: the
nested conditional sampling would be unaffordable against the original model.

References:
    Li, Rabitz, Yelvington et al. (2010). J. Phys. Chem. A 114:6022-6032.
    Mara & Tarantola (2012). Reliab. Eng. Syst. Saf. 107:115-121.
    Hilhorst, Quicken, van de Vosse & Huberts (2024). Int. J. Numer. Meth.
        Biomed. Engng. 40(2):e3797.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Sequence
from math import factorial
from typing import Callable, NamedTuple

import numpy as np
from scipy.special import ndtri

from jaxgsa._core.batching import resolve_batch_size
from jaxgsa._core.dependence import Draws, sobol_uniforms
from jaxgsa._core.legendre import legendre_orthonormal
from jaxgsa._core.warning_types import JaxgsaWarning
from jaxgsa.problem import GaussianSpec, Problem

# Degree of the marginal basis used to recover the first-order component
# functions f_i. The basis itself depends on each parameter's marginal (see
# _component_basis_kinds). Six terms capture the smooth univariate shapes a
# kernel surrogate produces without making the joint least squares ill-posed.
_COMPONENT_DEGREE = 6

# Largest dimension a scipy Sobol' engine has direction numbers for. The
# inner draw in _total_uncorrelated_and_conditional uses ``draws.self_dimension``
# Sobol' dimensions per outer point, so it splits into blocks above this width.
_MAX_QMC_DIM = 21201

# Keeps the probit of a marginal CDF coordinate finite at exact 0 and 1.
_UNIT_CLIP = 1e-12

# Ridge added to the component-function normal equations. The basis is
# marginally orthonormal, so the Gram matrix is well conditioned unless two
# parameters are near-perfectly correlated; this only covers that case.
_COMPONENT_RIDGE = 1e-8

# How far S_U may exceed S_TU before the clip is reported. Both are fractions
# of V(Y), so this is one percent of the output variance: below it the excess
# is estimator noise, above it the additive projection is genuinely inadequate.
_S_U_CLIP_TOLERANCE = 0.01


class _Streams(NamedTuple):
    """One independent host-side RNG stream per consumer of randomness.

    These estimators are scipy loops, so they cannot take a JAX key and split
    it. :class:`numpy.random.SeedSequence` is the host-side equivalent:
    ``spawn`` derives children whose states are decorrelated by construction,
    which an integer offset such as ``seed + 1`` does not do. Every draw in
    this module reaches its scrambling seed through one of these fields, so
    there is nowhere left for an offset to be reintroduced.

    Attributes:
        variance: Stream for the unconditional joint sample.
        total_correlated: One stream per parameter for
            :func:`_total_correlated`.
        total_uncorrelated: One stream per parameter for
            :func:`_total_uncorrelated_and_conditional`.
    """

    variance: np.random.SeedSequence
    total_correlated: list[np.random.SeedSequence]
    total_uncorrelated: list[np.random.SeedSequence]


def _spawn_streams(entropy: int, n_parameters: int) -> _Streams:
    """Spawn the independent streams the estimators need from one root.

    ``SeedSequence.spawn`` decorrelates its children by construction,
    regardless of how deep the spawning tree is, so one flat
    ``1 + 2 * n_parameters``-way spawn is exactly as independent as spawning
    three then spawning ``n_parameters`` from two of those.

    Args:
        entropy: Root entropy, folded down from the caller's PRNG key.
        n_parameters: Number of parameters ``D``.

    Returns:
        A :class:`_Streams` holding ``2 * D + 1`` independent child sequences.
    """
    children = np.random.SeedSequence(entropy).spawn(1 + 2 * n_parameters)
    return _Streams(
        variance=children[0],
        total_correlated=children[1 : 1 + n_parameters],
        total_uncorrelated=children[1 + n_parameters :],
    )


def _qmc_seed(stream: np.random.SeedSequence) -> int:
    """Draw the scrambling seed ``scipy.stats.qmc`` takes from a stream.

    ``qmc.Sobol`` wants a plain integer (or a ``Generator``), so a spawned
    child is turned into one here, and only here. The value comes out of the
    sequence's own hashed state, so two children never produce neighbouring
    integers the way ``seed + 1`` does.

    Args:
        stream: One spawned child sequence.

    Returns:
        A non-negative integer below ``2**32``.
    """
    return int(stream.generate_state(1, dtype=np.uint32)[0])


class CorrelatedIndices(NamedTuple):
    """Raw index arrays produced by :func:`estimate_correlated_indices`.

    Index arrays have shape ``(S, D)`` for ``S`` output slices and ``D``
    parameters, matching the surrogate's flattened output layout.

    Attributes:
        S_TC: Total correlated indices ``V(E(Y|X_i)) / V(Y)``, shape ``(S, D)``.
        S_TU: Total uncorrelated indices ``E(V(Y|X_-i)) / V(Y)``, shape
            ``(S, D)``.
        S_U: Uncorrelated (independent) contribution, shape ``(S, D)``. Clipped
            to at most ``S_TU``.
        S_C: Correlated contribution ``S_TC - S_U``, shape ``(S, D)``. It may be
            negative.
        S_IU: Independent interaction contribution ``S_TU - S_U``, shape
            ``(S, D)``. Non-negative by construction of the clip.
        variance: Output variance under the correlated input measure, shape
            ``(S,)``.
    """

    S_TC: np.ndarray
    S_TU: np.ndarray
    S_U: np.ndarray
    S_C: np.ndarray
    S_IU: np.ndarray
    variance: np.ndarray


def estimate_correlated_indices(
    *,
    problem: Problem,
    draws: Draws,
    predict: Callable[[np.ndarray], np.ndarray],
    n_outer: int,
    n_inner: int,
    n_variance: int,
    entropy: int,
    batch_size: int | None,
    param_names: Sequence[str] | None = None,
) -> CorrelatedIndices:
    """Estimate the five correlated variance-based indices by quasi-Monte-Carlo.

    The surrogate is fitted in marginal CDF coordinates ``u_i = F_i(x_i)``,
    so every draw reaches it through :meth:`Draws.to_unit`. While
    conditioning, the draws stay in the implementation's own space: latent
    normal for the Gaussian copula, physical for a custom sampler.

    Every stage takes its noise as scrambled Sobol' points of width
    ``draws.dimension``. Conditioning rows are tiled ``n_inner`` times, so
    one chunk of outer points costs one conditional draw and one surrogate
    call, whichever implementation ``draws`` is.

    Args:
        problem: Problem whose declared marginals pick each parameter's
            component-fit basis (see :func:`_component_basis_kinds`).
        draws: Joint and conditional draws of the inputs, from
            :func:`jaxgsa._core.dependence.draws_for` or built directly. The
            conditionals and the joint draw come from the same object, so
            they always describe the same distribution.
        predict: Fitted surrogate. It maps ``(n, D)`` unit-cube samples to
            ``(n, S)`` outputs.
        n_outer: Outer (conditioning) sample size per parameter.
        n_inner: Inner (conditional) sample size per outer point.
        n_variance: Sample size for the unconditional output variance and for
            the component-function fit.
        entropy: Root entropy for the host-side draws, folded down from the
            caller's PRNG key. Each parameter and stage gets its own
            :class:`numpy.random.SeedSequence` child spawned from it, never a
            seed derived by arithmetic.
        batch_size: Sample rows per device call, or ``None`` to derive one
            from the memory budget. Required, with no default, because the
            value used to be droppable: it bounds the nested conditional draws as
            well as the surrogate evaluation: a chunk of outer points expands
            to ``chunk * n_inner`` rows, and that expansion is what the
            caller's value has to hold down.
        param_names: Parameter names for the ``S_U`` clip warning, or
            ``None`` to report column positions.

    Returns:
        A :class:`CorrelatedIndices` whose index arrays have shape ``(S, D)``.
    """
    D = problem.n_params
    streams = _spawn_streams(entropy, D)
    basis_kinds = _component_basis_kinds(problem)

    # --- Unconditional variance, and the additive component functions --------
    rows_var = draws.joint(
        sobol_uniforms(n_variance, draws.dimension, seed=_qmc_seed(streams.variance))
    )
    U_var = draws.to_unit(rows_var)
    Z_var = _probit(U_var)
    Y_var = np.asarray(predict(U_var), dtype=np.float64)
    variance = Y_var.var(axis=0)
    # A zero-variance slice makes every index undefined rather than zero; NaN
    # propagates that honestly, matching the convention in hdmr._fit._get_stats_step.
    safe_variance = np.where(variance > 0.0, variance, np.nan)

    coefficients = _fit_component_functions(U_var, Z_var, Y_var, basis_kinds)

    n_slices = Y_var.shape[1]
    S_TC = np.empty((n_slices, D), dtype=np.float64)
    S_TU = np.empty((n_slices, D), dtype=np.float64)
    S_U = np.empty((n_slices, D), dtype=np.float64)

    # Every nested draw goes through one queue, so the surrogate sees a few
    # large calls instead of 2D small ones. Draws do not depend on the
    # predictions, so evaluation can wait until the queue is flushed.
    flush_rows = (
        batch_size
        if batch_size is not None
        else resolve_batch_size(8 * (2 * draws.dimension + n_slices), 1 << 62, None)
    )
    queue = _PredictQueue(predict, flush_rows)
    finish_total: list[Callable[[], np.ndarray]] = []
    finish_uncorrelated: list[Callable[[], tuple[np.ndarray, np.ndarray]]] = []
    for i in range(D):
        finish_total.append(
            _total_correlated(
                draws=draws,
                index=i,
                queue=queue,
                n_outer=n_outer,
                n_inner=n_inner,
                n_slices=n_slices,
                stream=streams.total_correlated[i],
                batch_size=batch_size,
            )
        )
        finish_uncorrelated.append(
            _total_uncorrelated_and_conditional(
                draws=draws,
                index=i,
                queue=queue,
                component=coefficients[:, i, :],
                basis_kind=basis_kinds[i],
                n_outer=n_outer,
                n_inner=n_inner,
                n_slices=n_slices,
                stream=streams.total_uncorrelated[i],
                batch_size=batch_size,
            )
        )
    queue.flush()

    for i in range(D):
        S_TC[:, i] = finish_total[i]()
        total_uncorrelated, conditional_component_var = finish_uncorrelated[i]()
        S_TU[:, i] = total_uncorrelated
        # The independent contribution is what f_i explains beyond the part of
        # itself that X_-i already determines through the correlation. V(f_i)
        # is a marginal quantity, so it comes from the joint variance sample
        # rather than from the nested conditional draws.
        value = _component_value(U_var[:, i], Z_var[:, i], basis_kinds[i])
        f_i = _evaluate_component(value, coefficients[:, i, :], basis_kinds[i])
        S_U[:, i] = np.maximum(f_i.var(axis=0) - conditional_component_var, 0.0)

    S_TC /= safe_variance[:, None]
    S_TU /= safe_variance[:, None]
    S_U /= safe_variance[:, None]
    S_U = _clip_independent_part(S_U, S_TU, param_names)
    return CorrelatedIndices(
        S_TC=S_TC,
        S_TU=S_TU,
        S_U=S_U,
        # Identities (24) and (28) of Hilhorst et al.: the correlated part is
        # whatever the total correlated index holds beyond the independent
        # contribution, and likewise for independent interactions.
        S_C=S_TC - S_U,
        S_IU=S_TU - S_U,
        variance=variance,
    )


def _probit(u: np.ndarray) -> np.ndarray:
    """Map marginal CDF coordinates to normal scores ``Phi^-1(u)``.

    For the Gaussian copula this recovers the latent normal draw. For a
    custom sampler it is the same transform of its CDF coordinates, so the
    Hermite component basis reads the same coordinate under either.

    Args:
        u: Values in ``[0, 1]``, any shape.

    Returns:
        Finite normal scores, same shape as ``u``.
    """
    return ndtri(np.clip(u, _UNIT_CLIP, 1.0 - _UNIT_CLIP))


def _clip_independent_part(
    S_U: np.ndarray, S_TU: np.ndarray, param_names: Sequence[str] | None = None
) -> np.ndarray:
    """Hold ``S_U <= S_TU`` so ``S_IU`` cannot go negative, and report the clip.

    ``S_TU`` is everything ``X_i`` alone explains, and ``S_U`` is the part of
    that which the additive component ``f_i`` accounts for. The remainder,
    ``S_IU``, is the independent interaction share, so ``S_U <= S_TU`` is a
    definitional invariant of the split.

    The two indices come from different estimators, and only ``S_TU`` sees the
    surrogate directly. On a model with interactions the degree-6 additive
    projection ``f_i`` absorbs interaction variance that it cannot represent as
    a function of ``X_i`` alone, so under a correlated measure the raw ``S_U``
    can exceed ``S_TU``. Clipping restores the invariant. A clip wider than
    ``_S_U_CLIP_TOLERANCE`` is not noise: it says the additive projection is
    inadequate for this model, and it warns.

    ``S_C = S_TC - S_U`` stays free to go negative. That is a real reading, a
    correlation that opposes the direct effect. It is not an estimator artefact.

    Args:
        S_U: Independent contribution normalised by ``V(Y)``, shape ``(S, D)``.
        S_TU: Total uncorrelated index normalised by ``V(Y)``, shape ``(S, D)``.
        param_names: Names to report clipped parameters by. ``None`` falls
            back to column positions.

    Returns:
        ``S_U`` clipped elementwise to at most ``S_TU``.

    Warns:
        JaxgsaWarning: If any entry is clipped by more than
            ``_S_U_CLIP_TOLERANCE`` of the output variance.
    """
    excess = S_U - S_TU
    # NaN slices (zero output variance) compare False and stay out of the report.
    flagged = excess > _S_U_CLIP_TOLERANCE
    if flagged.any():
        cols = sorted({int(j) for j in np.argwhere(flagged)[:, 1]})
        # Name the parameters like the zero-variance warning does; fall back
        # to column positions when the caller has no names to give.
        params = [param_names[j] for j in cols] if param_names is not None else cols
        warnings.warn(
            f"jaxgsa.vkoga: S_U exceeded S_TU by up to {float(excess[flagged].max()):.3g} "
            f"of the output variance for parameter(s) {params}; S_U was clipped to S_TU so "
            "that S_IU stays non-negative. The additive component functions cannot represent "
            "this model's interactions under the correlated measure, so read S_U, S_C and "
            "S_IU for those parameters as indicative only. S_TC and S_TU are unaffected.",
            # 1 here, 2 estimate_correlated_indices, 3 vkoga.analyze, 4 the caller.
            stacklevel=4,
            category=JaxgsaWarning,
        )
    return np.minimum(S_U, S_TU)


def _legendre_basis(u: np.ndarray, degree: int) -> np.ndarray:
    """Evaluate shifted Legendre polynomials 1..degree on ``u`` in (0, 1).

    Each ``u_i = F_i(X_i)`` is exactly uniform on (0, 1) by construction of the
    copula, whatever the dependency structure, so shifted Legendre polynomials
    are marginally orthonormal and mean-zero here. That makes the additive
    least squares below well posed without any extra centering.

    Thin wrapper over the shared recurrence in :mod:`jaxgsa._core.legendre`:
    shift ``(0, 1)`` onto ``(-1, 1)`` and drop the constant term.

    Args:
        u: Values in (0, 1), shape ``(...,)``.
        degree: Highest polynomial degree. Term 0 (the constant) is dropped.

    Returns:
        Basis values, shape ``(..., degree)``.
    """
    return legendre_orthonormal(2.0 * u - 1.0, degree)[..., 1:]


def _hermite_basis(z: np.ndarray, degree: int) -> np.ndarray:
    """Evaluate orthonormal probabilists' Hermite polynomials 1..degree on ``z``.

    Each ``z_i = Phi^-1(u_i)`` is exactly standard normal, so probabilists'
    Hermite polynomials are marginally orthonormal and mean-zero here, the
    same reasoning :func:`_legendre_basis` uses for a uniform marginal.

    Same three-term recurrence as
    :func:`jaxgsa.pce._engine._hermite_1d`, transcribed for a host NumPy
    array: ``He_0(z) = 1``, ``He_1(z) = z``,
    ``He_{n+1}(z) = z He_n(z) - n He_{n-1}(z)``, normalised by
    ``1 / sqrt(n!)``.

    Args:
        z: Standard normal values, shape ``(...,)``.
        degree: Highest polynomial degree. Term 0 (the constant) is dropped.

    Returns:
        Basis values, shape ``(..., degree)``.
    """
    columns = [np.ones_like(z), z] if degree >= 1 else [np.ones_like(z)]
    for n in range(1, degree):
        columns.append(z * columns[n] - n * columns[n - 1])
    basis = np.stack(columns, axis=-1)
    norms = np.array([1.0 / np.sqrt(factorial(k)) for k in range(degree + 1)])
    return (basis * norms)[..., 1:]


def _component_basis_kinds(problem: Problem) -> tuple[str, ...]:
    """Pick each parameter's component-fit basis from its declared marginal.

    ``"hermite"`` for an untruncated Gaussian marginal, where the physical
    variable is an affine map of the normal score ``Z_i`` and Hermite
    polynomials of ``Z_i`` represent a polynomial response exactly.
    ``"legendre"`` otherwise (uniform, or a truncated Gaussian, whose CDF
    is not the standard normal CDF, so ``Z_i`` no longer maps affinely to
    the physical variable): the marginal CDF value ``U_i`` is always exactly
    uniform, whatever the marginal, so Legendre stays a safe fallback there.

    Args:
        problem: Problem whose ``input_specs`` decide the per-parameter
            basis.

    Returns:
        Length-``D`` tuple of ``"hermite"`` or ``"legendre"``.
    """
    kinds = []
    for spec in problem.input_specs:
        is_untruncated_gaussian = (
            isinstance(spec, GaussianSpec) and spec.low is None and spec.high is None
        )
        kinds.append("hermite" if is_untruncated_gaussian else "legendre")
    return tuple(kinds)


def _component_value(u: np.ndarray, z: np.ndarray, kind: str) -> np.ndarray:
    """Pick the coordinate a component-fit basis of the given kind reads.

    Args:
        u: Marginal CDF values for one parameter, shape ``(...,)``.
        z: Normal scores ``Phi^-1(u)`` for the same parameter, shape ``(...,)``.
        kind: ``"hermite"`` or ``"legendre"``, from
            :func:`_component_basis_kinds`.

    Returns:
        ``z`` for ``"hermite"``, ``u`` for ``"legendre"``.
    """
    return z if kind == "hermite" else u


def _fit_component_functions(
    U: np.ndarray, Z: np.ndarray, Y: np.ndarray, basis_kinds: Sequence[str]
) -> np.ndarray:
    """Fit the additive first-order component functions ``f_i``.

    Least squares of the output onto ``sum_i f_i(X_i)`` under the correlated
    input measure is exactly the decomposition Li et al. use. The fitted
    ``f_i`` absorbs both what ``X_i`` explains alone and what it explains
    through its correlation with the rest. The independent part therefore has
    to be extracted afterwards by conditioning.

    Args:
        U: Marginal CDF values of the sample, shape ``(N, D)``.
        Z: Normal scores ``Phi^-1(u)`` of the sample, shape ``(N, D)``.
        Y: Surrogate outputs, shape ``(N, S)``.
        basis_kinds: Per-parameter basis choice from
            :func:`_component_basis_kinds`, length ``D``.

    Returns:
        Basis coefficients, shape ``(degree, D, S)``.
    """
    N, D = U.shape
    columns = []
    for d, kind in enumerate(basis_kinds):
        value = _component_value(U[:, d], Z[:, d], kind)
        basis_fn = _hermite_basis if kind == "hermite" else _legendre_basis
        columns.append(basis_fn(value, _COMPONENT_DEGREE))
    basis = np.stack(columns, axis=1)  # (N, D, degree)
    design = basis.reshape(N, D * _COMPONENT_DEGREE)

    gram = design.T @ design
    gram.flat[:: gram.shape[0] + 1] += _COMPONENT_RIDGE * N
    coefficients = np.linalg.solve(gram, design.T @ (Y - Y.mean(axis=0)))
    return coefficients.reshape(D, _COMPONENT_DEGREE, -1).transpose(1, 0, 2)


def _evaluate_component(value: np.ndarray, coefficients: np.ndarray, kind: str) -> np.ndarray:
    """Evaluate one component function ``f_i``.

    Args:
        value: ``u`` (``kind="legendre"``) or ``z`` (``kind="hermite"``)
            values for parameter ``i``, shape ``(...,)``.
        coefficients: Coefficients for that parameter, shape ``(degree, S)``.
        kind: ``"hermite"`` or ``"legendre"``, from
            :func:`_component_basis_kinds`.

    Returns:
        Component values, shape ``(..., S)``.
    """
    basis_fn = _hermite_basis if kind == "hermite" else _legendre_basis
    return basis_fn(value, _COMPONENT_DEGREE) @ coefficients


def _outer_chunk(n_outer: int, n_inner: int, n_columns: int, batch_size: int | None) -> int:
    """Return how many outer points fit in the transient-memory budget.

    Each outer point expands into ``n_inner`` rows. Each row holds the
    uniform noise, the tiled conditioning row, the drawn row, its unit-cube
    image and the ``S``-column prediction, in float64. Thin wrapper over the
    shared :func:`resolve_batch_size` with one outer point as the "row".

    ``batch_size`` counts sample rows, not outer points, so a caller's value
    is divided by the expansion factor before it is applied here. Passing it
    through unchanged would let the loop hold ``n_inner`` times the rows the
    caller asked for.

    Args:
        n_outer: Total outer points.
        n_inner: Inner draws per outer point.
        n_columns: Columns held per expanded row (``dimension + 3 * D + S``).
        batch_size: Caller's row budget, or ``None`` to derive one from the
            memory budget. A budget smaller than one outer point's expansion
            still processes one outer point: the inner block is the smallest
            unit the estimator can average over.

    Returns:
        Chunk size in ``[1, n_outer]``.
    """
    requested = None if batch_size is None else max(1, batch_size // n_inner)
    return resolve_batch_size(8 * n_inner * n_columns, n_outer, requested)


class _PredictQueue:
    """Collect unit-cube blocks and evaluate them in few surrogate calls.

    Prediction is row-independent, so blocks from different estimators can
    share one device call. The queue holds at most ``flush_rows`` rows, so
    host memory stays bounded. Each block carries a callback that receives
    the predictions for exactly its own rows.

    Attributes:
        predict: Fitted surrogate on unit-cube rows.
        flush_rows: Pending row count that triggers a flush.
    """

    def __init__(self, predict: Callable[[np.ndarray], np.ndarray], flush_rows: int) -> None:
        """Create an empty queue.

        Args:
            predict: Fitted surrogate. It maps ``(n, D)`` unit-cube samples
                to ``(n, S)`` outputs.
            flush_rows: Pending row count that triggers a flush, at least 1.
        """
        self.predict = predict
        self.flush_rows = max(1, flush_rows)
        self._blocks: list[np.ndarray] = []
        self._callbacks: list[Callable[[np.ndarray], None]] = []
        self._pending = 0

    def submit(self, U: np.ndarray, callback: Callable[[np.ndarray], None]) -> None:
        """Queue a block, flushing first if it would not fit.

        Args:
            U: Unit-cube rows, shape ``(n, D)``.
            callback: Called with the float64 predictions for ``U``, shape
                ``(n, S)``, when the block is evaluated.
        """
        if self._pending + U.shape[0] > self.flush_rows:
            self.flush()
        self._blocks.append(U)
        self._callbacks.append(callback)
        self._pending += U.shape[0]

    def flush(self) -> None:
        """Evaluate every pending block in one surrogate call."""
        if not self._blocks:
            return
        blocks, callbacks = self._blocks, self._callbacks
        self._blocks, self._callbacks, self._pending = [], [], 0
        joined = blocks[0] if len(blocks) == 1 else np.concatenate(blocks, axis=0)
        Y = np.asarray(self.predict(joined), dtype=np.float64)
        start = 0
        for block, callback in zip(blocks, callbacks, strict=True):
            stop = start + block.shape[0]
            callback(Y[start:stop])
            start = stop


def _total_correlated(
    *,
    draws: Draws,
    index: int,
    queue: _PredictQueue,
    n_outer: int,
    n_inner: int,
    n_slices: int,
    stream: np.random.SeedSequence,
    batch_size: int | None,
) -> Callable[[], np.ndarray]:
    """Estimate ``V(E(Y|X_i))`` for one parameter.

    The outer points are joint draws, of which only column ``index`` is kept:
    that column follows the marginal of ``X_i``. Each one is tiled
    ``n_inner`` times and the rest is redrawn from ``X_-i | X_i``.

    Outer points are processed in memory-budgeted chunks, one conditional
    draw and one surrogate call per chunk. Only the ``(n_outer, S)`` inner
    means stay resident; the full ``(n_outer * n_inner, S)`` block is never
    materialised. All uniform noise is drawn up front, so chunking changes
    evaluation batching only, not the sample.

    Note:
        The estimator drops the usual ``inner_var / n_inner`` correction and
        relies on the shared QMC inner block to cancel most of the inner error.
        That argument is empirical, and it holds only while ``n_inner`` is
        large against the outer conditioning signal. Do not lower ``n_inner``
        below the default to buy speed: the residual inner variance then leaks
        into the outer variance and inflates a small ``S_TC``. Raise
        ``n_outer`` instead.

    Args:
        draws: Joint and conditional draws of the inputs.
        index: Position of the parameter ``X_i`` being conditioned on.
        queue: Prediction queue for the fitted surrogate.
        n_outer: Outer (conditioning) sample size.
        n_inner: Inner (conditional) sample size per outer point.
        n_slices: Number of output slices ``S``.
        stream: This parameter's spawned :class:`numpy.random.SeedSequence`.
            The outer and inner draws take two further children of it.
        batch_size: Sample rows per device call, or ``None`` to derive one
            from the memory budget.

    Returns:
        A function to call after ``queue`` has been flushed. It returns the
        conditional-expectation variance, shape ``(S,)``. Not yet normalised.
    """
    outer_stream, inner_stream = stream.spawn(2)
    outer = draws.joint_led_by(
        sobol_uniforms(n_outer, draws.dimension, seed=_qmc_seed(outer_stream)), index
    )
    # One inner block, reused across every outer point. Sharing it makes the
    # inner integration rule identical for all conditioning values, which
    # cancels much of its error out of the outer variance.
    inner = sobol_uniforms(n_inner, draws.rest_dimension, seed=_qmc_seed(inner_stream))

    inner_mean = np.empty((n_outer, n_slices), dtype=np.float64)
    n_columns = draws.rest_dimension + 3 * outer.shape[1] + n_slices
    chunk = _outer_chunk(n_outer, n_inner, n_columns, batch_size)
    for start in range(0, n_outer, chunk):
        conditioning = np.repeat(outer[start : start + chunk], n_inner, axis=0)
        noise = np.tile(inner, (conditioning.shape[0] // n_inner, 1))
        rows = draws.rest_given_self(noise, conditioning, index)

        def _store(Y: np.ndarray, start: int = start) -> None:
            means = Y.reshape(-1, n_inner, n_slices).mean(axis=1)
            inner_mean[start : start + means.shape[0]] = means

        queue.submit(draws.to_unit(rows), _store)

    # No inner-noise correction. The textbook term ``inner_var / n_inner``
    # assumes iid inner draws. Here the inner block is QMC and shared across
    # outer points, so the inner error is far smaller and mostly constant in
    # the outer variance. Chosen empirically against the linear-Gaussian
    # closed form: the correction clamps small true S_TC to near zero, while
    # the uncorrected estimator is unbiased at equal spread.
    return lambda: inner_mean.var(axis=0)


def _inner_noise(
    n_inner: int, n_outer: int, dimension: int, stream: np.random.SeedSequence
) -> Iterator[tuple[int, np.ndarray]]:
    """Draw one independent stratified inner block per outer point.

    Each outer point gets its own ``dimension`` columns of a scrambled Sobol'
    set, so the inner integration error is independent across outer points
    and averages away in ``S_TU``.

    A Sobol' engine carries direction numbers for at most ``_MAX_QMC_DIM``
    dimensions, so one call serves at most ``_MAX_QMC_DIM // dimension``
    outer points. A larger ``n_outer`` is split into consecutive blocks of
    outer points, each drawn from its own spawned child stream. The blocks
    stay independent across the split, which is all this estimator asks of
    them. Blocks are yielded one at a time, so at most one block of noise is
    resident.

    Args:
        n_inner: Inner draws per outer point.
        n_outer: Number of outer points.
        dimension: Uniform coordinates per draw, ``draws.self_dimension``.
        stream: This estimator's inner :class:`numpy.random.SeedSequence`.

    Yields:
        ``(start, noise)`` pairs: the first outer point the block covers and
        its uniform noise, shape ``(width, n_inner, dimension)``.
    """
    per_block = max(1, _MAX_QMC_DIM // dimension)
    if n_outer <= per_block:
        starts, seeds = [0], [stream]
    else:
        starts = list(range(0, n_outer, per_block))
        seeds = stream.spawn(len(starts))
    for start, seed in zip(starts, seeds, strict=True):
        width = min(per_block, n_outer - start)
        block = sobol_uniforms(n_inner, width * dimension, seed=_qmc_seed(seed))
        yield start, block.reshape(n_inner, width, dimension).transpose(1, 0, 2)


def _total_uncorrelated_and_conditional(
    *,
    draws: Draws,
    index: int,
    queue: _PredictQueue,
    component: np.ndarray,
    basis_kind: str,
    n_outer: int,
    n_inner: int,
    n_slices: int,
    stream: np.random.SeedSequence,
    batch_size: int | None,
) -> Callable[[], tuple[np.ndarray, np.ndarray]]:
    """Estimate ``E(V(Y|X_-i))`` and ``V(E(f_i|X_-i))`` for one parameter.

    Both quantities condition on ``X_-i`` and resample ``X_i``, so they share
    one nested sample: the surrogate gives the total uncorrelated index, and
    the fitted component function evaluated on the same draws gives the
    conditional expectation that ``S_U`` subtracts from ``V(f_i)``.

    The outer points are joint draws, of which every column except ``index``
    is kept. Outer points are processed in memory-budgeted chunks, as in
    :func:`_total_correlated`. Only ``(n_outer, S)`` accumulators stay
    resident, and chunking does not change the sample.

    Args:
        draws: Joint and conditional draws of the inputs.
        index: Position of the parameter ``X_i`` being resampled.
        queue: Prediction queue for the fitted surrogate.
        component: Basis coefficients of ``f_i``, shape ``(degree, S)``.
        basis_kind: ``"hermite"`` or ``"legendre"``, the basis ``component``
            was fitted in (see :func:`_component_basis_kinds`).
        n_outer: Outer (conditioning) sample size.
        n_inner: Inner (conditional) sample size per outer point.
        n_slices: Number of output slices ``S``.
        stream: This parameter's spawned :class:`numpy.random.SeedSequence`.
            The outer and inner draws take two further children of it.
        batch_size: Sample rows per device call, or ``None`` to derive one
            from the memory budget.

    Returns:
        A function to call after ``queue`` has been flushed. It returns
        ``(conditional_variance_mean, component_conditional_variance)``, each
        shape ``(S,)``. Neither is normalised by ``V(Y)``.
    """
    outer_stream, inner_stream = stream.spawn(2)
    outer = draws.joint(sobol_uniforms(n_outer, draws.dimension, seed=_qmc_seed(outer_stream)))

    inner_var = np.empty((n_outer, n_slices), dtype=np.float64)
    f_hat = np.empty((n_outer, n_slices), dtype=np.float64)
    n_columns = draws.self_dimension + 3 * outer.shape[1] + n_slices
    chunk = _outer_chunk(n_outer, n_inner, n_columns, batch_size)
    # A fresh inner block per outer point (see _inner_noise). A single shared
    # block (the old behaviour) freezes the inner sampling error into a
    # per-run bias that a mean over outer points cannot average away, because
    # S_TU is a mean of inner variances, not a variance of inner means
    # (contrast :func:`_total_correlated`, where the shared block is correct).
    width = draws.self_dimension
    for block_start, block in _inner_noise(n_inner, n_outer, width, inner_stream):
        for offset in range(0, block.shape[0], chunk):
            start = block_start + offset
            noise = block[offset : offset + chunk]
            stop = start + noise.shape[0]
            conditioning = np.repeat(outer[start:stop], n_inner, axis=0)
            rows = draws.self_given_rest(noise.reshape(-1, width), conditioning, index)
            U = draws.to_unit(rows)

            def _store(Y: np.ndarray, start: int = start, stop: int = stop) -> None:
                inner_var[start:stop] = Y.reshape(-1, n_inner, n_slices).var(axis=1, ddof=1)

            queue.submit(U, _store)

            # E(f_i | X_-i), averaged over the same inner draws of X_i. Its
            # variance over the outer sample is the correlation-explained part
            # of f_i that Equation (25) removes.
            u_self = U[:, index].reshape(-1, n_inner)
            value = _component_value(u_self, _probit(u_self), basis_kind)
            f_hat[start:stop] = _evaluate_component(value, component, basis_kind).mean(axis=1)

    return lambda: (inner_var.mean(axis=0), f_hat.var(axis=0))
