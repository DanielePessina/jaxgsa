"""Shapley-effect analysis entry point.

Computes global Shapley effects (Owen 2014; Song, Nelson & Staum 2016)
analytically from a fitted surrogate's variance decomposition -- either
RS-HDMR component-function variances or PCE coefficients -- with no
permutation Monte Carlo. Assumes independent inputs.
"""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
from jax import Array

from gsax.hdmr._analyze import analyze_hdmr
from gsax.pce._analyze import analyze_pce
from gsax.problem import Problem
from gsax.shapley._engine import build_membership, shapley_from_variances
from gsax.shapley._result import ShapleyResult

# Backend defaults, applied when the corresponding kwarg is left as None.
# They mirror the signatures of analyze_hdmr / analyze_pce exactly.
_HDMR_DEFAULTS: dict = {
    "prenormalize": False,
    "maxorder": 2,
    "maxiter": 100,
    "m": 2,
    "lambdax": 0.01,
    "chunk_size": 2048,
}
_PCE_DEFAULTS: dict = {"order": 3, "ridge": 1e-8, "fit_ratio": 0.5}


def _resolve_backend_kwargs(
    backend: str,
    hdmr_kwargs: dict,
    pce_kwargs: dict,
) -> dict:
    """Validate per-backend kwargs and fill in the backend's defaults.

    Args:
        backend: Selected backend, ``"hdmr"`` or ``"pce"``.
        hdmr_kwargs: HDMR-only kwargs as passed by the user (None = unset).
        pce_kwargs: PCE-only kwargs as passed by the user (None = unset).

    Returns:
        The selected backend's kwargs with None entries replaced by defaults.

    Raises:
        ValueError: If ``backend`` is unknown, or a kwarg belonging to the
            non-selected backend was explicitly set.
    """
    if backend == "hdmr":
        selected, foreign, defaults = hdmr_kwargs, pce_kwargs, _HDMR_DEFAULTS
        foreign_backend = "pce"
    elif backend == "pce":
        selected, foreign, defaults = pce_kwargs, hdmr_kwargs, _PCE_DEFAULTS
        foreign_backend = "hdmr"
    else:
        raise ValueError(f"backend must be 'hdmr' or 'pce', got {backend!r}")

    bad = sorted(k for k, v in foreign.items() if v is not None)
    if bad:
        raise ValueError(
            f"{bad} only apply to backend='{foreign_backend}', but backend='{backend}' "
            "was selected"
        )

    return {k: (defaults[k] if v is None else v) for k, v in selected.items()}


def analyze_shapley(
    problem: Problem,
    X: Array,
    Y: Array,
    *,
    backend: Literal["hdmr", "pce"] = "hdmr",
    prenormalize: bool | None = None,
    maxorder: int | None = None,
    maxiter: int | None = None,
    m: int | None = None,
    lambdax: float | None = None,
    chunk_size: int | None = None,
    order: int | None = None,
    ridge: float | None = None,
    fit_ratio: float | None = None,
) -> ShapleyResult:
    """Compute Shapley effects analytically from a fitted surrogate.

    Fits an RS-HDMR or PCE surrogate to arbitrary (X, Y) pairs -- no
    structured sampling required -- and allocates the output variance
    across inputs via the exact Shapley formula
    ``Sh_i = sum_{u : i in u} V_u / |u|``, where ``V_u`` are the
    surrogate's per-subset partial variances. No permutation sampling
    is involved.

    Inputs are assumed independent. Under independence the Shapley
    effects satisfy ``S1_i <= Sh_i <= ST_i`` and split each interaction's
    variance equally among its participants. All indices are normalized
    by the empirical ``Var(Y)``, so their sum equals the surrogate's
    explained-variance fraction; interactions beyond the surrogate's
    truncation order (``maxorder`` / ``order``) are absent from the
    allocation.

    Args:
        problem: Parameter names and distributions.
        X: (N, D) input samples.
        Y: Model outputs. ``backend="hdmr"`` accepts (N,), (N, K), or
            (N, T, K); ``backend="pce"`` accepts scalar (N,) only.
        backend: Surrogate providing the variance decomposition.
            ``"hdmr"`` (default) fits B-spline component functions and
            supports all output shapes; ``"pce"`` reads subset variances
            off orthonormal polynomial coefficients (Sudret, 2008).
        prenormalize: HDMR-only; see ``analyze_hdmr``. Defaults to False.
        maxorder: HDMR-only; maximum expansion order (1-3). Defaults to 2.
        maxiter: HDMR-only; backfitting iterations. Defaults to 100.
        m: HDMR-only; B-spline intervals per dimension. Defaults to 2.
        lambdax: HDMR-only; Tikhonov regularization. Defaults to 0.01.
        chunk_size: HDMR-only; output slices per vmap batch. Defaults to 2048.
        order: PCE-only; maximum total polynomial degree. Defaults to 3.
        ridge: PCE-only; Tikhonov regularization. Defaults to 1e-8.
        fit_ratio: PCE-only; maximum terms-to-samples ratio. Defaults to 0.5.

    Returns:
        ShapleyResult with ``Sh``, ``S1``, and ``ST`` (all normalized by
        empirical output variance), the problem, and the backend name.

    Raises:
        ValueError: If ``backend`` is unknown, or a kwarg belonging to the
            non-selected backend is explicitly set.
    """
    resolved = _resolve_backend_kwargs(
        backend,
        hdmr_kwargs={
            "prenormalize": prenormalize,
            "maxorder": maxorder,
            "maxiter": maxiter,
            "m": m,
            "lambdax": lambdax,
            "chunk_size": chunk_size,
        },
        pce_kwargs={"order": order, "ridge": ridge, "fit_ratio": fit_ratio},
    )

    if backend == "hdmr":
        result = analyze_hdmr(problem, X, Y, **resolved)
        emulator = result.emulator
        if emulator is None:  # pragma: no cover - analyze_hdmr always sets it
            raise RuntimeError("analyze_hdmr returned no emulator")
        # HDMR terms enumerate [singles, pairs, triples]; the emulator holds
        # the exact pair/triple index tuples used (post maxorder clamping).
        subsets: list[tuple[int, ...]] = [(i,) for i in range(problem.num_vars)]
        subsets += [tuple(u) for u in emulator["c2"]]
        subsets += [tuple(u) for u in emulator["c3"]]
        # Sa is each component function's variance / Var(Y): exactly the
        # normalized partial variances the Shapley formula needs.
        V = result.Sa
    else:
        pce_result = analyze_pce(problem, X, Y, **resolved)
        # Orthonormality makes each squared coefficient the partial variance
        # of its basis function; the constant term (row 0) carries none.
        partial = pce_result.coefficients[1:] ** 2
        active = pce_result.multi_index[1:] > 0  # (n_terms - 1, D)
        subsets = [tuple(int(d) for d in row.nonzero()[0]) for row in active]
        total_var = jnp.var(jnp.asarray(Y))
        inv_var = jnp.where(total_var == 0, jnp.nan, 1.0 / total_var)
        V = partial * inv_var

    membership = build_membership(subsets, problem.num_vars)
    Sh, S1, ST = shapley_from_variances(V, membership)

    return ShapleyResult(Sh=Sh, S1=S1, ST=ST, problem=problem, backend=backend)
