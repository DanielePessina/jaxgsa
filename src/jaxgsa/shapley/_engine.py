"""Shapley-effect aggregation from ANOVA subset variances.

Implements the exact Shapley-value allocation of output variance across
inputs (Owen, 2014). Given the partial variance ``V_u`` of every modelled
ANOVA subset ``u``, the Shapley effect of input ``i`` is
``Sh_i = sum_{u : i in u} V_u / |u|``. Each term's variance is split equally
among its participants. The aggregation is a single weighted matrix product,
batched over any leading output dimensions.

:func:`_shapley_result_from_variances` is the shared tail both surrogate
backends finish with. It lives here rather than in ``_analyze`` because
``PCEResult`` and ``HDMRResult`` call it, and a method package should not have
to import a sibling's private analysis module to reach it.
"""

from __future__ import annotations

import inspect
import math
import os
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxgsa._core.invalid import InvalidReport
from jaxgsa._core.warning_types import JaxgsaWarning
from jaxgsa.shapley._result import ShapleyResult

if TYPE_CHECKING:
    from jaxgsa.problem import Problem


def build_membership(subsets: Sequence[tuple[int, ...]], D: int) -> np.ndarray:
    """Build the boolean membership matrix of ANOVA terms over parameters.

    Args:
        subsets: One parameter-index tuple per modelled ANOVA term,
            e.g. ``[(0,), (1,), (2,), (0, 1), (0, 2)]``.
        D: Number of parameters.

    Returns:
        Boolean array of shape ``(n_terms, D)`` where entry ``[t, d]`` marks
        whether parameter ``d`` participates in term ``t``.
    """
    membership = np.zeros((len(subsets), D), dtype=bool)
    for t, u in enumerate(subsets):
        membership[t, list(u)] = True
    return membership


def shapley_from_variances(
    V: Array,
    membership: np.ndarray,
) -> tuple[Array, Array, Array]:
    """Aggregate per-term variance fractions into Shapley, S1, and ST indices.

    Args:
        V: Variance fraction of each modelled ANOVA term, shape
            ``(..., n_terms)``. It is already normalized so the modelled
            terms sum to 1, that is, divided by the total decomposed variance
            ``sum_u V_u``. Leading dimensions (output, time) are batched
            through unchanged.
        membership: Boolean membership matrix from
            :func:`build_membership`, shape ``(n_terms, D)``.

    Returns:
        Tuple ``(Sh, S1, ST)``, each of shape ``(..., D)``:
            Sh: Shapley effects ``sum_{u:i in u} V_u / |u|``.
            S1: First-order indices (singleton terms only).
            ST: Total-order indices ``sum_{u:i in u} V_u``.
    """
    card = membership.sum(axis=1)  # |u| per term
    M = jnp.asarray(membership, dtype=V.dtype)
    # The three indices only differ in how a term's variance is credited to
    # its participants: split evenly (Shapley), singletons only (S1), or in
    # full (ST). All three are therefore weighted variants of one matmul.
    Sh = (V / jnp.asarray(card, dtype=V.dtype)) @ M
    S1 = V @ jnp.asarray(membership & (card[:, None] == 1), dtype=V.dtype)
    ST = V @ M
    return Sh, S1, ST


_POORFIT_THRESHOLD = 0.5
_OVERFIT_THRESHOLD = 1.3

# The out-of-sample twin of _POORFIT_THRESHOLD, for the PCE backend.
#
# ``explained_variance`` is a genuine in-sample R-squared, so it cannot exceed
# 1 and the _OVERFIT_THRESHOLD branch below is unreachable for PCE. (It stays
# live for HDMR, whose diagnostic is ``sum(V_u) / Var(Y)`` and can legitimately
# pass 1.) The signal that does move under a PCE overfit is the leave-one-out
# error turning back up towards the spread of the data itself: predicting every
# row by the output mean already scores ``loo_rmse == std(Y)``, so a ratio
# anywhere near 1 says the expansion carries no usable predictive information,
# whatever its in-sample fit looks like.
#
# The threshold is placed where the leave-one-out R-squared falls to
# _POORFIT_THRESHOLD, that is ``1 - ratio**2 < 0.5``, so both warnings fire at
# "half the variance is unaccounted for" and only the sample they measure on
# differs. The measured cases sit well clear of it on both sides. An order-5
# PCE on 64 Ishigami rows (fit_ratio=0.9) gives loo_rmse 9.59 against std(Y)
# 3.59, a ratio of 2.67, while its in-sample explained_variance reads 0.98 and
# so reports nothing wrong at all. An order-8 fit on 2000 rows gives loo_rmse
# 0.076 against std(Y) 3.75, a ratio of 0.020.
_LOO_RATIO_THRESHOLD = math.sqrt(1.0 - _POORFIT_THRESHOLD)


def _external_stacklevel(default: int = 2) -> int:
    """``warnings`` stacklevel of the first caller frame outside jaxgsa.

    A shared tail emits the Shapley fit-quality warning, and two call chains
    of different depth reach that tail: ``result.shapley()`` directly, or the
    ``jaxgsa.shapley.analyze`` wrapper, which adds a frame. No fixed
    ``stacklevel`` points at the user's frame in both cases. Walking out of
    the package locates it regardless of chain length.

    Args:
        default: Fallback level for interpreters without frame introspection.

    Returns:
        The ``stacklevel`` that attributes a warning issued by the immediate
        caller to the nearest frame outside the ``jaxgsa`` package.
    """
    import jaxgsa

    frame = inspect.currentframe()
    if frame is None:  # pragma: no cover - non-CPython fallback
        return default
    pkg_root = os.path.dirname(jaxgsa.__file__)
    frame = frame.f_back  # the frame that will call warnings.warn
    level = 1
    while frame is not None:
        if not frame.f_code.co_filename.startswith(pkg_root):
            return level
        frame = frame.f_back
        level += 1
    return level


def _normalize_partial_variances(
    partial: Array,
    explained_variance: Array,
    total: Array,
) -> Array:
    """Normalize term contributions while preserving degenerate output slices.

    Args:
        partial: Per-term variance contributions, shape ``(..., n_terms)``.
        explained_variance: Per-slice fit diagnostic, shape ``(...,)``;
            non-finite slices are propagated as NaN.
        total: Pre-computed ``partial.sum(axis=-1)``, shape ``(...,)``.
            Passed in so callers that already hold the sum do not trigger a
            recomputation.

    Returns:
        ``partial / total`` with degenerate slices (zero or non-finite total,
        non-finite explained variance) set to NaN.
    """
    total = jnp.asarray(total)[..., None]
    invalid = (
        (total == 0)
        | ~jnp.isfinite(total)
        | ~jnp.isfinite(jnp.asarray(explained_variance)[..., None])
    )
    return jnp.where(invalid, jnp.nan, partial / jnp.where(invalid, 1.0, total))


def _warn_pathological_fit(explained_variance: Array) -> None:
    """Warn when a surrogate captured implausibly little or too much variance.

    Two call chains of different depth reach this function:
    ``result.shapley()`` and the ``jaxgsa.shapley.analyze`` wrapper.
    :func:`_external_stacklevel` therefore resolves the warning's
    ``stacklevel`` dynamically, so the warning points at the user's frame in
    either case.
    """
    ev = jnp.asarray(explained_variance)
    if bool(jnp.any(ev > _OVERFIT_THRESHOLD)):
        warnings.warn(
            f"jaxgsa.shapley: surrogate explained_variance exceeds {_OVERFIT_THRESHOLD}; "
            "Shapley effects may be unreliable",
            stacklevel=_external_stacklevel(),
            category=JaxgsaWarning,
        )
    elif bool(jnp.any(ev < _POORFIT_THRESHOLD)):
        warnings.warn(
            f"jaxgsa.shapley: surrogate explained_variance is below {_POORFIT_THRESHOLD}; "
            "Shapley effects may be unreliable",
            stacklevel=_external_stacklevel(),
            category=JaxgsaWarning,
        )


def _warn_pce_overfit(
    loo_rmse: Array | None, std_y: Array, explained_variance: Array | None = None
) -> None:
    """Warn when a PCE surrogate's leave-one-out error approaches ``std(Y)``.

    This is the PCE backend's overfit signal, and it replaces nothing that
    still works: :func:`_warn_pathological_fit` catches a surrogate that missed
    variance in sample, and its opposite branch cannot fire for PCE now that
    ``explained_variance`` is a true R-squared. An overfit expansion scores
    near 1 in sample, so only an out-of-sample number sees it. Predicting every
    row by the output mean already gives ``loo_rmse == std(Y)``, which makes
    that ratio the natural scale to read the error on. See
    :data:`_LOO_RATIO_THRESHOLD` for where the line sits and why.

    The check is per output slice, and one warning is emitted for the worst
    slice rather than one per slice. Slices whose ratio is not finite (a
    constant output has no spread to divide by) are skipped.

    Args:
        loo_rmse: Leave-one-out RMSE per output slice, in the units of ``Y``,
            or ``None`` when the surrogate reported none, which silences the
            check.
        std_y: Sample standard deviation of ``Y`` per output slice, over the
            rows the surrogate was fitted on. Same shape as ``loo_rmse``.

    Warns:
        JaxgsaWarning: If any slice's ratio passes the threshold.
    """
    if loo_rmse is None:
        return
    ratio = jnp.asarray(loo_rmse) / jnp.asarray(std_y)
    finite = ratio[jnp.isfinite(ratio)]
    if finite.size == 0:
        return
    worst = float(jnp.max(finite))
    if worst <= _LOO_RATIO_THRESHOLD:
        return
    # Only speak when the surrogate looks healthy in sample. An expansion that
    # already missed variance on the rows it was fitted to is underfitting, and
    # _warn_pathological_fit says exactly that. Calling the same fit overfit as
    # well would be two warnings for one problem, and the wrong name for it.
    # A real overfit scores near 1 in sample by definition, so this gate costs
    # no sensitivity.
    if explained_variance is not None:
        in_sample = jnp.asarray(explained_variance)
        healthy = in_sample[jnp.isfinite(in_sample)]
        if healthy.size and float(jnp.min(healthy)) < _POORFIT_THRESHOLD:
            return
    warnings.warn(
        f"jaxgsa.shapley: the PCE surrogate's loo_rmse is {worst:.2f} times std(Y) "
        "on at least one output slice, so it predicts less than half of the output "
        "variance out of sample. A leave-one-out error approaching std(Y) is the "
        "signature of an overfit expansion, which a high in-sample "
        "explained_variance will not show. Refit with a lower order or more "
        "samples; the Shapley effects from this fit may be unreliable.",
        stacklevel=_external_stacklevel(),
        category=JaxgsaWarning,
    )


def _shapley_result_from_variances(
    partial: Array,
    membership: np.ndarray,
    explained: Array,
    *,
    total: Array,
    problem: "Problem",
    backend: Literal["hdmr", "pce"],
    order: int,
    invalid: InvalidReport,
    include_correlative: bool = False,
) -> ShapleyResult:
    """Finish a Shapley analysis from a surrogate's variance decomposition.

    This is the single shared tail of both ``shapley()`` result methods. It
    normalizes the partial variances so the modelled terms sum to 1,
    allocates each term's share equally among its participants, warns on a
    pathological fit, and assembles the result. Callers perform their
    fitted-state precheck and decomposition before delegating here.

    Args:
        partial: Per-term variance contributions, shape ``(..., n_terms)``.
        membership: Boolean matrix marking which parameters participate in
            each term, shape ``(n_terms, D)``.
        explained: Per-slice explained-variance diagnostic, shape ``(...,)``.
        total: Pre-computed ``partial.sum(axis=-1)`` (for HDMR this is the
            same array as ``explained``, so passing it avoids a recompute).
        problem: Problem definition carried onto the result.
        backend: Surrogate backend label, ``"hdmr"`` or ``"pce"``.
        order: Effective surrogate order actually used.
        invalid: The non-finite report the surrogate's ``analyze`` produced,
            carried through unchanged. This tail never applies the policy
            itself; the backend already did, exactly once.
        include_correlative: Whether the correlative ANCOVA part was folded
            into ``partial`` (HDMR only).

    Returns:
        ShapleyResult with ``Sh``/``S1``/``ST``, the diagnostic, and
        provenance fields.

    Warns:
        JaxgsaWarning: If ``explained`` flags a pathological fit (well below
            1, or above 1, which is an overfit).
    """
    normalized = _normalize_partial_variances(partial, explained, total)
    Sh, S1, ST = shapley_from_variances(normalized, membership)
    _warn_pathological_fit(explained)
    return ShapleyResult(
        Sh=Sh,
        S1=S1,
        ST=ST,
        problem=problem,
        backend=backend,
        explained_variance=explained,
        order=order,
        invalid=invalid,
        include_correlative=include_correlative,
    )
