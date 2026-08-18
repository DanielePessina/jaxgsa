"""Shapley-effect result dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from jax import Array

from jaxgsa._core.invalid import InvalidReport
from jaxgsa._core.result import FieldSpec, ResultSchema, SchemaResult
from jaxgsa.problem import Problem


@dataclass(repr=False)
class ShapleyResult(SchemaResult):
    """Shapley-effect sensitivity analysis results.

    Stores Shapley effects computed analytically from a fitted surrogate's
    variance decomposition, alongside the first-order (S1) and total-order
    (ST) indices derived from the same decomposition so the ordering
    ``S1 <= Sh <= ST`` is visible at a glance.

    All indices are normalized by the surrogate's total decomposed variance
    ``sum_u V_u``, so ``Sh.sum(axis=-1)`` is exactly 1 (the Shapley-value
    efficiency property; Owen 2014). ``explained_variance`` reports
    separately how much of the output variance the surrogate captured.

    For the ``"hdmr"`` backend the indices are normalized by ``sum_u V_u``
    rather than ``Var(Y)``. They therefore relate to HDMR indices by a factor
    of ``explained_variance``: multiply to recover the HDMR scale. For PCE,
    S1/ST match the fitted result exactly.

    Shapes mirror the layout of the ``Y`` used to fit the source surrogate:
    ``(D,)`` for scalar output, ``(K, D)`` for multi-output, and
    ``(T, K, D)`` for time-resolved analyses.

    Attributes:
        Sh: Shapley effects, shape ``(D,)`` / ``(K, D)`` / ``(T, K, D)``.
            Sums to 1 along the parameter axis.
        S1: First-order indices from the same surrogate, shape ``(D,)`` /
            ``(K, D)`` / ``(T, K, D)``.
        ST: Total-order indices from the same surrogate, shape ``(D,)`` /
            ``(K, D)`` / ``(T, K, D)``.
        problem: Problem definition used for the analysis.
        backend: Surrogate backend used, ``"hdmr"`` or ``"pce"``.
        explained_variance: Fraction of ``Var(Y)`` captured by the surrogate,
            ``sum_u V_u / Var(Y)``, shape ``()`` / ``(K,)`` / ``(T, K)``.
            Close to 1 for a good fit, below 1 when truncation or fit error
            leaves variance unexplained, and above 1 when an overfit surrogate
            over-counts shared variance.
        order: Effective surrogate order actually used. For ``"pce"`` it is
            the polynomial degree, which may be reduced from the requested
            value to fit the sample budget. For ``"hdmr"`` it is the HDMR
            expansion order.
        invalid: What the non-finite check found in ``(X, Y)`` and what the
            ``on_invalid`` policy did about it, as reported by whichever
            backend ran. See :class:`jaxgsa._core.invalid.InvalidReport`.
        include_correlative: Whether the correlative ANCOVA variance (``Sb``)
            was folded into the allocation (HDMR backend only). When ``True``
            the indices credit variance shared through input correlation, so
            ``Sh``/``S1``/``ST`` may be negative and the ordering
            ``S1 <= Sh <= ST`` need not hold. Efficiency (``Sh`` sums to 1) is
            preserved regardless.
    """

    Sh: Array
    S1: Array
    ST: Array
    problem: Problem
    backend: Literal["hdmr", "pce"]
    explained_variance: Array
    order: int
    invalid: InvalidReport
    include_correlative: bool = False

    _schema = ResultSchema(
        primary="Sh",
        fields=(
            FieldSpec("Sh"),
            FieldSpec("S1"),
            FieldSpec("ST"),
            FieldSpec("explained_variance", "slice"),
        ),
        meta=("backend", "order", "include_correlative"),
    )

    def _dataset_attrs(self) -> dict[str, Any]:
        """Record which surrogate produced the effects, and on what basis."""
        return {"backend": self.backend, "include_correlative": self.include_correlative}
