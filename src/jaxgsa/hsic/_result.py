"""Result container for HSIC kernel-based sensitivity indices."""

from __future__ import annotations

from dataclasses import dataclass

from jax import Array

from jaxgsa._core.invalid import InvalidReport
from jaxgsa._core.result import FieldSpec, ResultSchema, SchemaResult
from jaxgsa.problem import Problem


@dataclass(frozen=True, repr=False)
class HSICResult(SchemaResult):
    """HSIC (Hilbert-Schmidt Independence Criterion) sensitivity analysis results.

    Index arrays have shape ``(D,)`` for a scalar output, ``(K, D)`` for a
    multi-output model, and ``(T, K, D)`` for a time-resolved analysis.

    Attributes:
        R2_HSIC: Normalized first-order HSIC indices, shape ``(..., D)``.
            Each entry is ``HSIC(x_i, Y)`` divided by
            ``sqrt(HSIC(x_i, x_i) * HSIC(Y, Y))`` and lies in [0, 1]. A value
            of 0 means the parameter and the output are independent, and
            larger values mean stronger dependence.
        T_HSIC: Total-order HSIC indices, shape ``(..., D)``. Each entry is
            the fraction of the joint dependence lost when parameter i is
            removed, so it also counts influence carried through interactions
            (analogous to ST).
        p_values: Permutation-test p-values, shape ``(..., D)``, for the null
            hypothesis that ``x_i`` and ``Y`` are independent. Small values
            mean the detected dependence is unlikely to be sampling noise.
        hsic_raw: Unnormalized ``HSIC(x_i, Y)`` values, shape ``(..., D)``.
            They depend on the kernel and on the scale of the data, so
            compare them only within one analysis.
        problem: Problem definition used for the analysis.
        invalid: What the non-finite check found in the sample, and which
            ``on_invalid`` policy ran. ``invalid.n_invalid == 0`` means the
            check ran and found nothing.
    """

    R2_HSIC: Array
    T_HSIC: Array
    p_values: Array
    hsic_raw: Array
    problem: Problem
    invalid: InvalidReport

    _schema = ResultSchema(
        primary="R2_HSIC",
        fields=(
            FieldSpec("R2_HSIC"),
            FieldSpec("T_HSIC"),
            FieldSpec("p_values"),
            FieldSpec("hsic_raw"),
        ),
    )
