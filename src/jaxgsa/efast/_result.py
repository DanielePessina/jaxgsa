"""Defines the EFASTResult dataclass for eFAST sensitivity analysis."""

from dataclasses import dataclass

from jax import Array

from jaxgsa._core.invalid import InvalidReport
from jaxgsa._core.result import FieldSpec, ResultSchema, SchemaResult
from jaxgsa.problem import Problem


@dataclass(repr=False)
class EFASTResult(SchemaResult):
    """Extended FAST sensitivity analysis results.

    Stores first-order (S1) and total-order (ST) Sobol indices computed from a
    Fourier amplitude decomposition. eFAST produces no second-order
    interaction indices.

    Index shapes mirror the shape of the analyzed output Y: ``(D,)`` for a
    scalar output, ``(K, D)`` when the time dimension is squeezed, or
    ``(T, K, D)`` for time-resolved analyses. *D* is the number of parameters,
    *K* the number of outputs, and *T* the number of time steps.

    Attributes:
        S1: First-order indices, shape ``(D,)`` / ``(K, D)`` / ``(T, K, D)``.
        ST: Total-order indices, same shape as ``S1``.
        problem: Problem definition used for the analysis.
        invalid: What the non-finite check found and what it did about it. A
            report with ``n_invalid == 0`` means the check ran and found
            nothing. A search curve can never be dropped, so the report is
            only ever informational here. See
            :class:`jaxgsa._core.invalid.InvalidReport`.
        omega_0: Primary frequency used in the analysis.
        M: Interference factor, the number of harmonics summed.
    """

    S1: Array
    ST: Array
    problem: Problem
    invalid: InvalidReport
    omega_0: int
    M: int

    _schema = ResultSchema(
        primary="S1",
        fields=(FieldSpec("S1"), FieldSpec("ST")),
        meta=("omega_0", "M"),
    )
