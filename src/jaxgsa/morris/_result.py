"""Result container for Morris elementary-effects screening measures."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
from jax import Array

from jaxgsa._core.invalid import InvalidReport
from jaxgsa._core.result import CIInfo, FieldSpec, ResultSchema, SchemaResult
from jaxgsa.problem import Problem


@dataclass(repr=False)
class MorrisResult(SchemaResult):
    """Morris elementary-effects screening measures.

    Measure arrays have shape ``(D,)`` for a scalar output, ``(K, D)`` for a
    multi-output model, and ``(T, K, D)`` for a time-resolved analysis.

    The analysis computes elementary effects in unit-cube coordinates by
    default, which the ``space == "unit"`` field records. In that space
    ``mu_star`` compares directly across parameters with different ranges. Use
    :meth:`to_physical_units` for derivative-scale values in the problem's
    native units.

    Attributes:
        mu: Mean elementary effect, shape ``(..., D)``. The sign is kept, so
            effects of opposite sign can cancel and hide a non-monotonic
            parameter.
        mu_star: Mean absolute elementary effect, shape ``(..., D)``
            (Campolongo et al. 2007). This is the headline importance measure
            and a proxy for total-order ranking.
        sigma: Standard deviation of the elementary effects (ddof=1), shape
            ``(..., D)``. A value that is large next to ``mu_star`` shows
            nonlinearity or interactions.
        problem: Problem definition used for the analysis.
        invalid: What the non-finite check found and what it did about it. A
            report with ``n_invalid == 0`` means the check ran and found
            nothing. See :class:`jaxgsa._core.invalid.InvalidReport`.
        mu_conf: Bootstrap confidence bounds on ``mu``, shape ``(2, ...)`` for
            ``[lower, upper]``. ``None`` when the analysis ran no bootstrap.
        mu_star_conf: Bootstrap confidence bounds on ``mu_star``, shape
            ``(2, ...)`` for ``[lower, upper]``. ``None`` when the analysis ran
            no bootstrap.
        sigma_conf: Bootstrap confidence bounds on ``sigma``, shape
            ``(2, ...)`` for ``[lower, upper]``. ``None`` when the analysis ran
            no bootstrap.
        space: Coordinate space of the measures, ``"unit"`` or ``"physical"``.
        ci: How the intervals were produced: the confidence level, the
            endpoint rule, the resample count, and the bootstrap draws when
            the analysis ran with ``keep_replicates=True``. ``None`` without
            a bootstrap. :meth:`to_physical_units` rescales any kept draws
            along with the measures. See
            :class:`jaxgsa._core.result.CIInfo`.
    """

    mu: Array
    mu_star: Array
    sigma: Array
    problem: Problem
    invalid: InvalidReport
    mu_conf: Array | None = None
    mu_star_conf: Array | None = None
    sigma_conf: Array | None = None
    space: Literal["unit", "physical"] = "unit"
    ci: CIInfo | None = None

    _schema = ResultSchema(
        primary="mu",
        fields=(
            FieldSpec("mu", "param", interval=True),
            FieldSpec("mu_star", "param", interval=True),
            FieldSpec("sigma", "param", interval=True),
        ),
        meta=("space",),
    )

    def _dataset_attrs(self) -> dict[str, Any]:
        """Record the coordinate space the measures are in."""
        return {"space": self.space}

    def to_physical_units(self) -> MorrisResult:
        """Return a copy with measures rescaled to physical input units.

        A unit-cube elementary effect divides the output change by a step in
        ``[0, 1]`` coordinates. Dividing each measure by the parameter range
        ``high - low`` turns it into a per-physical-unit effect. That is a
        derivative scale, comparable to DGSM's mean derivative.

        Returns:
            A new :class:`MorrisResult` with ``space == "physical"``.

        Raises:
            ValueError: If the result is already in physical units or the
                problem has no finite bounds.
        """
        if self.space == "physical":
            raise ValueError("Result is already in physical units")
        if self.problem.bounds is None:
            raise ValueError("to_physical_units requires a problem with finite uniform bounds")

        # Ranges broadcast against the trailing parameter axis of every field.
        ranges = np.asarray([high - low for low, high in self.problem.bounds])

        def _scale(arr: Array | None) -> Array | None:
            return None if arr is None else arr / ranges

        # Kept bootstrap draws are measures too. Leaving them in unit-cube
        # coordinates would make them disagree with the intervals beside them.
        ci = self.ci
        if ci is not None and ci.replicates is not None:
            ci = replace(ci, replicates={k: v / ranges for k, v in ci.replicates.items()})

        return replace(
            self,
            ci=ci,
            mu=self.mu / ranges,
            mu_star=self.mu_star / ranges,
            sigma=self.sigma / ranges,
            mu_conf=_scale(self.mu_conf),
            mu_star_conf=_scale(self.mu_star_conf),
            sigma_conf=_scale(self.sigma_conf),
            space="physical",
        )
