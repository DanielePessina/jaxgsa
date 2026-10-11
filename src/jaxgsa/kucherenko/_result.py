"""Result container for Kucherenko dependent-input sensitivity indices."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Literal, Self, cast

import jax.numpy as jnp
import numpy as np
import xarray as xr
from jax import Array

from jaxgsa._core.invalid import InvalidReport
from jaxgsa._core.result import (
    CIInfo,
    FieldSpec,
    ResultSchema,
    SchemaResult,
    aggregate_indices,
)
from jaxgsa.kucherenko._subsets import pme_from_total, shapley_from_cost
from jaxgsa.problem import Dependence, Problem

if TYPE_CHECKING:
    from jaxgsa.shapley import ShapleyResult


@dataclass(frozen=True, repr=False)
class KucherenkoResult(SchemaResult):
    """Kucherenko sensitivity indices, returned by :func:`jaxgsa.kucherenko.analyze`.

    Index arrays have shape ``(D,)`` for a scalar output, ``(K, D)`` for a
    multi-output model, and ``(T, K, D)`` for a time-resolved analysis.

    ``S1`` estimates ``V(E(Y|X_i)) / V(Y)`` and ``ST`` estimates
    ``E(V(Y|X_{~i})) / V(Y)`` under the problem's declared dependence
    structure. Under independent inputs they are the classic Sobol' first-order
    and total-order indices. Under a declared correlation ``S1`` is
    correlation-inclusive and matches ``jaxgsa.vkoga``'s ``S_TC``, and ``ST``
    is correlation-exclusive and matches ``S_TU``. In that case ``ST >= S1`` no
    longer holds in general.

    Attributes:
        S1: First-order (correlation-inclusive) indices, shape ``(..., D)``.
            Rank parameters by ``S1`` to decide which ones to measure.
        ST: Total-order (correlation-exclusive) indices, shape ``(..., D)``.
            Rank parameters by ``ST`` to decide which ones to fix.
        problem: Problem definition used for the analysis.
        dependence: How the inputs depend on each other: ``"independent"``,
            ``"gaussian"`` or ``"custom"`` (a sampler declared on the problem).
        variance: Output variance under the joint input measure, one value per
            output slice (shape ``()``, ``(K,)``, or ``(T, K)``).
        invalid: What the non-finite check found and what it did about it. A
            report with ``n_invalid == 0`` means the check ran and found
            nothing. See :class:`jaxgsa._core.invalid.InvalidReport`.
        S1_conf: Bootstrap confidence interval for ``S1``, shape ``(2, ...)``
            for ``[lower, upper]``. ``None`` when ``n_bootstrap=0``.
        ST_conf: The same for ``ST``.
        ci: How the intervals were produced: the confidence level, the
            endpoint rule, the resample count, and the bootstrap draws when
            the analysis ran with ``keep_replicates=True``. ``None`` without a
            bootstrap. The resampling unit is one base point, not one design
            row. ``variance`` carries no interval, because it is the
            denominator of both indices rather than a sensitivity measure.
            See :class:`jaxgsa._core.result.CIInfo`.
        closed: Closed Sobol' index ``V(E(Y|X_u)) / V(Y)`` of every subset
            ``u``, shape ``(..., 2^D - 1)``. ``None`` unless the design came
            from ``sample(..., subsets="all")``. The subsets run in the order
            of their bitmask ``m = 1 .. 2^D - 1`` (bit ``j`` is parameter
            ``j``), and subset ``m`` sits at position ``m - 1``. The last
            entry is the full set, which is 1.
        subset_total: Total Sobol' index ``E(V(Y|X_{~u})) / V(Y)`` of every
            subset ``u``, with the same layout as ``closed``. The singleton
            entries equal ``ST``. This is the cost that ``shapley()`` and
            ``pme()`` use. It is named ``subset_total`` so that it does not
            read as ``ST``, which is the same index for single inputs only.
        subsets: Membership matrix of the subsets, boolean, shape
            ``(2^D - 1, D)``. Row ``m - 1`` marks the members of subset ``m``.
        closed_conf: Bootstrap interval for ``closed``, shape
            ``(2, ..., 2^D - 1)``.
        subset_total_conf: Bootstrap interval for ``subset_total``, the same
            shape.
    """

    S1: Array
    ST: Array
    problem: Problem
    variance: Array
    invalid: InvalidReport
    S1_conf: Array | None = None
    ST_conf: Array | None = None
    ci: CIInfo | None = None
    dependence: Dependence = "independent"
    closed: Array | None = None
    subset_total: Array | None = None
    subsets: np.ndarray | None = None
    closed_conf: Array | None = None
    subset_total_conf: Array | None = None

    _schema = ResultSchema(
        primary="S1",
        fields=(
            FieldSpec("S1", interval=True),
            FieldSpec("ST", interval=True),
            FieldSpec("variance", "slice"),
        ),
        meta=("dependence",),
    )

    def aggregate(self, *, weights: Literal["variance", "equal"] = "variance") -> Self:
        """Collapse the output axes into one index per input.

        Computes ``S_agg_i = sum_t w_t * S_{i,t}`` over every output slice
        ``t`` (the ``T`` and ``K`` axes together). With ``weights="variance"``
        the weight is ``w_t = V_t / sum(V)``, where ``V_t`` is the variance
        of output ``t``. This is the generalised Sobol index of Gamboa,
        Janon, Klein and Lagnoux (2014, Electron. J. Stat. 8:575-603,
        doi:10.1214/14-EJS895), also used by Lamboni, Monod and Makowski
        (2011, Reliab. Eng. Syst. Saf. 96:450-459,
        doi:10.1016/j.ress.2010.12.002). It is the share of the total output
        variance that the input explains. With ``weights="equal"`` every
        output counts the same, ``w_t = 1 / n_outputs``.

        The method applies to every index field this result holds. The
        result keeps its class and its output axes, now of size 1, so
        ``to_dataset`` and the printed summary work as before.

        When the result holds subset indices, ``closed`` and ``subset_total``
        collapse the same way. Both subset costs are linear in the output
        weights, so ``shapley()`` of the aggregated result is the
        variance-weighted average of the per-output Shapley effects.

        The ``variance`` of the returned result holds the summed variance of
        the outputs. The weights come from this variance, which is taken
        under the joint input measure.

        A slice of zero variance (a constant output) has weight 0 and adds
        nothing. A result with one output slice comes back with the same
        index values.

        Confidence intervals do not aggregate bound by bound. When the
        analysis ran with ``keep_replicates=True``, each bootstrap draw is
        aggregated and the interval is drawn again from those draws, at the
        same level and with the same rule. The weights stay fixed at the
        point-estimate values in every draw, so the interval leaves out the
        sampling noise of the weights themselves. Otherwise the ``*_conf`` fields
        and ``ci`` become ``None`` and one ``JaxgsaWarning`` says that
        ``keep_replicates=True`` is needed.

        The output axes collapse to size 1. A result with no output axes
        (``S1`` of shape ``(D,)``) is returned unchanged.

        Args:
            weights: ``"variance"`` (default) or ``"equal"``.

        Returns:
            A result of the same class with output axes of size 1.

        Raises:
            ValueError: If ``weights`` is not one of the two names.

        Warns:
            JaxgsaWarning: If the result has bootstrap intervals but kept no
                replicates, so the intervals are dropped.
        """
        view = self
        if self.closed is not None:
            # Treat the subset indices as extra index fields. The collapse acts
            # on the last axis whatever its length, so the same code serves.
            view = cast(
                Self,
                SimpleNamespace(
                    **{f.name: getattr(self, f.name) for f in dataclasses.fields(self)},
                    _schema=ResultSchema(
                        primary="S1",
                        fields=(
                            *self._schema.fields,
                            FieldSpec("closed", interval=True),
                            FieldSpec("subset_total", interval=True),
                        ),
                    ),
                ),
            )
        updates = aggregate_indices(view, weights=weights, variance=self.variance)
        if updates:
            updates["variance"] = self.variance.sum(keepdims=True)
        return dataclasses.replace(self, **updates)

    # -- subset allocations ------------------------------------------------

    def _require_subsets(self, what: str) -> tuple[np.ndarray, int]:
        """Return ``(subset_total, D)`` or explain that the design lacked subsets."""
        if self.closed is None or self.subset_total is None:
            raise ValueError(
                f"{what} needs the closed and total index of every subset. This result "
                "comes from a design with subsets='singletons'. Draw the design with "
                "jaxgsa.kucherenko.sample(problem, n, subsets='all') and analyze that."
            )
        return np.asarray(self.subset_total), self.problem.n_params

    def _allocation_result(
        self,
        values: np.ndarray,
        conf: Array | None,
        ci: CIInfo | None,
    ) -> "ShapleyResult":
        """Wrap an ``(..., D)`` allocation in a :class:`ShapleyResult`."""
        from jaxgsa.shapley import ShapleyResult

        return ShapleyResult(
            Sh=jnp.asarray(values),
            S1=self.S1,
            ST=self.ST,
            problem=self.problem,
            # No surrogate is fitted. "exact" marks an allocation of exact
            # subset indices; the field is typed for the two surrogate backends.
            backend="exact",
            # A surrogate fit quality and a surrogate order mean nothing for an
            # exact design, so both are None rather than a made-up value.
            explained_variance=None,
            order=None,
            invalid=self.invalid,
            Sh_conf=conf,
            S1_conf=self.S1_conf,
            ST_conf=self.ST_conf,
            ci=ci,
        )

    def _allocate(
        self,
        rule: Literal["shapley_total", "shapley_closed", "pme"],
    ) -> "ShapleyResult":
        """Apply an allocation rule to the point estimate and to every replicate."""
        total, D = self._require_subsets("shapley() and pme()")
        assert self.closed is not None

        def apply(closed: np.ndarray, total: np.ndarray) -> np.ndarray:
            if rule == "shapley_closed":
                return shapley_from_cost(closed, D)
            if rule == "shapley_total":
                return shapley_from_cost(total, D)
            return pme_from_total(total, D)

        point = apply(np.asarray(self.closed), total)
        conf = None
        ci = self.ci
        reps = None if ci is None else ci.replicates
        if ci is not None and reps is not None and "closed" in reps and "subset_total" in reps:
            from jaxgsa._core.bootstrap import interval

            draws = apply(np.asarray(reps["closed"]), np.asarray(reps["subset_total"]))
            confs, new_ci = interval(
                {"Sh": jnp.asarray(point)},
                {"Sh": jnp.asarray(draws)},
                level=ci.level,
                method=cast(Literal["quantile", "gaussian"], ci.method),
                n_bootstrap=ci.n_bootstrap,
                keep_replicates=True,
            )
            conf, ci = confs["Sh"], new_ci
        else:
            # Without kept replicates there is nothing to transform, so the
            # interval is unknown. NaN says so, and the shape stays regular.
            conf = jnp.full((2, *point.shape), jnp.nan)
            ci = None
        return self._allocation_result(point, conf, ci)

    def shapley(self, *, cost: Literal["total", "closed"] = "total") -> "ShapleyResult":
        """Exact Shapley effects of the inputs, from the subset indices.

        Computes ``Sh_i = sum_{u not containing i} |u|! (D-|u|-1)! / D! *
        [c(u + i) - c(u)]`` over all ``2^(D-1)`` subsets (Owen 2014,
        doi:10.1137/130936233), with ``c(empty) = 0`` and ``c(full) = 1``.
        The effects are conditional-variance Shapley effects, so they stay
        valid when the inputs are dependent. They sum to 1.

        The closed cost and the total cost give the same Shapley effects
        (Song, Nelson & Staum 2016, doi:10.1137/15M1048070). The default is
        the total cost, which they recommend for its lower variance. Pass
        ``cost="closed"`` for the other. The two agree within Monte Carlo
        error. The effects are nonnegative in expectation. A single estimate
        can be slightly negative from sampling noise, and is not clipped.

        If the analysis ran with ``n_bootstrap > 0`` and
        ``keep_replicates=True``, every bootstrap draw is transformed and
        ``Sh_conf`` is the interval of the transformed draws. Otherwise
        ``Sh_conf`` is ``NaN``.

        Args:
            cost: ``"total"`` (default) or ``"closed"``.

        Returns:
            A :class:`~jaxgsa.shapley.ShapleyResult` with ``backend="exact"``.
            No surrogate is fitted, so the surrogate-only fields
            ``explained_variance`` and ``order`` are ``None``. They are left
            out of ``to_dataset()``. ``S1`` and ``ST``
            are this result's own indices.

        Raises:
            ValueError: If the result does not hold subset indices, because the
                design used ``subsets="singletons"``, or if ``cost`` is not one
                of the two names.
        """
        if cost not in ("total", "closed"):
            raise ValueError(f"cost must be 'total' or 'closed', got {cost!r}")
        return self._allocate("shapley_total" if cost == "total" else "shapley_closed")

    def pme(self) -> "ShapleyResult":
        """Proportional marginal effects of the inputs (Herin et al. 2024).

        The PME are the proportional values of the total-index game
        ``A -> E[V(Y | X_{~A})] / V(Y)`` (Definition 4 of Herin, Il Idrissi,
        Chabridon & Iooss 2024, SIAM/ASA J. Uncertain. Quantif. 12(2):667-692,
        doi:10.1137/22M153032X, arXiv:2210.13065). Like the Shapley effects,
        they are nonnegative and sum to 1. Unlike them, an exogenous input
        gets exactly 0. An exogenous input is one the model does not use,
        even if it is correlated with an input that the model uses.

        A subset whose total index is exactly 0 is a null coalition. The
        continuous extension of Theorem 1 (Eq. 15) then applies. Sampling
        noise rarely gives an exact 0, so an input that is only nearly
        exogenous gets a small positive effect, not 0. The Jansen estimator
        does give exactly 0 when the model ignores the redrawn inputs.

        Intervals follow the same rule as :meth:`shapley`.

        Returns:
            A :class:`~jaxgsa.shapley.ShapleyResult` whose ``Sh`` holds the
            PME and whose ``backend`` is ``"exact"``. ``explained_variance``
            and ``order`` are ``None``, as for :meth:`shapley`. ``S1`` and
            ``ST`` are this result's own indices.

        Raises:
            ValueError: If the result does not hold subset indices, because the
                design used ``subsets="singletons"``.
        """
        return self._allocate("pme")

    # -- export ----------------------------------------------------------

    def to_dataset(self, time_coords: np.ndarray | list | None = None) -> xr.Dataset:
        """Convert the result to a labeled xarray Dataset.

        Adds the subset indices when the result has them: ``closed`` and
        ``subset_total`` (and their ``*_lower``/``*_upper``) on a ``subset``
        dimension, and ``subset_members``, a boolean ``(subset, param)``
        matrix. Each subset is labeled by its member names joined with ``+``.
        """
        ds = super().to_dataset(time_coords)
        if self.closed is None:
            return ds
        base, _ = self._base_dims(time_coords)
        lead = base[:-1]
        D = self.problem.n_params
        masks = np.arange(1, 2**D)
        names = list(self.problem.names)
        labels = ["+".join(names[j] for j in range(D) if (m >> j) & 1) for m in masks.tolist()]
        ds = ds.assign_coords(subset=labels)
        dims = (*lead, "subset")
        for name in ("closed", "subset_total"):
            ds[name] = (dims, np.asarray(getattr(self, name)))
            conf = getattr(self, f"{name}_conf")
            if conf is not None:
                ds[f"{name}_lower"] = (dims, np.asarray(conf)[0])
                ds[f"{name}_upper"] = (dims, np.asarray(conf)[1])
        if self.subsets is not None:
            ds["subset_members"] = (("subset", "param"), np.asarray(self.subsets))
        return ds

    def __repr__(self) -> str:
        """Return a one-line summary that also shows the subset indices."""
        text = super().__repr__()
        if self.closed is None or self.subset_total is None:
            return text
        extra = (
            f"closed={tuple(np.shape(self.closed))}, "
            f"subset_total={tuple(np.shape(self.subset_total))}"
        )
        return text[:-1] + f", {extra})"
