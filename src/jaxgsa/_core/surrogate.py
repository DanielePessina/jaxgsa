"""Shared template for surrogate-backed results (PCE, HDMR).

Both surrogate results expose the same two capabilities. They predict at new
inputs in batches, and they read analytical Shapley effects off the fitted
variance decomposition. :class:`SurrogateResult` implements the prediction
plumbing once, as a template method: validate ``X``, size the row batches
against a transient-memory budget, then run a subclass-supplied kernel. It
also declares the ``shapley`` contract that each subclass fulfils with its own
decomposition.

Nothing here is public API (promote-later policy). ``SurrogateResult`` is not
exported from ``jaxgsa``, and user code must type against ``PCEResult`` or
``HDMRResult`` directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp
from jax import Array

from jaxgsa._core.batching import apply_batched, resolve_batch_size
from jaxgsa._core.validation import _validate_x
from jaxgsa.problem import Problem

if TYPE_CHECKING:
    from jaxgsa.shapley._result import ShapleyResult


class _PredictPlan(NamedTuple):
    """Everything :meth:`SurrogateResult.predict` needs from a subclass.

    The prepared inputs, the per-row cost estimate, and the batch kernel are
    produced together because they share hoisted state: the kernel closes
    over tables derived from the same input transform whose output dtype
    prices ``bytes_per_row``.

    Attributes:
        X: ``(N_new, D)`` inputs already mapped to the surrogate's fitting
            domain (polynomial reference domain, unit hypercube, ...).
        bytes_per_row: Estimated transient memory needed to predict a single
            row (basis tensors plus contraction intermediates), used to
            derive the automatic batch size.
        kernel: Row-independent function mapping an ``(n, D)`` batch of the
            prepared inputs to its ``(n, ...)`` predictions on the original
            output scale. Any output standardization applied during fitting
            must be inverted inside the kernel.
    """

    X: Array
    bytes_per_row: int
    kernel: Callable[[Array], Array]


class SurrogateResult(ABC):
    """Base class for results that carry a reusable fitted surrogate.

    Subclasses provide two methods. :meth:`_predict_plan` supplies the input
    transform, the per-row memory estimate, and the prediction kernel.
    :meth:`shapley` supplies the variance decomposition, which the shared
    Shapley pipeline in ``jaxgsa.shapley._analyze`` then finishes.
    :meth:`predict` is implemented once here.
    """

    problem: Problem

    def predict(self, X: Array, *, batch_size: int | None = None) -> Array:
        """Predict outputs at new input rows using the fitted surrogate.

        Rebuilds the surrogate's basis at ``X`` and applies the fitted
        coefficients, so no model evaluations are needed. Accuracy degrades
        outside the input region the surrogate was fitted on.

        Args:
            X: ``(N_new, D)`` new input points, in the same physical units
                as the ``X`` the surrogate was fitted on.
            batch_size: Rows of ``X`` predicted per batch. The basis tensors
                are linear in the batch size with a large per-row constant,
                so single-shot evaluation at large ``N_new`` can exhaust
                memory. ``None`` (default) derives a batch size from a fixed
                transient-memory budget (~512 MiB); an int fixes the rows
                per batch, and ``batch_size >= N_new`` forces a single-shot
                call. Each row's prediction is independent, so batching only
                perturbs predictions at the level of floating-point
                reassociation.

        Returns:
            Predicted outputs mirroring the training ``Y`` layout:
            ``(N_new,)``, ``(N_new, K)``, or ``(N_new, T, K)``, on the
            original output scale (any output standardization applied
            during fitting is inverted).

        Raises:
            ValueError: If ``X`` is not 2-D with one column per problem
                parameter, ``batch_size`` is not a positive integer, or this
                result carries no fitted surrogate state.
        """
        X = jnp.asarray(X)
        _validate_x(self.problem, X)
        plan = self._predict_plan(X)
        batch = resolve_batch_size(plan.bytes_per_row, plan.X.shape[0], batch_size)
        return apply_batched(plan.kernel, plan.X, batch)

    @abstractmethod
    def _predict_plan(self, X: Array) -> _PredictPlan:
        """Build the prediction plan for already-validated inputs ``X``.

        Args:
            X: ``(N_new, D)`` validated input points in physical units.

        Returns:
            The prepared inputs, per-row transient-memory estimate, and
            batch kernel consumed by :meth:`predict`.

        Raises:
            ValueError: If this result carries no fitted surrogate state.
        """

    @abstractmethod
    def shapley(self) -> "ShapleyResult":
        """Compute Shapley effects from this fitted surrogate's decomposition.

        Each subclass supplies its per-term partial variances and term
        membership; the shared pipeline normalizes them, allocates each
        term's variance equally among its participants (Owen, 2014), and
        warns when the fit diagnostic flags an unreliable surrogate.
        Overrides may widen the signature with backend-specific keyword-only
        options (e.g. ``HDMRResult.shapley(include_correlative=...)``).

        Returns:
            ShapleyResult with per-parameter effects ``Sh`` (summing to 1
            per output slice), the matching ``S1``/``ST`` bounds, and an
            explained-variance diagnostic.

        Raises:
            ValueError: If this result carries no fitted surrogate state or
                fit diagnostics.
        """
