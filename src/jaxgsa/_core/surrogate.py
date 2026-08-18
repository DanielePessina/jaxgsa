"""Shared template for surrogate-backed results (PCE, HDMR).

Every surrogate result predicts at new inputs in batches.
:class:`SurrogateResult` implements that plumbing once, as a template method:
validate ``X``, size the row batches against a transient-memory budget, then
run a subclass-supplied kernel. The body is short, but the contract it
documents is long, and writing it here states it once instead of three times.

Shapley effects are deliberately *not* declared here. Two of the three
subclasses read them off a term-wise variance decomposition, and the third,
``VKOGAResult``, has no such decomposition and could only satisfy an abstract
``shapley`` by raising. ``HDMRResult`` widens the signature with
``include_correlative`` besides, so the declaration fixed no signature either.
Typing its return also forced this module to import ``jaxgsa.shapley._result``
under ``TYPE_CHECKING``, which inverted the dependency: ``_core`` named a
method package's private module. ``PCEResult.shapley`` and
``HDMRResult.shapley`` are now plain methods on their own classes.

Nothing here is public API (promote-later policy). ``SurrogateResult`` is not
exported from ``jaxgsa``, and user code must type against ``PCEResult`` or
``HDMRResult`` directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from jaxgsa._core.batching import apply_batched, resolve_batch_size
from jaxgsa._core.validation import _validate_x
from jaxgsa.problem import Problem


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

    A subclass provides one method. :meth:`_predict_plan` supplies the input
    transform, the per-row memory estimate, and the prediction kernel.
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
