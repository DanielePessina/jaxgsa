"""Result type for VKOGA surrogate-based correlated sensitivity analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from jax import Array

from jaxgsa._core.copula import is_independent
from jaxgsa._core.invalid import InvalidReport
from jaxgsa._core.result import FieldSpec, ResultSchema, SchemaResult
from jaxgsa._core.surrogate import SurrogateResult, _PredictPlan
from jaxgsa.problem import Problem
from jaxgsa.vkoga._engine import _VKOGAState

if TYPE_CHECKING:
    from jaxgsa.shapley._result import ShapleyResult


@dataclass(repr=False)
class VKOGAResult(SchemaResult, SurrogateResult):
    """Correlated variance-based sensitivity indices from a VKOGA surrogate.

    Produced by :func:`jaxgsa.vkoga.analyze`. Index arrays have shape ``(D,)``
    for a scalar output, ``(K, D)`` for a multi-output model, and ``(T, K, D)``
    for a time-resolved analysis.

    Under independent inputs the indices collapse to the familiar ones.
    ``S_TC`` is the first-order Sobol' index ``S1``, ``S_TU`` is the total
    index ``S_T``, ``S_U`` equals ``S_TC``, and ``S_C`` is zero. Under
    dependence they separate (Li et al. 2010).

    Attributes:
        S_TC: Total correlated index ``V(E(Y|X_i)) / V(Y)``, shape
            ``(..., D)``. It counts what ``X_i`` explains through itself and
            through its correlation with the other parameters. The word
            "total" names the pathways it counts, not the interaction order:
            the formula is a first-order conditional variance, so ``S_TC`` is
            not a total-order Sobol' index. Rank parameters by ``S_TC`` to
            decide which ones to measure, because it answers how much variance
            learning ``X_i`` would remove.
        S_TU: Total uncorrelated index ``E(V(Y|X_-i)) / V(Y)``, shape
            ``(..., D)``. It counts what only ``X_i`` can explain, with every
            correlated pathway removed. Rank parameters by ``S_TU`` to decide
            which ones to fix: a parameter with ``S_TU`` near zero can be
            frozen.
        S_U: Independent contribution of ``X_i`` alone,
            ``E(V(f_i|X_-i)) / V(Y)``, shape ``(..., D)``. Here ``f_i`` is the
            fitted additive component of the output, so this is the
            decorrelated first-order index of Mara & Tarantola (2012). It is
            clipped to at most ``S_TU``, and a wide clip raises a
            ``JaxgsaWarning``.
        S_C: Correlation-borne contribution ``S_TC - S_U``, shape ``(..., D)``.
            It can be negative when a correlation opposes a direct effect.
        S_IU: Independent interaction contribution ``S_TU - S_U``, shape
            ``(..., D)``. Zero for an additive model, and non-negative by
            construction of the ``S_U`` clip.
        problem: Problem definition used for the analysis.
        correlation: Gaussian-copula correlation matrix the indices were
            computed under, shape ``(D, D)``. The identity when inputs were
            treated as independent.
        variance: Output variance under the correlated input measure, one value
            per output slice (shape ``()``, ``(K,)``, or ``(T, K)``).
        n_centers: Number of kernel centres the greedy selected.
        gamma: Fitted RBF shape parameter.
        ridge: Fitted regularisation parameter.
        invalid: What the non-finite check found in ``(X, Y)`` and what the
            ``on_invalid`` policy did about it. See
            :class:`jaxgsa._core.invalid.InvalidReport`. ``n_invalid == 0``
            means the check ran and found nothing.
        rmse: Training-fit RMSE, one value per output slice (shape ``()``,
            ``(K,)``, or ``(T, K)``). It measures how well the surrogate
            reproduces the rows it was fitted on, so it is optimistic. Read
            ``cv_rmse`` to judge the fit.
        cv_rmse: Pooled out-of-sample RMSE of the chosen hyperparameters from
            the k-fold cross-validation, one scalar for the whole fit. This is
            the honest accuracy estimate. Every index is measured against the
            surrogate, so a large ``cv_rmse`` relative to ``std(Y)`` makes the
            indices unreliable, and ``analyze`` warns in that case. It is
            ``None`` when the caller fixed both ``gamma`` and ``ridge``,
            because no cross-validation ran.
    """

    S_TC: Array
    S_TU: Array
    S_U: Array
    S_C: Array
    S_IU: Array
    problem: Problem
    correlation: np.ndarray
    variance: Array
    n_centers: int
    gamma: float
    ridge: float
    invalid: InvalidReport
    rmse: Array | None = None
    cv_rmse: float | None = None
    _fit: _VKOGAState | None = field(default=None, repr=False)
    _y_mean: Array | None = field(default=None, repr=False)
    _output_shape: tuple[int, ...] = field(default=(), repr=False)

    _schema = ResultSchema(
        primary="S_TC",
        fields=(
            FieldSpec("S_TC"),
            FieldSpec("S_TU"),
            FieldSpec("S_U"),
            FieldSpec("S_C"),
            FieldSpec("S_IU"),
            FieldSpec("variance", "slice"),
            FieldSpec("rmse", "slice"),
            # The copula matrix describes the input model, not any output
            # slice, so it carries no output or time axis.
            FieldSpec("correlation", "pair_only"),
        ),
        meta=("n_centers", "gamma", "ridge", "is_correlated"),
    )

    def _dataset_attrs(self) -> dict[str, Any]:
        """Record the fit's hyperparameters and its dependence assumption."""
        attrs: dict[str, Any] = {
            "method": "vkoga",
            "n_centers": self.n_centers,
            "gamma": self.gamma,
            "ridge": self.ridge,
            "correlated": bool(self.is_correlated),
        }
        # Omitted rather than stored as None: netCDF has no null attribute.
        if self.cv_rmse is not None:
            attrs["cv_rmse"] = float(self.cv_rmse)
        return attrs

    def _predict_plan(self, X: Array) -> _PredictPlan:
        """Plan a batched evaluation of the fitted surrogate at ``X``.

        Maps ``X`` through the same marginal-CDF transform used at fit time.
        The kernel is isotropic, so it only behaves if every column is on a
        common scale. See :meth:`predict` for the full contract.

        Args:
            X: Points to predict at, shape ``(N, D)``.

        Returns:
            A batched evaluation plan for the fitted surrogate.
        """
        from jaxgsa.vkoga._analyze import _vkoga_predict_plan

        return _vkoga_predict_plan(self, X)

    def shapley(self) -> "ShapleyResult":
        """Not available for a kernel surrogate.

        Shapley effects need a decomposition of the output variance into terms
        with known parameter membership. A VKOGA expansion is a sum over kernel
        centres, not over parameter subsets. Every centre involves every
        parameter, so there is no membership matrix to allocate from.

        This method no longer overrides an abstract one: ``SurrogateResult``
        stopped declaring ``shapley``, because a contract that one of its three
        subclasses can only meet by refusing is not a contract. It stays as a
        signpost, so the call names the two backends that do support it.

        Raises:
            NotImplementedError: Always. Use ``jaxgsa.hdmr`` or ``jaxgsa.pce``
                for Shapley effects; ``hdmr`` additionally supports
                ``shapley(include_correlative=True)`` for dependent inputs.
        """
        raise NotImplementedError(
            "VKOGAResult has no term-wise variance decomposition, so Shapley effects are "
            "undefined for it; use jaxgsa.hdmr or jaxgsa.pce instead"
        )

    @property
    def is_correlated(self) -> bool:
        """Whether the indices were computed under a non-trivial dependency.

        Delegates to the same identity test as
        ``Problem.has_correlated_inputs``, so the two classifications always
        agree.
        """
        return not is_independent(self.correlation)
