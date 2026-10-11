"""Result type for VKOGA surrogate-based correlated sensitivity analysis."""

from __future__ import annotations

from dataclasses import dataclass, field

from jax import Array

from jaxgsa._core.invalid import InvalidReport
from jaxgsa._core.result import CIInfo, FieldSpec, ResultSchema, SchemaResult
from jaxgsa._core.surrogate import SurrogateResult, _PredictPlan
from jaxgsa.problem import Dependence, Problem
from jaxgsa.vkoga._engine import _VKOGAState


@dataclass(frozen=True, repr=False)
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
            fitted additive component of the output. It reads like the
            decorrelated first-order index of Mara & Tarantola (2012), but it
            is not their estimator: theirs conditions on the Rosenblatt
            residual, and the two agree only when ``f_i`` is linear. ``S_U``
            is clipped to at most ``S_TU``, and a wide clip raises a
            ``JaxgsaWarning``.
        S_C: Correlation-borne contribution ``S_TC - S_U``, shape ``(..., D)``.
            It can be negative when a correlation opposes a direct effect.
        S_IU: Independent interaction contribution ``S_TU - S_U``, shape
            ``(..., D)``. Zero for an additive model, and non-negative by
            construction of the ``S_U`` clip.
        problem: Problem definition used for the analysis.
        dependence: The input dependence the indices were computed under:
            ``"independent"``, ``"gaussian"`` or ``"custom"``. It always
            equals ``problem.dependence``: the dependence comes from the
            problem alone, so read the copula matrix from
            ``problem.correlation``.
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
        cv_rmse: Out-of-sample RMSE of the chosen hyperparameters from the
            k-fold cross-validation, one value per output slice (shape
            ``()``, ``(K,)``, or ``(T, K)``). The hyperparameters are chosen
            on the RMSE pooled over the slices, and each slice's value comes
            from the same fold residuals. This is the honest accuracy
            estimate. Every index is measured against the
            surrogate, so a large ``cv_rmse`` relative to ``std(Y)`` makes the
            indices unreliable, and ``analyze`` warns in that case. It is
            ``None`` when the caller fixed both ``gamma`` and ``ridge``,
            because no cross-validation ran.
        explained_variance: ``1 - rmse**2 / var(Y)`` per output slice, from
            the training rows. Like ``rmse`` it is in-sample, so it is
            optimistic; read ``cv_rmse`` to judge the fit. ``nan`` for a
            slice with zero variance.
        S_TC_conf: Bootstrap confidence interval for ``S_TC``, shape
            ``(2, ...)`` for ``[lower, upper]``. ``None`` when
            ``n_bootstrap=0``.
        S_TU_conf: The same for ``S_TU``.
        S_U_conf: The same for ``S_U``.
        S_C_conf: The same for ``S_C``.
        S_IU_conf: The same for ``S_IU``.
        ci: How the intervals were produced: the confidence level, the
            endpoint rule, the resample count, and the bootstrap draws when
            the analysis ran with ``keep_replicates=True``. ``None`` without a
            bootstrap. Each replicate resampled the training rows, refitted
            the surrogate and re-integrated, so the interval covers the
            training-sample and Monte-Carlo error together, at fixed
            ``gamma`` and ``ridge``. It says nothing about how far the
            surrogate is from the true model; read ``cv_rmse`` for that. See
            :class:`jaxgsa._core.result.CIInfo`.
    """

    S_TC: Array
    S_TU: Array
    S_U: Array
    S_C: Array
    S_IU: Array
    problem: Problem
    variance: Array
    n_centers: int
    gamma: float
    ridge: float
    invalid: InvalidReport
    dependence: Dependence = "independent"
    rmse: Array | None = None
    cv_rmse: Array | None = None
    explained_variance: Array | None = None
    _fit: _VKOGAState | None = field(default=None, repr=False)
    _y_mean: Array | None = field(default=None, repr=False)
    _output_shape: tuple[int, ...] = field(default=(), repr=False)
    S_TC_conf: Array | None = None
    S_TU_conf: Array | None = None
    S_U_conf: Array | None = None
    S_C_conf: Array | None = None
    S_IU_conf: Array | None = None
    ci: CIInfo | None = None

    _schema = ResultSchema(
        primary="S_TC",
        fields=(
            FieldSpec("S_TC", interval=True),
            FieldSpec("S_TU", interval=True),
            FieldSpec("S_U", interval=True),
            FieldSpec("S_C", interval=True),
            FieldSpec("S_IU", interval=True),
            FieldSpec("variance", "slice"),
            FieldSpec("rmse", "slice"),
            # A fit that ran no cross-validation carries ``None``, and the
            # derived export drops the field rather than writing a null.
            FieldSpec("cv_rmse", "slice"),
            FieldSpec("explained_variance", "slice"),
        ),
        meta=("n_centers", "gamma", "ridge", "dependence"),
    )

    @property
    def S1(self) -> Array:
        """First-order Sobol' index, an alias of ``S_TC``.

        Under independent inputs ``S_TC = V(E(Y|X_i)) / V(Y)`` is exactly the
        first-order index, so this reads like ``kucherenko`` ``S1``. Under
        dependence it also counts the correlated pathways.
        """
        return self.S_TC

    @property
    def ST(self) -> Array:
        """Total Sobol' index, an alias of ``S_TU``.

        Under independent inputs ``S_TU = E(V(Y|X_-i)) / V(Y)`` is exactly the
        total index, so this reads like ``kucherenko`` ``ST``. Under
        dependence it counts only the uncorrelated pathways.
        """
        return self.S_TU

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
