"""Defines the HDMRResult dataclass for RS-HDMR sensitivity analysis results."""

import itertools
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any, TypedDict

import jax.numpy as jnp
from jax import Array

from jaxgsa._core.invalid import InvalidReport
from jaxgsa._core.result import FieldSpec, ResultSchema, SchemaResult
from jaxgsa._core.surrogate import SurrogateResult, _PredictPlan
from jaxgsa.problem import Problem

if TYPE_CHECKING:
    from jaxgsa.shapley import ShapleyResult


class _HDMRFit(TypedDict):
    """Fitted HDMR surrogate state carried inside ``HDMRResult``.

    The coefficient arrays are stored on the output scale the caller passed
    in, so :meth:`HDMRResult.predict` needs no inverse transform.
    """

    C1: Array
    C2: Array | None
    C3: Array | None
    f0: Array
    m: int
    maxorder: int


@dataclass(repr=False)
class HDMRResult(SchemaResult, SurrogateResult):
    """RS-HDMR (Random Sampling High-Dimensional Model Representation) results.

    Stores ANCOVA-decomposed sensitivity indices. A term is one component
    function of the HDMR expansion: a single parameter, a pair, or a triple,
    up to the ``maxorder`` used. ``terms`` holds their names. The per-term
    indices (Sa, Sb, S) have a trailing ``n_terms`` axis. The per-parameter
    ST has a trailing ``D`` axis.

    Shapes follow ``(T, K, n_terms)`` for time-resolved multi-output analyses.
    Singleton T and/or K dimensions are squeezed when the original Y had fewer
    than 3 dimensions.

    The :attr:`S1`, :attr:`S2`, and :attr:`S3` properties reshape the structural
    (``Sa``) blocks into the conventional Sobol-index layouts: a ``(D,)``
    vector, a ``(D, D)`` matrix, and a ``(D, D, D)`` tensor respectively.

    Warning:
        Under correlated inputs, :attr:`ST` and :attr:`S1` do not carry their
        usual Sobol meaning. :attr:`ST` is the SCSA total of Li et al.
        (2010), Section 2.2.3, and :attr:`S1` is a structural share only.
        Neither one measures expected conditional-variance reduction, and
        neither one tells you that a parameter can be fixed. Read the notes on
        each attribute before you rank parameters from a correlated fit.

    Attributes:
        Sa: Structural (uncorrelated) variance fraction per term, shape
            ``(n_terms,)`` / ``(K, n_terms)`` / ``(T, K, n_terms)``. This is
            the part of a term's contribution independent of other inputs.
        Sb: Correlative contribution per term, same shape as ``Sa``. It is
            near zero when inputs are independent. A non-zero value flags
            variance shared through input correlation, and it can be
            negative.
        S: Total contribution per term, ``S = Sa + Sb``, same shape as ``Sa``.
        ST: SCSA total per parameter, shape ``(D,)`` / ``(K, D)`` /
            ``(T, K, D)``. It sums ``S = Sa + Sb`` over every term that
            contains the parameter: ``ST_i = sum over u containing i of
            (Sa_u + Sb_u)``. Li, Rabitz et al. (2010) define it in Section
            2.2.3: "the total sensitivity indices ... can be calculated by
            adding together all the sensitivity indices containing X_i".
            They build it from the per-term indices of their Eqs. (19)-(22).
            Sarazin, Viaud & Cournede (2017) restate it as their Eq. (8).
            SALib and Vrugt's ``HDMR_end.m`` use the same convention.

            Li et al. attach a precondition to it: the totals are reliable
            only when the per-term ``S`` values sum to about 1 (their
            Eq. (24)). The shortfall is the surrogate's unexplained
            variance, so check ``S.sum()`` before ranking parameters.

            With independent inputs the correlative shares ``Sb`` vanish and
            ``ST`` reduces to the ordinary Sobol total-order index.

            With correlated inputs it does not reduce to that index. It can
            be negative. It is not bounded in ``[0, 1]``. It does not measure
            the expected reduction of output variance from fixing the
            parameter. Do not use it to decide that a parameter can be fixed:
            the bias runs toward "cannot be fixed", and a parameter the model
            ignores can outrank one with a negative value. Li et al. reuse
            the symbol ``S_Ti`` for two different quantities: this
            term-membership sum, and the classical conditional-variance total
            of their Eq. (4). Only the first one is computed here. Sarazin et
            al. state explicitly that the ``[0, 1]`` bound no longer holds.
            ``ST`` is also not comparable with the ``ST`` of
            ``jaxgsa.kucherenko`` or the ``S_TU`` of ``jaxgsa.vkoga``. Use one
            of those for a genuine conditional-variance total under
            dependence.
        problem: Problem definition used for the analysis.
        terms: Human-readable term labels, e.g. ``("x1", "x2", "x1/x2")``.
            Interaction terms join parameter names with ``/``.
        _fit: Private fitted surrogate state used by :meth:`predict`.
        select: F-test significance count per term, summed over the T*K
            output slices (maximum value T*K), or None. A low count marks a
            term the F-test deems insignificant.
        rmse: Emulator fit RMSE per output slice in the units of ``Y``,
            shape ``()`` / ``(K,)`` / ``(T, K)``, or None.
        streamed: True when the fit ran the row-streamed path, False when it
            ran the in-memory path. Both paths fit the same components, pick
            the same F-test term set, and report the same indices; they differ
            only in float32 summation order and in peak memory. The streamed
            path engages when ``batch_size`` is an explicit int, or when the
            in-memory fit would exceed the memory budget (see
            :func:`jaxgsa.config.set_memory_budget`). Read it when a fit takes
            much longer than expected: True means the budget engaged.
        invalid: What the non-finite check found in ``(X, Y)`` and what the
            ``on_invalid`` policy did about it. See
            :class:`jaxgsa._core.invalid.InvalidReport`. ``n_invalid == 0``
            means the check ran and found nothing.
        _c2: Private parameter index pairs of the second-order terms, in the
            order they occupy in the ``n_terms`` axis. Empty at
            ``maxorder=1``.
        _c3: Private parameter index triples of the third-order terms, in the
            order they occupy in the ``n_terms`` axis. Empty below
            ``maxorder=3``.
    """

    Sa: Array
    Sb: Array
    S: Array
    ST: Array
    problem: Problem
    terms: tuple[str, ...]
    invalid: InvalidReport
    _fit: _HDMRFit | None = field(default=None, repr=False)
    select: Array | None = None
    rmse: Array | None = None
    streamed: bool = False
    _c2: tuple[tuple[int, int], ...] = field(default=(), repr=False)
    _c3: tuple[tuple[int, int, int], ...] = field(default=(), repr=False)

    # Sa/Sb/S are indexed by expansion term, not by parameter, so they cannot
    # share the "param" axis that ST uses. "select" carries the term axis
    # alone: it is one flag per term, with no output or time axis at all.
    _schema = ResultSchema(
        primary="Sa",
        fields=(
            FieldSpec("Sa", "term"),
            FieldSpec("Sb", "term"),
            FieldSpec("S", "term"),
            FieldSpec("ST", "param"),
            FieldSpec("S2", "pair"),
            FieldSpec("S3", "triple"),
            FieldSpec("select", "term_only"),
            FieldSpec("rmse", "slice"),
        ),
        meta=("streamed",),
    )

    def _extra_coords(self) -> dict[str, Any]:
        """Name the expansion-term axis, which the parameter names cannot."""
        return {"term": list(self.terms)}

    def _omit_fields(self) -> frozenset[str]:
        """Skip the interaction tensors when the fit has no terms of that order.

        ``S2`` and ``S3`` always materialize, filled with NaN where a term is
        absent. Exporting an all-NaN tensor from a ``maxorder=1`` fit would
        claim an interaction structure the fit never modelled.
        """
        omit = set()
        if not self._c2:
            omit.add("S2")
        if not self._c3:
            omit.add("S3")
        return frozenset(omit)

    def _predict_plan(self, X: Array) -> _PredictPlan:
        """Plan a batched evaluation of the fitted HDMR surrogate at ``X``.

        Re-applies the CDF transform used during fitting. It then packages a
        kernel that rebuilds the B-spline tensor-product bases and contracts
        them with the fitted component coefficients. Those bases are a large
        per-row constant: up to ``m1^3`` floats per interaction term at
        ``maxorder=3``. See :meth:`predict` for the full contract.

        Raises:
            ValueError: If this result carries no fitted surrogate state.
        """
        from jaxgsa.hdmr._analyze import _hdmr_predict_plan

        return _hdmr_predict_plan(self, X)

    def shapley(self, *, include_correlative: bool = False) -> "ShapleyResult":
        """Compute Shapley effects from this fitted HDMR decomposition.

        Allocates each fitted ANCOVA term's variance share equally among the
        parameters that take part in that term. The per-parameter Shapley
        effects sum to one.

        Args:
            include_correlative: When ``True``, allocate the total ANCOVA
                contribution ``Sa + Sb`` (structural plus correlative
                fold-in). That keeps the allocation meaningful under
                correlated inputs. Defaults to ``False``, which allocates the
                structural part ``Sa`` only.

        Returns:
            ShapleyResult with per-parameter effects ``Sh`` (plus ``S1`` and
            ``ST``) and explained-variance diagnostics.

        Raises:
            ValueError: If this result carries no fitted surrogate state.
        """
        from jaxgsa.shapley._engine import _shapley_result_from_variances, build_membership

        fit = self._fit
        if fit is None:
            raise ValueError("HDMRResult does not contain fitted surrogate state")
        partial = self.Sa + self.Sb if include_correlative else self.Sa
        subsets: list[tuple[int, ...]] = [(i,) for i in range(self.problem.num_vars)]
        subsets.extend(self._c2)
        subsets.extend(self._c3)
        membership = build_membership(subsets, self.problem.num_vars)
        # For HDMR the per-term sum doubles as the explained-variance
        # diagnostic (indices are already output-variance fractions);
        # compute it once and reuse it as the normalizer.
        explained = partial.sum(axis=-1)
        return _shapley_result_from_variances(
            partial,
            membership,
            explained,
            total=explained,
            problem=self.problem,
            backend="hdmr",
            order=fit["maxorder"],
            include_correlative=include_correlative,
            invalid=self.invalid,
        )

    @property
    def S1(self) -> Array:
        """Structural first-order share (Sobol' S1 only under independent inputs).

        Equivalent to ``Sa[:D]``: the uncorrelated variance fraction of each
        single-parameter component function. That matches the definition of
        first-order Sobol indices.

        Note:
            This is the structural share only. With independent inputs it
            equals the first-order Sobol index. With correlated inputs it does
            not: the correlative share ``Sb`` is left out, so the value can sit
            far below the Sobol ``S1`` of the same parameter. See :attr:`ST`
            for the matching caveat on the total.

        Returns:
            Array of shape ``(D,)`` / ``(K, D)`` / ``(T, K, D)``.
        """
        D = self.problem.num_vars
        return self.Sa[..., :D]

    @cached_property
    def S2(self) -> Array:
        """Second-order structural Sobol indices as a symmetric matrix.

        Mirrors :attr:`S1`, scattering the second-order block of ``Sa`` into a
        dense ``(D, D)`` matrix. Entry ``[i, j]`` is the structural variance
        fraction of the ``(x_i, x_j)`` interaction component, and the matrix
        is symmetric. Cells for the diagonal and for any parameter pair absent
        from the expansion (e.g. ``maxorder < 2``) are ``NaN``.

        Returns:
            Array of shape ``(D, D)`` / ``(K, D, D)`` / ``(T, K, D, D)``.
        """
        D = self.problem.num_vars
        n2 = len(self._c2)
        out = jnp.full(self.Sa.shape[:-1] + (D, D), jnp.nan, dtype=self.Sa.dtype)
        if n2 == 0:
            return out
        n1 = self.problem.num_vars
        vals = self.Sa[..., n1 : n1 + n2]  # (..., n2)
        pairs = jnp.asarray(self._c2)  # (n2, 2)
        i, j = pairs[:, 0], pairs[:, 1]
        # A single vectorized scatter fills both symmetric halves at once
        # (one full-tensor materialization instead of two).
        rows = jnp.concatenate([i, j])
        cols = jnp.concatenate([j, i])
        out = out.at[..., rows, cols].set(jnp.concatenate([vals, vals], axis=-1))
        return out

    @cached_property
    def S3(self) -> Array:
        """Third-order structural Sobol indices as a symmetric tensor.

        Mirrors :attr:`S1`/:attr:`S2`, scattering the third-order block of
        ``Sa`` into a dense ``(D, D, D)`` tensor. Entry ``[i, j, k]`` is the
        structural variance fraction of the ``(x_i, x_j, x_k)`` interaction
        component, and the tensor is symmetric under permutation of its axes.
        Cells where any two axes share an index, and any triple absent from
        the expansion (e.g. ``maxorder < 3``), are ``NaN``.

        Returns:
            Array of shape ``(D, D, D)`` / ``(K, D, D, D)`` /
            ``(T, K, D, D, D)``.
        """
        D = self.problem.num_vars
        n2 = len(self._c2)
        n3 = len(self._c3)
        out = jnp.full(self.Sa.shape[:-1] + (D, D, D), jnp.nan, dtype=self.Sa.dtype)
        if n3 == 0:
            return out
        vals = self.Sa[..., self.problem.num_vars + n2 :]  # (..., n3)
        combos = jnp.asarray(self._c3)  # (n3, 3)
        # Concatenate the six axis permutations of each index triple into one
        # (6*n3,) index set so a single scatter makes the tensor symmetric
        # (one full-tensor materialization instead of six sequential ones).
        perms = list(itertools.permutations(range(3)))
        ia = jnp.concatenate([combos[:, a] for a, _, _ in perms])
        ib = jnp.concatenate([combos[:, b] for _, b, _ in perms])
        ic = jnp.concatenate([combos[:, c] for _, _, c in perms])
        vals6 = jnp.concatenate([vals] * len(perms), axis=-1)  # (..., 6*n3)
        out = out.at[..., ia, ib, ic].set(vals6)
        return out
