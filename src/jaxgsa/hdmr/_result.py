"""Defines the HDMRResult dataclass for RS-HDMR sensitivity analysis results."""

import itertools
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, TypedDict

import jax.numpy as jnp
import numpy as np
import xarray as xr
from jax import Array

from jaxgsa._core.surrogate import SurrogateResult, _PredictPlan
from jaxgsa._core.validation import _dims_and_coords
from jaxgsa.problem import Problem

if TYPE_CHECKING:
    from jaxgsa.shapley import ShapleyResult


class _HDMRFit(TypedDict):
    """Typed emulator payload returned inside ``HDMRResult``.

    The coefficient arrays are stored on the fitted analysis scale. When
    ``prenormalize`` is ``True``, ``y_mean`` and ``y_std`` are used by
    :meth:`HDMRResult.predict` to map predictions back to the original scale.
    """

    C1: Array
    C2: Array | None
    C3: Array | None
    f0: Array
    prenormalize: bool
    y_mean: Array
    y_std: Array
    m: int
    maxorder: int


@dataclass
class HDMRResult(SurrogateResult):
    """RS-HDMR (Random Sampling High-Dimensional Model Representation) results.

    Stores ANCOVA-decomposed sensitivity indices. Each *term* is one component
    function of the HDMR expansion -- a single parameter, a pair, or a triple
    (up to the ``maxorder`` used) -- named in ``terms``. Per-term indices
    (Sa, Sb, S) have a trailing ``n_terms`` axis; the per-parameter ST has a
    trailing ``D`` axis.

    Shapes follow ``(T, K, n_terms)`` for time-resolved multi-output analyses.
    Singleton T and/or K dimensions are squeezed when the original Y had fewer
    than 3 dimensions.

    The :attr:`S1`, :attr:`S2`, and :attr:`S3` properties reshape the structural
    (``Sa``) blocks into the conventional Sobol-index layouts -- a ``(D,)``
    vector, a ``(D, D)`` matrix, and a ``(D, D, D)`` tensor respectively.

    Attributes:
        Sa: Structural (uncorrelated) variance fraction per term, shape
            ``(n_terms,)`` / ``(K, n_terms)`` / ``(T, K, n_terms)``. The part
            of a term's contribution independent of other inputs.
        Sb: Correlative contribution per term, same shape as ``Sa``. Near
            zero when inputs are independent; non-zero values flag variance
            shared through input correlation (and can be negative).
        S: Total contribution per term, ``S = Sa + Sb``, same shape.
        ST: Total-order index per parameter -- its first-order term plus
            every interaction term containing it -- shape ``(D,)`` /
            ``(K, D)`` / ``(T, K, D)``.
        problem: Problem definition used for the analysis.
        terms: Human-readable term labels, e.g. ``("x1", "x2", "x1/x2")``;
            interaction terms join parameter names with ``/``.
        _fit: Private fitted surrogate state used by :meth:`predict`.
        select: F-test significance count per term, summed over the T*K
            output slices (max value T*K), or None. Low counts mark terms
            the F-test deems insignificant.
        rmse: Emulator fit RMSE per output slice in the units of ``Y``,
            shape ``()`` / ``(K,)`` / ``(T, K)``, or None.
    """

    Sa: Array
    Sb: Array
    S: Array
    ST: Array
    problem: Problem
    terms: tuple[str, ...]
    _fit: _HDMRFit | None = field(default=None, repr=False)
    select: Array | None = None
    rmse: Array | None = None
    _c2: tuple[tuple[int, int], ...] = field(default=(), repr=False)
    _c3: tuple[tuple[int, int, int], ...] = field(default=(), repr=False)

    def _predict_plan(self, X: Array) -> _PredictPlan:
        """Plan a batched evaluation of the fitted HDMR surrogate at ``X``.

        Re-applies the CDF transform used during fitting and packages a
        kernel that rebuilds the B-spline tensor-product bases (a large
        per-row constant, up to ``m1^3`` floats per interaction term at
        ``maxorder=3``) and contracts them with the fitted component
        coefficients; when the fit used ``prenormalize=True``, the kernel
        inverse-transforms predictions back to the original output scale.
        See :meth:`predict` for the full contract.

        Raises:
            ValueError: If this result carries no fitted surrogate state.
        """
        from jaxgsa.hdmr._analyze import _hdmr_predict_plan

        return _hdmr_predict_plan(self, X)

    def shapley(self, *, include_correlative: bool = False) -> "ShapleyResult":
        """Compute Shapley effects from this fitted HDMR decomposition.

        Allocates each fitted ANCOVA term's variance share equally among the
        parameters participating in that term, yielding per-parameter Shapley
        effects that sum to one.

        Args:
            include_correlative: When ``True``, allocate the total ANCOVA
                contribution ``Sa + Sb`` (structural plus correlative
                fold-in), which keeps the allocation meaningful under
                correlated inputs. Defaults to ``False``, allocating the
                structural part ``Sa`` only.

        Returns:
            ShapleyResult with per-parameter effects ``Sh`` (plus ``S1`` and
            ``ST``) and explained-variance diagnostics.

        Raises:
            ValueError: If this result carries no fitted surrogate state.
        """
        from jaxgsa.shapley._analyze import _shapley_result_from_variances
        from jaxgsa.shapley._engine import build_membership

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
        )

    @property
    def S1(self) -> Array:
        """First-order Sobol indices (structural contribution of first-order terms).

        Equivalent to ``Sa[:D]`` — the uncorrelated variance fraction of each
        single-parameter component function, which matches the definition of
        first-order Sobol indices.

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
        fraction of the ``(x_i, x_j)`` interaction component; the matrix is
        symmetric. Cells for the diagonal and for any parameter pair absent
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
        component; the tensor is symmetric under permutation of its axes.
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

    def __repr__(self) -> str:
        """Return a concise summary showing index shapes."""
        shapes = {
            "Sa": self.Sa.shape,
            "Sb": self.Sb.shape,
            "S": self.S.shape,
            "ST": self.ST.shape,
        }
        return f"HDMRResult({shapes})"

    def to_dataset(
        self,
        time_coords: np.ndarray | list | None = None,
    ) -> xr.Dataset:
        """Convert results to a labeled xarray Dataset.

        Args:
            time_coords: Coordinate values for the time dimension when
                arrays are 3-D. Defaults to integer indices.

        Returns:
            An ``xr.Dataset`` with variables ``Sa``, ``Sb``, ``S``, ``ST``,
            ``S2`` when second-order terms exist (and ``S3`` when third-order
            terms exist), and optionally ``select`` and ``rmse``.
        """
        # ST is indexed by parameter, exactly the shared param/output/time
        # schema; Sa/Sb/S replace the trailing "param" with "term" (interaction
        # components), and select/rmse drop it entirely.
        dims_param, coords = _dims_and_coords(
            self.Sa.ndim, self.Sa.shape, self.problem, time_coords
        )
        coords = {**coords, "term": list(self.terms)}
        dims_term = (*dims_param[:-1], "term")

        data_vars: dict = {
            "Sa": (dims_term, np.asarray(self.Sa)),
            "Sb": (dims_term, np.asarray(self.Sb)),
            "S": (dims_term, np.asarray(self.S)),
            "ST": (dims_param, np.asarray(self.ST)),
        }

        # S2/S3 are symmetric interaction tensors; their trailing axes each span
        # the parameters, so they get their own dim names (param_i/j/k) to avoid
        # clashing with the 1-D "param" coord, mirroring PCEResult.to_dataset.
        # Each is emitted only when its expansion order has terms.
        param_names = list(self.problem.names)
        lead = dims_param[:-1]
        if len(self._c2) > 0:
            data_vars["S2"] = ((*lead, "param_i", "param_j"), np.asarray(self.S2))
            coords["param_i"] = param_names
            coords["param_j"] = param_names
        if len(self._c3) > 0:
            data_vars["S3"] = (
                (*lead, "param_i", "param_j", "param_k"),
                np.asarray(self.S3),
            )
            coords["param_k"] = param_names

        if self.select is not None:
            data_vars["select"] = (("term",), np.asarray(self.select))

        if self.rmse is not None:
            # RMSE has no param/term axis, so it uses the leading dims only:
            # () / (output,) / (time, output).
            data_vars["rmse"] = (dims_param[:-1], np.asarray(self.rmse))

        return xr.Dataset(data_vars, coords=coords)
