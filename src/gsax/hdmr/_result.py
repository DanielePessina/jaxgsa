"""Defines the HDMRResult dataclass for RS-HDMR sensitivity analysis results."""

import itertools
from dataclasses import dataclass, field
from typing import TypedDict

import jax.numpy as jnp
import numpy as np
import xarray as xr
from jax import Array

from gsax._normalization import _dims_and_coords
from gsax.problem import Problem


class HDMREmulator(TypedDict):
    """Typed emulator payload returned inside ``HDMRResult``.

    The coefficient arrays are stored on the fitted analysis scale. When
    ``prenormalize`` is ``True``, ``y_mean`` and ``y_std`` are used by
    ``emulate_hdmr()`` to map predictions back to the original output scale.
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
    c2: list[tuple[int, int]]
    c3: list[tuple[int, int, int]]


@dataclass
class HDMRResult:
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
        c2: Second-order parameter-index pairs, aligned with the second-order
            block of the term axis. Used by the ``S2`` property.
        c3: Third-order parameter-index triples, aligned with the third-order
            block of the term axis. Used by the ``S3`` property.
        n1: Number of first-order terms -- the slice boundary that separates
            first-order from higher-order blocks along the term axis.
        emulator: Fitted surrogate state for ``hdmr.emulate``, or None.
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
    c2: tuple[tuple[int, int], ...] = ()
    c3: tuple[tuple[int, int, int], ...] = ()
    n1: int = 0
    emulator: HDMREmulator | None = None
    select: Array | None = None
    rmse: Array | None = None
    # True when layout inference inserted the singleton output axis (a 2-D
    # (N, T) Y under a single named output). emulate_hdmr squeezes it back so
    # predictions mirror the training Y's rank.
    _inserted_output_axis: bool = field(default=False, repr=False)

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

    @property
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
        n2 = len(self.c2)
        out = jnp.full(self.Sa.shape[:-1] + (D, D), jnp.nan, dtype=self.Sa.dtype)
        if n2 == 0:
            return out
        vals = self.Sa[..., self.n1 : self.n1 + n2]  # (..., n2)
        for k, (i, j) in enumerate(self.c2):
            out = out.at[..., i, j].set(vals[..., k])
            out = out.at[..., j, i].set(vals[..., k])
        return out

    @property
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
        n2 = len(self.c2)
        n3 = len(self.c3)
        out = jnp.full(self.Sa.shape[:-1] + (D, D, D), jnp.nan, dtype=self.Sa.dtype)
        if n3 == 0:
            return out
        vals = self.Sa[..., self.n1 + n2 :]  # (..., n3)
        for k, combo in enumerate(self.c3):
            # Fill all permutations so the tensor is fully symmetric.
            for perm in itertools.permutations(combo):
                out = out.at[(..., *perm)].set(vals[..., k])
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
            ``S2`` (and ``S3`` when third-order terms are present), and
            optionally ``select`` and ``rmse``.
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
        param_names = list(self.problem.names)
        lead = dims_param[:-1]
        data_vars["S2"] = ((*lead, "param_i", "param_j"), np.asarray(self.S2))
        coords["param_i"] = param_names
        coords["param_j"] = param_names
        if len(self.c3) > 0:
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
