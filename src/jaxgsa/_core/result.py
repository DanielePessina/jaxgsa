"""Where a result declares its fields, and what is derived from that.

Nothing in this module is public API. See :mod:`jaxgsa._core`.

A result declares its fields once, as a :class:`ResultSchema`, and
:class:`SchemaResult` derives ``to_dataset``, ``__repr__`` and the
``*_lower``/``*_upper`` split of a confidence array from that declaration.
The export schema of a result is therefore readable off the declaration
rather than off a hand-written ``to_dataset`` body.

The scalar provenance of a result is declared the same way, as
``ResultSchema.meta``, and it feeds both the summary and the export.
``tests/test_result_schema.py`` reads the declaration and checks every
declared name reaches ``to_dataset().attrs``.

Three results are genuinely irregular, so the derivation has hooks rather
than special cases:

* ``OTResult`` picks its dimensions from its ``mode`` field, not from
  ``arr.ndim``. It overrides :meth:`SchemaResult._base_dims`.
* ``HDMRResult`` indexes its main arrays by expansion term, not by parameter.
  It declares those fields as ``"term"`` and supplies the coordinate through
  :meth:`SchemaResult._extra_coords`.
* ``HDMRResult`` also emits ``S2``/``S3`` only when the fit has terms of that
  order. It says so through :meth:`SchemaResult._omit_fields`.

Bending those three through one generic path is how HDMR's ``n_terms`` axis
would end up labelled ``param`` in an exported netCDF file.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import jax.numpy as jnp
import numpy as np
import xarray as xr

from jaxgsa._core.samples import _jaxgsa_version
from jaxgsa._core.validation import _dims_and_coords
from jaxgsa._core.warning_types import JaxgsaWarning

if TYPE_CHECKING:
    import pandas as pd
    from jax import Array

    from jaxgsa.problem import Problem


def _method_name(cls: type) -> str:
    """Return the jaxgsa method package that defines a result class.

    Walks the class hierarchy for the first class defined inside ``jaxgsa``
    and returns its package name, for example ``"sobol"``. A subclass defined
    in user code therefore reports the method of its jaxgsa parent.

    Args:
        cls: A result class.

    Returns:
        The second part of the module path of the first jaxgsa class found.
        Falls back to the module's own second part, or its full name, when no
        class in the hierarchy lives in ``jaxgsa``.
    """
    for klass in cls.__mro__:
        parts = klass.__module__.split(".")
        if parts[0] == "jaxgsa" and len(parts) > 1:
            return parts[1]
    parts = cls.__module__.split(".")
    return parts[1] if len(parts) > 1 else parts[0]


@dataclass(frozen=True)
class CIInfo:
    """How a result's confidence intervals were produced.

    A ``*_conf`` array on its own is an interval of unknown level: nothing in
    it says whether it is a 95% or a 68% interval, which endpoint rule drew
    it, or how many resamples it rests on. This record carries that, so a
    plot can label its error bars and a reader can tell a wide interval from
    a badly under-resampled one.

    The bootstrap draws themselves are kept only when the caller asks. They
    are large: 1000 resamples of a ``(T=100, K=5, D=20)`` index array is
    80 MB, which is more than the rest of the result put together.

    Attributes:
        level: Two-sided confidence level the interval was drawn at, in
            ``(0, 1)``. This is the ``conf_level`` the analysis ran with.
        method: Endpoint rule actually used. ``"quantile"`` takes empirical
            bootstrap quantiles and ``"gaussian"`` takes a normal
            approximation. Every bootstrapping method offers both through its
            ``ci_method`` argument and records the one that ran.
        n_bootstrap: Number of bootstrap resamples drawn.
        replicates: The per-resample index values, keyed by the name of the
            estimate they belong to (``"S1"``, ``"mu_star"``, ...). Each
            array has a leading axis of length ``n_bootstrap`` followed by
            the shape of that estimate. ``None`` unless the analysis ran with
            ``keep_replicates=True``. Keep them to recompute an interval at
            another level, or to compute a bias-corrected one, without
            re-running the analysis.
    """

    level: float
    method: str
    n_bootstrap: int
    replicates: Mapping[str, "Array"] | None = field(default=None, repr=False)

    def __repr__(self) -> str:
        """Summarize the interval without printing the stored draws."""
        if self.replicates is None:
            kept = "None"
        else:
            kept = "{" + ", ".join(sorted(self.replicates)) + "}"
        return (
            f"CIInfo(level={self.level}, method={self.method!r}, "
            f"n_bootstrap={self.n_bootstrap}, replicates={kept})"
        )


Axes = Literal["param", "slice", "pair", "triple", "term", "term_only", "pair_only"]
"""Which axes a declared field spans, relative to the result's base dims.

The base dims come from :func:`jaxgsa._core.validation._dims_and_coords`:
``("param",)``, ``("output", "param")`` or ``("time", "output", "param")``.
Writing ``lead`` for the base dims without their trailing ``param``:

* ``"param"``: the base dims themselves, the shape of an index array.
* ``"slice"``: ``lead`` — one value per output slice, such as a variance or
  an RMSE.
* ``"pair"`` / ``"triple"``: ``lead`` plus two or three parameter axes, for
  an interaction matrix or tensor. They get their own ``param_i``/``param_j``
  /``param_k`` names, because xarray cannot give one dataset two dimensions
  called ``param``.
* ``"term"``: ``lead`` plus an HDMR expansion-term axis.
* ``"term_only"`` / ``"pair_only"``: the term axis, or the two parameter
  axes, with no output or time axes at all.
"""

_PAIR_DIMS = ("param_i", "param_j")
_TRIPLE_DIMS = ("param_i", "param_j", "param_k")


def _attr_value(value: Any) -> Any:
    """Coerce one provenance value into something a dataset can hold.

    ``xr.Dataset.attrs`` is written out as-is, so a JAX scalar would reach a
    netCDF file as a device array and reach a reader as
    ``Array(4, dtype=int32)``. The rule is: a string stays a string, and
    every other scalar becomes the plain Python ``int``, ``float`` or
    ``complex`` NumPy reads it as. A ``bool`` becomes the ``int`` 0 or 1,
    because netCDF cannot store a boolean attribute. Anything that is not a numeric scalar
    is recorded by its ``str``, so a declaration this function did not
    anticipate degrades to a readable label instead of raising in the middle
    of an export.

    ``None`` never arrives here: :meth:`SchemaResult._dataset_attrs` drops
    the key instead. netCDF has no null attribute, so a stored ``None``
    would fail at ``to_netcdf`` time, far from the result that wrote it. An
    absent key says the same thing and survives the round trip.

    Args:
        value: The declared provenance value read off the result.

    Returns:
        The value in a form ``xr.Dataset.attrs`` can carry.
    """
    if isinstance(value, str):
        return value
    array = np.asarray(value)
    if array.ndim == 0 and array.dtype.kind == "b":
        return int(array.item())
    if array.ndim == 0 and array.dtype.kind in "iufc":
        return array.item()
    return str(value)


_FRAME_DIMS = frozenset({"time", "output", "param"})
"""The dimensions a variable may have to appear in :func:`param_frame`."""

_BOUND_SUFFIXES = ("_lower", "_upper")


def param_frame(ds: xr.Dataset) -> pd.DataFrame:
    """Turn the per-parameter variables of a result dataset into a tidy table.

    A variable is kept when it has a ``param`` dimension and no other
    dimension than ``time`` and ``output``. That keeps one value per input
    (per output and time point) and leaves out the pair matrices (``S2``,
    on ``param_i``/``param_j``), HDMR's ``term`` arrays, Kucherenko's
    ``subset`` arrays and the per-output scalars such as ``variance``.

    The columns keep the dataset's variable names. Each point estimate is
    followed by its ``*_lower`` and ``*_upper`` bounds when the dataset has
    them. The index has one level per kept dimension, in the variables' own
    order, so ``param`` is the last level.

    Args:
        ds: A dataset produced by a result's ``to_dataset``.

    Returns:
        A :class:`pandas.DataFrame` indexed by ``param`` (and ``time`` and
        ``output`` when the result has them). ``DataFrame.attrs`` holds a
        copy of the dataset attributes.

    Raises:
        ValueError: If no variable in ``ds`` has a ``param`` dimension.
    """
    kept = [
        str(name)
        for name, var in ds.data_vars.items()
        if "param" in var.dims and set(var.dims) <= _FRAME_DIMS
    ]
    if not kept:
        raise ValueError(
            "this result has no per-parameter variable to put in a table; use to_dataset() instead"
        )
    bounds = {name for name in kept if name.endswith(_BOUND_SUFFIXES)}
    columns: list[str] = []
    for name in kept:
        if name in bounds and name.removesuffix("_lower").removesuffix("_upper") in kept:
            continue
        columns.append(name)
        columns.extend(f"{name}{sfx}" for sfx in _BOUND_SUFFIXES if f"{name}{sfx}" in bounds)
    dims = ds[kept[0]].dims
    frame = ds[columns].to_dataframe(dim_order=list(dims))
    frame = frame[columns]
    frame.attrs = dict(ds.attrs)
    return frame


@dataclass(frozen=True)
class FieldSpec:
    """One declared field of a result.

    Attributes:
        name: Attribute name on the result. A field whose value is ``None``
            is skipped everywhere.
        axes: Which axes the field spans. See :data:`Axes`.
        interval: Whether a companion ``<name>_conf`` field holds a
            ``(2, ...)`` ``[lower, upper]`` array. When it does, the dataset
            gains ``<name>_lower`` and ``<name>_upper``.
        dataset: Whether the field is exported by ``to_dataset``. A few
            fields are diagnostics that the dataset has never carried;
            declaring them with ``dataset=False`` keeps them in ``__repr__``
            without changing the export.
    """

    name: str
    axes: Axes = "param"
    interval: bool = False
    dataset: bool = True


@dataclass(frozen=True)
class ResultSchema:
    """What one result class exports, as data.

    Attributes:
        primary: Name of the field whose ``ndim`` and ``shape`` fix the base
            dimensions. Every other field is placed relative to it.
        fields: The declared array fields, in export order.
        meta: Names of scalar provenance fields (``mode``, ``backend``,
            ``order``, ...). They are shown in ``__repr__`` and written to
            ``xr.Dataset.attrs`` under the same names, so a saved dataset
            carries the same provenance the printed result does.
    """

    primary: str
    fields: tuple[FieldSpec, ...]
    meta: tuple[str, ...] = ()


class SchemaResult:
    """Derives ``to_dataset`` and ``__repr__`` from a declared schema.

    A result class mixes this in, sets :attr:`_schema`, and is decorated
    ``@dataclass(repr=False)`` so the generated ``__repr__`` does not shadow
    the one derived here.

    Attributes:
        _schema: The class's field declaration.
        _attr_aliases: Export names for ``meta`` fields whose own name
            cannot be a dataset attribute. Empty by default.
    """

    _schema: ClassVar[ResultSchema]
    _attr_aliases: ClassVar[dict[str, str]] = {}
    problem: "Problem"

    # -- hooks ------------------------------------------------------------

    def _base_dims(
        self, time_coords: np.ndarray | list | None
    ) -> tuple[tuple[str, ...], dict[str, Any]]:
        """Resolve the base dimensions and coordinates of this result.

        The default reads them off the primary field's ``ndim`` and
        ``shape``. Override it when the layout is fixed by something other
        than the array rank.

        Args:
            time_coords: Coordinate values for the time dimension, or
                ``None`` for integer indices.

        Returns:
            The base dims and the coordinate mapping to seed the dataset
            with.
        """
        arr = np.asarray(getattr(self, self._schema.primary))
        return _dims_and_coords(arr.ndim, arr.shape, self.problem, time_coords)

    def _extra_coords(self) -> dict[str, Any]:
        """Return coordinates the schema alone cannot supply.

        Returns:
            A mapping merged into the dataset coordinates. Empty by default;
            ``HDMRResult`` uses it for the ``term`` axis.
        """
        return {}

    def _omit_fields(self) -> frozenset[str]:
        """Return the names of declared fields this instance does not export.

        Returns:
            Field names to skip. Empty by default; ``HDMRResult`` uses it to
            drop ``S2``/``S3`` when the fit has no terms of that order.
        """
        return frozenset()

    def _dataset_attrs(self) -> dict[str, Any]:
        """Return the dataset-level attributes to record.

        The default exports the schema's :attr:`ResultSchema.meta` names, so
        one declaration drives both the summary and the export. A per-class
        attribute list would let the repr and the export drift apart.

        Each value passes through :func:`_attr_value`, and a value of
        ``None`` drops its key rather than writing a null. See that function
        for the rule.

        Override it only for a key the declaration cannot supply. An
        override that reproduces this mapping is drift waiting to happen; no
        result class needs one today.

        Every dataset also gets ``method`` (the method's package name, such
        as ``"sobol"``) and ``jaxgsa_version``, so a saved file says which
        analysis and which release made it.

        A saved interval must say what it is, so a result that carries a
        ``ci`` (a :class:`CIInfo`) also gets ``ci_level``, ``ci_method`` and
        ``ci_n_bootstrap`` written here, under those exact names. ``ci`` is
        not in ``_schema.meta``, because it is one object with three
        provenance values, not one; the loop above only handles a single
        scalar per name. The three values take the same
        :func:`_attr_value` coercion as every other attribute, so a
        ``conf_level`` the caller passed as a NumPy or JAX scalar still
        reaches a netCDF file as a plain number.

        Returns:
            A mapping written to ``xr.Dataset.attrs``.
        """
        attrs: dict[str, Any] = {
            "method": _method_name(type(self)),
            "jaxgsa_version": _jaxgsa_version(),
        }
        for name in self._schema.meta:
            value = getattr(self, name)
            if value is None:
                continue
            attrs[self._attr_aliases.get(name, name)] = _attr_value(value)
        ci = getattr(self, "ci", None)
        if ci is not None:
            attrs["ci_level"] = _attr_value(ci.level)
            attrs["ci_method"] = _attr_value(ci.method)
            attrs["ci_n_bootstrap"] = _attr_value(ci.n_bootstrap)
        return attrs

    # -- derivation -------------------------------------------------------

    def _field_dims(self, axes: Axes, base: tuple[str, ...]) -> tuple[str, ...]:
        """Map one field's declared axes onto concrete dimension names.

        Args:
            axes: The declared axes.
            base: This result's base dimensions.

        Returns:
            The dimension names for a field with those axes.
        """
        lead = base[:-1]
        if axes == "param":
            return base
        if axes == "slice":
            return lead
        if axes == "pair":
            return (*lead, *_PAIR_DIMS)
        if axes == "triple":
            return (*lead, *_TRIPLE_DIMS)
        if axes == "term":
            return (*lead, "term")
        if axes == "term_only":
            return ("term",)
        return _PAIR_DIMS

    def to_dataset(self, time_coords: np.ndarray | list | None = None) -> xr.Dataset:
        """Convert the result to a labeled xarray Dataset.

        Point estimates come first, then the ``*_lower`` / ``*_upper`` halves
        of every confidence interval. Splitting the ``(2, ...)`` interval
        arrays lets you select a bound by name instead of by integer index.

        Boolean provenance attributes (such as ``streamed``) are stored as the
        integers 0 and 1, because netCDF cannot hold a boolean attribute.

        Args:
            time_coords: Coordinate values for the time dimension of a
                time-resolved result. Defaults to integer indices.

        Returns:
            An :class:`xarray.Dataset` holding every declared field this
            result carries, on ``param`` / ``output`` / ``time`` dimensions.
        """
        base, coords = self._base_dims(time_coords)
        coords.update(self._extra_coords())
        omit = self._omit_fields()
        param_names = list(self.problem.names)

        specs = [
            spec
            for spec in self._schema.fields
            if spec.dataset and spec.name not in omit and getattr(self, spec.name) is not None
        ]

        def _name_axes(spec: FieldSpec) -> None:
            """Declare the coordinates of a field's extra axes.

            The coordinates belong to the axes, not to any one variable, so
            they are named whenever anything on that axis is exported. Naming
            them only alongside the point estimate would leave a lone
            ``*_lower`` unlabelled, and ``.sel(param_i=...)`` would raise.
            """
            dims = self._field_dims(spec.axes, base)
            for dim in dims:
                if dim in _TRIPLE_DIMS and dim not in coords:
                    coords[dim] = param_names

        data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
        for spec in specs:
            dims = self._field_dims(spec.axes, base)
            values = np.asarray(getattr(self, spec.name))
            data_vars[spec.name] = (dims, values)
            _name_axes(spec)

        for spec in self._schema.fields:
            if not (spec.interval and spec.dataset) or spec.name in omit:
                continue
            conf = getattr(self, f"{spec.name}_conf", None)
            if conf is None:
                continue
            dims = self._field_dims(spec.axes, base)
            conf = np.asarray(conf)
            data_vars[f"{spec.name}_lower"] = (dims, conf[0])
            data_vars[f"{spec.name}_upper"] = (dims, conf[1])
            _name_axes(spec)

        return xr.Dataset(data_vars, coords=coords, attrs=self._dataset_attrs())

    def to_pandas(self, time_coords: np.ndarray | list | None = None) -> pd.DataFrame:
        """Convert the per-parameter indices to a tidy pandas DataFrame.

        The table is built from :meth:`to_dataset`. It has one row per
        input, labelled with ``problem.names`` on an index level called
        ``param``. A result with several outputs or time points adds
        ``output`` and ``time`` index levels in front of ``param``.

        There is one column per index, named as in the dataset, and each
        index is followed by its ``*_lower`` and ``*_upper`` confidence
        bounds when the result has them.

        Variables without a single ``param`` axis are not in the table:
        second-order matrices such as ``S2``, HDMR's per-term arrays,
        Kucherenko's subset arrays and per-output scalars such as
        ``variance``. Read those from the dataset, for example
        ``result.to_dataset()["S2"].to_pandas()`` for a scalar output. An
        HDMR table therefore has only ``ST``; its first-order values are
        ``result.S1``.

        Args:
            time_coords: Coordinate values for the time dimension of a
                time-resolved result. Defaults to integer indices.

        Returns:
            A :class:`pandas.DataFrame`. ``DataFrame.attrs`` holds the same
            provenance attributes as the dataset.
        """
        return param_frame(self.to_dataset(time_coords))

    def __repr__(self) -> str:
        """Return a one-line summary: field shapes, then provenance.

        Arrays are reported by shape and the ``Problem`` is left out, so the
        summary stays one line no matter how large the analysis was.

        Returns:
            A string of the form
            ``SobolResult(S1=(3,), ST=(3,), ci=CIInfo(...))``.
        """
        parts = []
        for spec in self._schema.fields:
            value = getattr(self, spec.name, None)
            if value is None:
                continue
            parts.append(f"{spec.name}={tuple(np.shape(value))}")
            conf = getattr(self, f"{spec.name}_conf", None) if spec.interval else None
            if conf is not None:
                parts.append(f"{spec.name}_conf={tuple(np.shape(conf))}")
        for name in self._schema.meta:
            parts.append(f"{name}={getattr(self, name)!r}")
        ci = getattr(self, "ci", None)
        if ci is not None:
            parts.append(f"ci={ci!r}")
        return f"{type(self).__name__}({', '.join(parts)})"


AggregateWeights = Literal["variance", "equal"]
"""How :func:`aggregate_indices` weights the output slices."""


def aggregate_indices(
    result: SchemaResult,
    *,
    weights: AggregateWeights,
    variance: Any,
    extra_fields: tuple[FieldSpec, ...] = (),
) -> dict[str, Any]:
    """Collapse the output axes of a result's index fields into one index per input.

    This is the shared core of ``aggregate()`` on ``SobolResult``,
    ``KucherenkoResult`` and ``PCEResult``. It acts on every declared field
    that has an interval companion and the axes ``"param"`` or ``"pair"``
    (``S1``, ``ST``, ``S2``). Each one becomes ``sum_t w_t * field_t`` over
    the leading output axes, which stay in place with size 1. A result with
    no output axes (a ``(D,)`` index) comes back unchanged, and so does one
    whose output axes already have a single slice.

    A slice with weight exactly 0 (a constant output) adds 0, even when its
    index is ``NaN``. A slice with a positive weight and a ``NaN`` index makes
    the aggregate ``NaN``.

    The interval is never aggregated bound by bound, because that is not a
    valid interval. When the result kept its bootstrap draws, each draw is
    aggregated and the interval is drawn again from the aggregated draws.
    Otherwise the interval is dropped (``None``, ``ci`` too) and one
    :class:`JaxgsaWarning` says so.

    Args:
        result: A result whose primary field has 1, 2 or 3 dimensions.
        weights: ``"variance"`` weights slice ``t`` by ``V_t / sum(V)``.
            ``"equal"`` weights every slice by ``1 / n_slices``.
        variance: Output variance per slice, shape ``()``, ``(K,)`` or
            ``(T, K)``. It is read only for ``weights="variance"``.
        extra_fields: Index fields to collapse in addition to the schema's
            own, for a result that holds optional extra indices.

    Returns:
        A mapping of field name to new value, ready for
        :func:`dataclasses.replace`. It holds the aggregated index fields,
        their ``*_conf`` fields and ``ci``, and the summed ``variance`` when
        the result has a ``variance`` field.

    Warns:
        JaxgsaWarning: If the result has a bootstrap interval but kept no
            replicates, so the interval cannot be aggregated and is dropped.

    Raises:
        ValueError: If ``weights`` is not ``"variance"`` or ``"equal"``, if
            the result has no regular output layout, or if ``weights`` is
            ``"variance"`` and ``variance`` is missing or has the wrong shape.
    """
    if weights not in ("variance", "equal"):
        raise ValueError(f"weights must be 'variance' or 'equal', got {weights!r}")
    schema = result._schema
    primary = getattr(result, schema.primary)
    ndim = np.ndim(primary)
    if ndim not in (1, 2, 3):
        raise ValueError(
            f"cannot aggregate a result whose {schema.primary} has {ndim} dimensions: "
            "only the regular (D,), (K, D) and (T, K, D) layouts have output axes to collapse"
        )
    lead = tuple(np.shape(primary)[:-1])
    names = [
        spec
        for spec in (*schema.fields, *extra_fields)
        if spec.interval
        and spec.axes in ("param", "pair")
        and getattr(result, spec.name) is not None
    ]
    n_slices = int(np.prod(lead)) if lead else 1

    ci = getattr(result, "ci", None)
    out: dict[str, Any] = {}
    if n_slices == 1:
        # One output slice: the weights are 1, so the indices and their
        # intervals are already the aggregate.
        return out

    if weights == "equal":
        w = jnp.full(lead, 1.0 / n_slices)
    else:
        if variance is None:
            raise ValueError(
                "weights='variance' needs the per-output variance, but this result "
                "carries none (it was built without one). "
                "Use weights='equal', or run the analysis again."
            )
        v = jnp.asarray(variance)
        if v.shape != lead:
            raise ValueError(f"variance has shape {v.shape}, expected {lead}")
        w = v / jnp.sum(v)

    # A result that stores its own output variance keeps the summed variance.
    own_variance = getattr(result, "variance", None)
    if own_variance is not None:
        out["variance"] = jnp.asarray(own_variance).sum(keepdims=True)

    # The output axis now holds one slice, so its label is "aggregate". Keeping
    # the original output names would break to_dataset (K names, one slice).
    problem = getattr(result, "problem", None)
    if problem is not None:
        out["problem"] = problem.with_output_names(("aggregate",))

    def collapse(arr: Any, n_trailing: int, offset: int) -> Any:
        arr = jnp.asarray(arr)
        wb = w.reshape((1,) * offset + lead + (1,) * n_trailing).astype(arr.dtype)
        terms = jnp.where(wb == 0, 0.0, wb * arr)
        axes = tuple(range(offset, offset + len(lead)))
        return jnp.sum(terms, axis=axes, keepdims=True)

    points: dict[str, Any] = {}
    for spec in names:
        n_trailing = 1 if spec.axes == "param" else 2
        points[spec.name] = collapse(getattr(result, spec.name), n_trailing, 0)
        out[spec.name] = points[spec.name]

    replicates = ci.replicates if ci is not None else None
    have_conf = [s for s in names if getattr(result, f"{s.name}_conf", None) is not None]
    if ci is not None and replicates is not None and all(s.name in replicates for s in have_conf):
        from jaxgsa._core.bootstrap import interval

        draws = {
            s.name: collapse(replicates[s.name], 1 if s.axes == "param" else 2, 1)
            for s in have_conf
        }
        conf, new_ci = interval(
            {s.name: points[s.name] for s in have_conf},
            draws,
            level=ci.level,
            method=ci.method,
            n_bootstrap=ci.n_bootstrap,
            keep_replicates=True,
        )
        for s in have_conf:
            out[f"{s.name}_conf"] = conf[s.name]
        out["ci"] = new_ci
    else:
        if have_conf:
            warnings.warn(
                "aggregate() dropped the confidence intervals, because it needs the "
                "bootstrap replicates to aggregate them. Run analyze again with "
                "keep_replicates=True.",
                JaxgsaWarning,
                stacklevel=3,
            )
            for s in have_conf:
                out[f"{s.name}_conf"] = None
            out["ci"] = None
    return out
