"""Where a result declares its fields, and what is derived from that.

Nothing in this module is public API. See :mod:`jaxgsa._core`.

Before this module, each of the thirteen result classes wrote its own
``to_dataset``. The four results with a second-order array each carried their
own copy of the ``param_i``/``param_j`` splice, five results had no
``__repr__`` at all (so printing one dumped whole index arrays and the entire
``Problem``), and the eight that had one used three different styles. The
export schema of a result was therefore only knowable by reading its
``to_dataset`` body.

A result now declares its fields once, as a :class:`ResultSchema`, and
:class:`SchemaResult` derives ``to_dataset``, ``__repr__`` and the
``*_lower``/``*_upper`` split of a confidence array from that declaration.

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

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np
import xarray as xr

from jaxgsa._core.validation import _dims_and_coords

if TYPE_CHECKING:
    from jax import Array

    from jaxgsa.problem import Problem


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
            approximation; ``jaxgsa.sobol`` and ``jaxgsa.morris`` offer both.
            The other three bootstrapping methods are hard-wired to the
            percentile interval and record ``"quantile"``.
        n_resamples: Number of bootstrap resamples drawn.
        replicates: The per-resample index values, keyed by the name of the
            estimate they belong to (``"S1"``, ``"mu_star"``, ...). Each
            array has a leading axis of length ``n_resamples`` followed by
            the shape of that estimate. ``None`` unless the analysis ran with
            ``keep_replicates=True``. Keep them to recompute an interval at
            another level, or to compute a bias-corrected one, without
            re-running the analysis.
    """

    level: float
    method: str
    n_resamples: int
    replicates: Mapping[str, "Array"] | None = field(default=None, repr=False)

    def __repr__(self) -> str:
        """Summarize the interval without printing the stored draws."""
        if self.replicates is None:
            kept = "None"
        else:
            kept = "{" + ", ".join(sorted(self.replicates)) + "}"
        return (
            f"CIInfo(level={self.level}, method={self.method!r}, "
            f"n_resamples={self.n_resamples}, replicates={kept})"
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
            ``order``, ...) shown in ``__repr__``.
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
    """

    _schema: ClassVar[ResultSchema]
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

        Returns:
            A mapping written to ``xr.Dataset.attrs``. Empty by default.
        """
        return {}

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
            """Declare the coordinates of a field's extra parameter axes.

            The coordinates belong to the axes, not to any one variable, so
            they are named whenever anything on that axis is exported. Naming
            them only alongside the point estimate would leave a lone
            ``*_lower`` unlabelled, and ``.sel(param_i=...)`` would raise.
            """
            for dim in self._field_dims(spec.axes, base):
                if dim in _TRIPLE_DIMS and dim not in coords:
                    coords[dim] = param_names

        data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
        for spec in specs:
            dims = self._field_dims(spec.axes, base)
            data_vars[spec.name] = (dims, np.asarray(getattr(self, spec.name)))
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
