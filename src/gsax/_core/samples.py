"""Private shared base for deduplicated sample designs.

Samplers that follow the two-layer contract (return only the *unique* rows a
user must evaluate, plus an index map back to the full expanded design) share
their dedup bookkeeping, output re-expansion, NPZ persistence skeleton, and
prefix-slicing mechanics through :class:`UniqueDesignSamples`.

Nothing in this module is public API. Concrete sample classes
(``gsax.sobol.SobolSamples``, ``gsax.morris.MorrisSamples``) remain the
user-facing surface; this base may be promoted later if a public extension
point is warranted.
"""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Self

import jax.numpy as jnp
import numpy as np

from gsax.problem import Problem, _normalize_input_spec, _normalized_input_to_dict

if TYPE_CHECKING:
    from jax import Array

# NPZ array names owned by the base skeleton; everything else in a saved file
# belongs to the concrete subclass (collected into the ``arrays`` payload).
_RESERVED_ARRAY_NAMES = frozenset({"samples", "expanded_to_unique", "metadata"})


def _npz_path(path: str | Path) -> Path:
    """Return a path with the canonical ``.npz`` suffix.

    A missing suffix is *appended* (never substituted), matching NumPy's own
    ``savez`` convention: ``"run.A"`` becomes ``"run.A.npz"``, not
    ``"run.npz"``. Substituting via ``Path.with_suffix`` would make dotted
    stems like ``"run.A"`` and ``"run.B"`` silently collide on disk.
    """
    path = Path(path)
    return path if path.suffix == ".npz" else Path(str(path) + ".npz")


def _gsax_version() -> str:
    """Return the installed gsax version, or ``"unknown"`` outside a package."""
    try:
        return _pkg_version("gsax")
    except PackageNotFoundError:
        return "unknown"


def _problem_to_meta(problem: Problem) -> dict[str, Any]:
    """Serialize a :class:`Problem` to a JSON-compatible metadata dict."""
    return {
        "names": list(problem.names),
        "input_specs": [_normalized_input_to_dict(spec) for spec in problem.input_specs],
        "output_names": list(problem.output_names) if problem.output_names is not None else None,
    }


def _problem_from_meta(problem_meta: Mapping[str, Any]) -> Problem:
    """Reconstruct a :class:`Problem` from its metadata dict."""
    output_names = problem_meta["output_names"]
    return Problem._from_normalized_inputs(
        names=tuple(problem_meta["names"]),
        input_specs=tuple(_normalize_input_spec(spec) for spec in problem_meta["input_specs"]),
        output_names=tuple(output_names) if output_names is not None else None,
    )


class UniqueDesignSamples:
    """Shared machinery for unique-row sample designs (private base).

    Subclasses are frozen dataclasses that declare (at least) the attributes
    annotated below. The base contributes, once for all designs:

    - the two row counts (``n_runs`` unique rows to evaluate vs.
      ``n_expanded`` pre-dedup design rows) and the ``expanded_to_unique``
      index map connecting them;
    - :meth:`expand_outputs`, gathering unique-row model outputs back into
      the full expanded layout;
    - the NPZ :meth:`save`/:meth:`load` skeleton (one compressed file, JSON
      metadata blob, identity-mapping optimization);
    - :meth:`_prefix_slice`, the mechanics behind per-class ``downsample``.

    Persistence hooks (template method pattern): subclasses override
    :meth:`_extra_arrays` / :meth:`_extra_metadata` to declare their design
    specific payload, and :meth:`_from_payload` to rebuild an instance from
    the loaded common fields plus that payload.
    """

    # Attributes every concrete design dataclass must declare as fields.
    samples: np.ndarray
    n_expanded: int
    expanded_to_unique: np.ndarray
    problem: Problem

    @property
    def n_runs(self) -> int:
        """Number of unique rows in ``samples`` (model runs to evaluate)."""
        return self.samples.shape[0]

    # ------------------------------------------------------------------
    # Output re-expansion
    # ------------------------------------------------------------------

    def expand_outputs(self, Y: Array) -> Array:
        """Rebuild expanded-layout outputs from unique user-evaluated outputs.

        The sampler deduplicates the design before returning samples to the
        user, so Y has only unique rows (shape ``(n_runs, ...)``). This
        method gathers them back into the full expanded layout the
        estimators expect, shape ``(n_expanded, ...)``.

        Args:
            Y: Model outputs aligned with ``samples``, shape ``(n_runs, ...)``.

        Returns:
            Outputs in the expanded design layout, shape ``(n_expanded, ...)``.

        Raises:
            ValueError: If ``Y``'s first axis does not match ``n_runs``.
        """
        if Y.shape[0] != self.n_runs:
            raise ValueError(
                f"Y.shape[0] must match sampling_result.n_runs ({self.n_runs}), got {Y.shape[0]}"
            )
        # expanded_to_unique[i] gives the unique-row index for expanded position i
        expanded_to_unique = jnp.asarray(self.expanded_to_unique)
        return jnp.take(Y, expanded_to_unique, axis=0)

    # ------------------------------------------------------------------
    # Downsample mechanics (semantics stay per-class)
    # ------------------------------------------------------------------

    def _validate_downsample_Y(self, Y: np.ndarray | None) -> None:
        """Raise if optional co-sliced outputs are misaligned with ``samples``."""
        if Y is not None and Y.shape[0] != self.n_runs:
            raise ValueError(f"Y.shape[0]={Y.shape[0]} does not match n_runs={self.n_runs}")

    def _prefix_slice(
        self, new_expanded_n: int, Y: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, int, np.ndarray | None]:
        """Slice the design to its first ``new_expanded_n`` expanded rows.

        First-occurrence dedup order makes the unique index set of any
        expanded prefix itself a prefix, so ``max() + 1`` on the sliced index
        map recovers the new unique count.

        Args:
            new_expanded_n: Expanded row count of the smaller design.
            Y: Optional model outputs aligned with ``samples`` to co-slice.

        Returns:
            ``(samples_small, expanded_to_unique_small, n_unique_new,
            Y_small)`` where ``Y_small`` is ``None`` when ``Y`` is ``None``.
            All returned arrays are copies (no memory shared with ``self``).
        """
        new_exp2uniq = self.expanded_to_unique[:new_expanded_n].copy()
        n_unique_new = int(new_exp2uniq.max()) + 1
        samples_small = self.samples[:n_unique_new].copy()
        Y_small = Y[:n_unique_new].copy() if Y is not None else None
        return samples_small, new_exp2uniq, n_unique_new, Y_small

    # ------------------------------------------------------------------
    # Persistence hooks
    # ------------------------------------------------------------------

    def _extra_arrays(self) -> dict[str, np.ndarray]:
        """Subclass hook: design-specific arrays persisted in the NPZ file.

        Keys must not collide with the base-owned array names ``samples``,
        ``expanded_to_unique``, or ``metadata``.
        """
        return {}

    def _extra_metadata(self) -> dict[str, Any]:
        """Subclass hook: design-specific JSON-serializable metadata entries.

        Merged into the top level of the metadata blob next to the common
        keys (``gsax_version``, ``problem``, ``n_expanded``,
        ``identity_mapping``).
        """
        return {}

    @classmethod
    def _from_payload(
        cls,
        *,
        samples: np.ndarray,
        n_expanded: int,
        expanded_to_unique: np.ndarray,
        problem: Problem,
        arrays: Mapping[str, np.ndarray],
        meta: Mapping[str, Any],
    ) -> Self:
        """Subclass hook: rebuild an instance from a loaded NPZ payload.

        Args:
            samples: Unique sample matrix, shape ``(n_runs, D)``.
            n_expanded: Pre-dedup expanded design size.
            expanded_to_unique: Index map, identity-rebuilt when it was
                omitted from the file.
            problem: Reconstructed problem definition.
            arrays: All non-base arrays stored in the file (the subclass's
                :meth:`_extra_arrays` payload).
            meta: Full metadata dict (common keys plus the subclass's
                :meth:`_extra_metadata` payload).

        Returns:
            The reconstructed design instance.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Persistence skeleton
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the full design to one compressed NPZ file.

        The file contains the unique sample matrix, any design-specific
        arrays, the expansion map (omitted when it is a trivial identity
        mapping, i.e. the design has no duplicate rows), and a JSON metadata
        blob describing the problem definition, the design parameters, and
        the gsax version that wrote the file.

        Args:
            path: Destination file path. If it does not already end in
                ``.npz``, the suffix is appended (matching NumPy's ``savez``
                convention), so the file written to disk may differ from the
                exact string passed — e.g. ``"run.A"`` is saved as
                ``"run.A.npz"``.
        """
        # Identity-mapping optimization: when no duplicate rows were removed,
        # expanded_to_unique is exactly arange(n_expanded) (the common
        # case in higher dimensions). Record a boolean flag instead of storing
        # the full index array; load() reconstructs it with np.arange.
        identity_mapping = bool(
            np.array_equal(
                self.expanded_to_unique,
                np.arange(self.n_expanded, dtype=self.expanded_to_unique.dtype),
            )
        )
        meta = {
            "gsax_version": _gsax_version(),
            "problem": _problem_to_meta(self.problem),
            "n_expanded": self.n_expanded,
            "identity_mapping": identity_mapping,
            **self._extra_metadata(),
        }
        arrays: dict[str, np.ndarray] = {"samples": self.samples, **self._extra_arrays()}
        overlap = _RESERVED_ARRAY_NAMES.intersection(arrays) - {"samples"}
        if overlap:
            raise ValueError(f"_extra_arrays() may not use reserved names: {sorted(overlap)}")
        if not identity_mapping:
            arrays["expanded_to_unique"] = self.expanded_to_unique
        arrays["metadata"] = np.asarray(json.dumps(meta))
        np.savez_compressed(_npz_path(path), allow_pickle=False, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load a design saved by :meth:`save`.

        Args:
            path: Path to the saved design. If it does not already end in
                ``.npz``, the suffix is appended before opening (mirroring
                the convention used by :meth:`save`), so the file read may
                differ from the exact string passed.

        Returns:
            The reconstructed design, equal to the instance that was saved.

        Raises:
            FileNotFoundError: If no file exists at the resolved path.
            KeyError: If the file is missing required arrays or metadata
                entries (i.e. it was not written by :meth:`save`).
        """
        with np.load(_npz_path(path), allow_pickle=False) as data:
            meta = json.loads(data["metadata"].item())
            problem = _problem_from_meta(meta["problem"])
            n_expanded = int(meta["n_expanded"])
            if meta["identity_mapping"]:
                expanded_to_unique = np.arange(n_expanded, dtype=np.int64)
            else:
                expanded_to_unique = data["expanded_to_unique"]
            arrays = {name: data[name] for name in data.files if name not in _RESERVED_ARRAY_NAMES}
            return cls._from_payload(
                samples=data["samples"],
                n_expanded=n_expanded,
                expanded_to_unique=expanded_to_unique,
                problem=problem,
                arrays=arrays,
                meta=meta,
            )
