"""Sobol/Saltelli sampling with unique user-facing rows.

The public contract of this module is intentionally split in two layers:

1. ``sample()`` returns only the unique rows that a user should evaluate.
2. ``SobolSamples`` also carries enough metadata to reconstruct the full
   expanded Saltelli layout later inside :func:`gsax.sobol.analyze`.

This avoids wasted model evaluations in low-dimensional cases where the
expanded Saltelli design contains exact duplicate rows.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import overload

import numpy as np
from scipy.stats.qmc import Sobol

from gsax._core.sampling import (
    _is_power_of_2,
    _next_power_of_2,
    _power_of_2_error,
    _stable_unique_rows,
    _transform_samples,
)
from gsax.problem import Problem, _normalized_input_to_dict


@dataclass(frozen=True)
class SobolSamples:
    """Unique Sobol samples plus metadata for Saltelli reconstruction.

    Returned by :func:`gsax.sobol.sample`. Evaluate your model at every row of
    ``samples`` (in order) and pass this object together with the outputs to
    :func:`gsax.sobol.analyze`.

    Note the two row counts: ``n_runs`` is the number of *unique* rows you
    must evaluate (one model run per row), while ``n_expanded`` is the
    (larger or equal) pre-deduplication size of the full Saltelli design that
    the analysis reconstructs internally.

    Attributes:
        samples: Unique rows to evaluate with the user's model. Shape
            ``(n_runs, D)`` where ``D`` is the number of parameters, in
            physical units (each Sobol marginal transformed into the
            problem's declared input distribution).
        sample_ids: Stable integer identifiers aligned 1:1 with ``samples``.
            Useful for joining model outputs back onto the sampling table.
        n_expanded: Row count of the full expanded Saltelli layout before
            deduplication. This is the number of rows analyzed internally.
        expanded_to_unique: Integer index map of shape ``(n_expanded,)``.
            For each expanded Saltelli row, gives the corresponding row index in
            ``samples``.
        base_n: Number of base Sobol points used to construct the Saltelli
            design. Always a power of 2.
        n_params: Number of problem dimensions ``D``.
        calc_second_order: Whether the expanded design includes the extra
            cross-matrices needed for second-order Sobol indices.
        problem: Problem definition used to transform the samples.
    """

    samples: np.ndarray  # shape (n_unique, D), scaled to bounds
    sample_ids: np.ndarray
    n_expanded: int
    expanded_to_unique: np.ndarray
    base_n: int
    n_params: int
    calc_second_order: bool
    problem: Problem

    @property
    def n_runs(self) -> int:
        """Number of unique rows in ``samples`` (model runs to evaluate)."""
        return self.samples.shape[0]

    @overload
    def downsample(self, base_n: int) -> SobolSamples: ...

    @overload
    def downsample(self, base_n: int, Y: np.ndarray) -> tuple[SobolSamples, np.ndarray]: ...

    def downsample(
        self, base_n: int, Y: np.ndarray | None = None
    ) -> SobolSamples | tuple[SobolSamples, np.ndarray]:
        """Return a smaller ``SobolSamples`` by prefix-slicing to a lower ``base_n``.

        Sobol sequences are prefix-nested: the first *K* base points of a
        draw with *N > K* base points are bit-identical to drawing *K*
        base points directly (same seed and scramble).  This means you can
        simulate the model once at the largest ``base_n`` and recover exact
        results for any smaller power-of-2 ``base_n`` by slicing — no
        re-simulation needed.

        Optionally pass ``Y`` (model outputs aligned with ``samples``) to
        get the corresponding output slice back — similar to how
        ``sklearn.model_selection.train_test_split`` accepts both *X* and
        *y*.

        This property does **not** hold for Latin Hypercube Sampling (LHS),
        whose stratification depends on *N*.

        Args:
            base_n: Target base size (must be a power of 2 and
                ``<= self.base_n``).
            Y: Model outputs with shape ``(n_runs, ...)``.  When provided,
                the matching prefix is returned alongside the new result.

        Returns:
            ``SobolSamples`` when called without ``Y``, or
            ``(SobolSamples, Y_small)`` when ``Y`` is provided.

        Raises:
            ValueError: If ``base_n`` is not a power of 2, exceeds
                ``self.base_n``, or ``Y`` has too few rows.
        """
        if not _is_power_of_2(base_n):
            raise ValueError(_power_of_2_error("base_n", base_n, reason="Sobol' sequence balance"))
        if base_n > self.base_n:
            raise ValueError(
                f"Cannot upsample: requested base_n={base_n} > current base_n={self.base_n}"
            )
        if Y is not None and Y.shape[0] != self.n_runs:
            raise ValueError(f"Y.shape[0]={Y.shape[0]} does not match n_runs={self.n_runs}")
        if base_n == self.base_n:
            return (self, Y) if Y is not None else self

        step = _saltelli_step(self.n_params, self.calc_second_order)
        new_expanded_n = base_n * step
        new_exp2uniq = self.expanded_to_unique[:new_expanded_n]
        n_unique_new = int(new_exp2uniq.max()) + 1

        sr_small = SobolSamples(
            samples=self.samples[:n_unique_new].copy(),
            sample_ids=np.arange(n_unique_new, dtype=np.int64),
            n_expanded=new_expanded_n,
            expanded_to_unique=new_exp2uniq.copy(),
            base_n=base_n,
            n_params=self.n_params,
            calc_second_order=self.calc_second_order,
            problem=self.problem,
        )

        if Y is not None:
            return sr_small, Y[:n_unique_new].copy()
        return sr_small

    def save(self, path: str | Path) -> None:
        """Save the full design to one compressed NPZ file.

        The file contains the unique sample matrix, the sample identifiers,
        the Saltelli expansion map (omitted when it is a trivial identity
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
        path = _npz_path(path)
        try:
            gsax_version = _pkg_version("gsax")
        except PackageNotFoundError:
            gsax_version = "unknown"
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
            "gsax_version": gsax_version,
            "problem": {
                "names": list(self.problem.names),
                "input_specs": [
                    _normalized_input_to_dict(spec) for spec in self.problem.input_specs
                ],
                "output_names": list(self.problem.output_names)
                if self.problem.output_names is not None
                else None,
            },
            "base_n": self.base_n,
            "calc_second_order": self.calc_second_order,
            "n_expanded": self.n_expanded,
            "identity_mapping": identity_mapping,
        }
        metadata = np.asarray(json.dumps(meta))
        if identity_mapping:
            np.savez_compressed(
                path,
                samples=self.samples,
                sample_ids=self.sample_ids,
                metadata=metadata,
            )
        else:
            np.savez_compressed(
                path,
                samples=self.samples,
                sample_ids=self.sample_ids,
                expanded_to_unique=self.expanded_to_unique,
                metadata=metadata,
            )

    @classmethod
    def load(cls, path: str | Path) -> "SobolSamples":
        """Load a design saved by :meth:`save`.

        Args:
            path: Path to the saved design. If it does not already end in
                ``.npz``, the suffix is appended before opening (mirroring
                the convention used by :meth:`save`), so the file read may
                differ from the exact string passed.

        Returns:
            The reconstructed ``SobolSamples``, equal to the instance that
            was saved.

        Raises:
            FileNotFoundError: If no file exists at the resolved path.
            KeyError: If the file is missing required arrays or metadata
                entries (i.e. it was not written by :meth:`save`).
        """
        from gsax.problem import _normalize_input_spec

        with np.load(_npz_path(path), allow_pickle=False) as data:
            meta = json.loads(data["metadata"].item())
            problem_meta = meta["problem"]
            output_names = problem_meta["output_names"]
            problem = Problem._from_normalized_inputs(
                names=tuple(problem_meta["names"]),
                input_specs=tuple(
                    _normalize_input_spec(spec) for spec in problem_meta["input_specs"]
                ),
                output_names=tuple(output_names) if output_names is not None else None,
            )
            n_expanded = int(meta["n_expanded"])
            if meta["identity_mapping"]:
                expanded_to_unique = np.arange(n_expanded, dtype=np.int64)
            else:
                expanded_to_unique = data["expanded_to_unique"]
            return cls(
                samples=data["samples"],
                sample_ids=data["sample_ids"],
                n_expanded=n_expanded,
                expanded_to_unique=expanded_to_unique,
                base_n=int(meta["base_n"]),
                n_params=problem.num_vars,
                calc_second_order=bool(meta["calc_second_order"]),
                problem=problem,
            )


def _npz_path(path: str | Path) -> Path:
    """Return a path with the canonical ``.npz`` suffix.

    A missing suffix is *appended* (never substituted), matching NumPy's own
    ``savez`` convention: ``"run.A"`` becomes ``"run.A.npz"``, not
    ``"run.npz"``. Substituting via ``Path.with_suffix`` would make dotted
    stems like ``"run.A"`` and ``"run.B"`` silently collide on disk.
    """
    path = Path(path)
    return path if path.suffix == ".npz" else Path(str(path) + ".npz")


def _saltelli_step(n_params: int, calc_second_order: bool) -> int:
    """Return the number of expanded Saltelli rows per base Sobol point."""
    # Saltelli step = A + D*AB [+ D*BA] + B = D+2 [or 2D+2]
    return 2 * n_params + 2 if calc_second_order else n_params + 2


def _build_expanded_samples(
    n_params: int,
    base_n: int,
    *,
    calc_second_order: bool,
    scramble: bool,
    seed: int | np.random.Generator | None,
) -> np.ndarray:
    """Generate the full expanded Saltelli matrix for a fixed ``base_n``.

    The returned matrix still includes exact duplicate rows when the Saltelli
    construction collapses in low dimensions. Deduplication happens later.
    """
    D = n_params
    # Draw from a 2D-dimensional Sobol sequence; first D dims become matrix A, last D become B
    sampler = Sobol(d=2 * D, scramble=scramble, seed=seed)
    base = sampler.random(base_n)

    # Split the 2D-dimensional Sobol draw into the standard Saltelli base
    # matrices A and B, each with shape (base_n, D).
    A = base[:, :D]
    B = base[:, D:]
    # Saltelli interleaved layout per base point i:
    #   [A_i, AB_0, ..., AB_{D-1}, BA_0, ..., BA_{D-1}, B_i]
    # AB_j = A with column j replaced by B's column j (and vice-versa for BA_j).
    # .copy() prevents aliasing: without it, overwriting element j would mutate
    # the original A[i] or B[i] row shared across iterations.
    rows = []
    for i in range(base_n):
        rows.append(A[i])
        for j in range(D):
            AB_j = A[i].copy()
            AB_j[j] = B[i, j]
            rows.append(AB_j)
        if calc_second_order:
            for j in range(D):
                BA_j = B[i].copy()
                BA_j[j] = A[i, j]
                rows.append(BA_j)
        rows.append(B[i])

    return np.array(rows)


def _print_sampling_summary(
    *,
    n_params: int,
    target_n: int,
    n_runs: int,
    n_expanded: int,
    base_n: int,
    calc_second_order: bool,
    scramble: bool,
) -> None:
    """Print a compact summary of the generated unique Sobol design."""
    duplicates_removed = n_expanded - n_runs
    duplicate_fraction = duplicates_removed / n_expanded if n_expanded else 0.0
    order_label = "second-order" if calc_second_order else "first/total-order"
    print(
        "gsax.sobol.sample: "
        f"D={n_params}, mode={order_label}, base_n={base_n}, "
        f"requested_runs>={target_n}, n_runs={n_runs}, "
        f"n_expanded={n_expanded}, duplicates_removed={duplicates_removed} "
        f"({duplicate_fraction:.1%}), scramble={scramble}"
    )


def sample(
    problem: Problem,
    n_samples: int,
    *,
    base_n: int | None = None,
    calc_second_order: bool = True,
    scramble: bool = True,
    seed: int | np.random.Generator | None = None,
    verbose: bool = True,
) -> SobolSamples:
    """Generate the input samples needed for Sobol analysis with ``gsax.sobol.analyze``.

    Typical usage: call this once, evaluate your model at every row of the
    returned ``result.samples`` (shape ``(n_runs, D)``), then pass the result
    object and your outputs to :func:`gsax.sobol.analyze`.

    Internally this builds a Saltelli design — the structured layout of base
    and column-swapped sample matrices that Sobol index estimators require —
    for a candidate ``base_n``, then removes exact duplicate rows while
    preserving first-occurrence order (in low dimensions the Saltelli
    construction repeats rows, and evaluating a model twice on the same input
    is wasted work). If the unique matrix is still smaller than the requested
    evaluation budget, ``base_n`` is doubled and the process repeats until
    enough unique rows are available.

    Args:
        problem: Problem definition with parameter names and distributions.
        n_samples: Minimum desired number of unique model evaluations. The
            returned design may contain somewhat more rows than this, never
            fewer. Ignored when ``base_n`` is provided.
        base_n: If given, use this exact Sobol base size (must be a power
            of 2) instead of searching for one. The expanded Saltelli
            design will have ``base_n * (2*D + 2)`` rows (second order)
            or ``base_n * (D + 2)`` rows (first/total only). This gives
            direct control over the sampling budget.
        calc_second_order: If ``True`` (default), include the extra
            cross-matrices needed to estimate second-order (pairwise
            interaction) Sobol indices.  Set to ``False`` if you only need
            first/total-order indices — it nearly halves the evaluation
            budget (Saltelli step ``D + 2`` instead of ``2*D + 2``).
        scramble: Whether to apply Owen scrambling to the Sobol sequence
            (recommended; randomizes the sequence so different seeds give
            statistically independent designs).
        seed: Random seed or generator for reproducibility.
        verbose: If ``True`` (default), print a short summary describing the
            requested unique count, returned unique count, expanded Saltelli
            size, and how many duplicate rows were removed.

    Returns:
        SobolSamples with a unique sample matrix plus expansion metadata for
        later Sobol analysis.
    """
    D = problem.num_vars
    # Rows per base point -- determines the ratio between base_n and total expanded rows
    step = _saltelli_step(D, calc_second_order)

    if base_n is not None:
        if not _is_power_of_2(base_n):
            raise ValueError(_power_of_2_error("base_n", base_n, reason="Sobol' sequence balance"))
        target_n = None
    else:
        target_n = max(1, n_samples)
        # Estimate initial base_n from requested unique count,
        # rounding up to next power of 2 (Sobol sequence requirement).
        base_n = _next_power_of_2(math.ceil(target_n / step))

    expanded_samples_unit = _build_expanded_samples(
        D,
        base_n,
        calc_second_order=calc_second_order,
        scramble=scramble,
        seed=seed,
    )
    expanded_samples = _transform_samples(problem, expanded_samples_unit)
    unique_samples, expanded_to_unique = _stable_unique_rows(expanded_samples)

    if target_n is not None:
        # Deduplication may reduce unique count below target;
        # double base_n and rebuild until we have enough.
        while unique_samples.shape[0] < target_n:
            base_n *= 2
            expanded_samples_unit = _build_expanded_samples(
                D,
                base_n,
                calc_second_order=calc_second_order,
                scramble=scramble,
                seed=seed,
            )
            expanded_samples = _transform_samples(problem, expanded_samples_unit)
            unique_samples, expanded_to_unique = _stable_unique_rows(expanded_samples)

    sample_ids = np.arange(unique_samples.shape[0], dtype=np.int64)
    if verbose:
        _print_sampling_summary(
            n_params=D,
            target_n=target_n if target_n is not None else unique_samples.shape[0],
            n_runs=unique_samples.shape[0],
            n_expanded=expanded_samples.shape[0],
            base_n=base_n,
            calc_second_order=calc_second_order,
            scramble=scramble,
        )

    return SobolSamples(
        samples=unique_samples,
        sample_ids=sample_ids,
        n_expanded=expanded_samples.shape[0],
        expanded_to_unique=expanded_to_unique,
        base_n=base_n,
        n_params=D,
        calc_second_order=calc_second_order,
        problem=problem,
    )
