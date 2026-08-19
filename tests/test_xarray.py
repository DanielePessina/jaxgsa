"""Tests for xarray Dataset conversion of SobolResult and HDMRResult."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxgsa
from jaxgsa._core.invalid import InvalidReport, InvalidUnit
from jaxgsa.hdmr import HDMRResult
from jaxgsa.problem import Problem
from jaxgsa.sobol._result import SobolResult


# Every result class carries an invalid report. These tests build results by
# hand rather than by running an analysis, so they need a stand-in that says
# the check ran and found nothing.
def _clean_invalid(unit: InvalidUnit) -> InvalidReport:
    """Build an empty report, for results assembled by hand in these tests.

    These tests build result objects directly to exercise ``to_dataset()``,
    so no analysis ran and no check ran either. The unit is passed in rather
    than fixed, because it is a property of the method the result came from:
    a Saltelli-based result counts groups, a given-data one counts rows.
    """
    return InvalidReport(
        policy="raise",
        unit=unit,
        n_units=0,
        n_invalid=0,
        unit_indices=(),
        row_indices=(),
        bad_row_indices=(),
        sources=(),
    )


_CLEAN_INVALID = _clean_invalid(InvalidUnit.SALTELLI_GROUP)
_CLEAN_INVALID_ROW = _clean_invalid(InvalidUnit.ROW)


@pytest.fixture
def problem():
    return Problem.from_dict({"x1": (0, 1), "x2": (0, 1), "x3": (0, 1)})


# ── SobolResult tests ──────────────────────────────────────────────────────────


class TestSAResultToDataset:
    def test_default_coordinate_values(self, problem):
        """The generated coordinate values, for every output rank.

        The parameter axis is labelled with the problem's names, an unnamed
        output axis becomes ``y0``/``y1``, and an unlabelled time axis becomes
        the integers. The dimension names themselves are pinned for every
        result class by ``test_result_schema.py``.
        """
        scalar = SobolResult(
            S1=jnp.array([0.1, 0.2, 0.3]),
            ST=jnp.array([0.4, 0.5, 0.6]),
            S2=None,
            problem=problem,
            invalid=_CLEAN_INVALID,
        ).to_dataset()
        assert list(scalar.coords["param"].values) == ["x1", "x2", "x3"]

        multi = SobolResult(
            S1=jnp.ones((2, 3)),
            ST=jnp.ones((2, 3)),
            S2=None,
            problem=problem,
            invalid=_CLEAN_INVALID,
        ).to_dataset()
        assert list(multi.coords["output"].values) == ["y0", "y1"]

        series = SobolResult(
            S1=jnp.ones((4, 2, 3)),
            ST=jnp.ones((4, 2, 3)),
            S2=None,
            problem=problem,
            invalid=_CLEAN_INVALID,
        ).to_dataset()
        assert list(series.coords["time"].values) == [0, 1, 2, 3]

    def test_custom_output_names(self):
        """Problem with explicit output_names passes through."""
        p = Problem(
            names=("x1", "x2"),
            bounds=((0, 1), (0, 1)),
            output_names=("temp", "pressure"),
        )
        r = SobolResult(
            S1=jnp.ones((2, 2)),
            ST=jnp.ones((2, 2)),
            S2=None,
            problem=p,
            invalid=_CLEAN_INVALID,
        )
        ds = r.to_dataset()
        assert list(ds.coords["output"].values) == ["temp", "pressure"]

    def test_custom_time_coords(self, problem):
        """Float time coords passed to to_dataset()."""
        r = SobolResult(
            S1=jnp.ones((3, 2, 3)),
            ST=jnp.ones((3, 2, 3)),
            S2=None,
            problem=problem,
            invalid=_CLEAN_INVALID,
        )
        ds = r.to_dataset(time_coords=[0.0, 0.5, 1.0])
        np.testing.assert_allclose(ds.coords["time"].values, [0.0, 0.5, 1.0])

    def test_output_names_length_mismatch(self):
        """Mismatched output_names raises ValueError."""
        p = Problem(
            names=("x1", "x2"),
            bounds=((0, 1), (0, 1)),
            output_names=("temp",),  # only 1, but K=2
        )
        r = SobolResult(
            S1=jnp.ones((2, 2)),
            ST=jnp.ones((2, 2)),
            S2=None,
            problem=p,
            invalid=_CLEAN_INVALID,
        )
        with pytest.raises(ValueError, match="output_names length"):
            r.to_dataset()

    @pytest.mark.parametrize("ci_method", ["quantile", "gaussian"])
    def test_analyze_bootstrap_export_preserves_lower_upper_variables(self, ci_method):
        """Real bootstrap analyze() output exports lower/upper CI variables."""
        problem = Problem.from_dict(
            {"x1": (0.0, 1.0), "x2": (0.0, 1.0), "x3": (0.0, 1.0)},
            output_names=("response",),
        )
        sampling_result = jaxgsa.sobol.sample(
            problem,
            n_samples=256,
            calc_second_order=True,
            seed=7,
            verbose=False,
        )
        X = jnp.asarray(sampling_result.samples)
        Y = 2.0 * X[:, 0] + 0.5 * X[:, 1] ** 2 + X[:, 0] * X[:, 2]

        result = jaxgsa.sobol.analyze(
            sampling_result,
            Y,
            num_resamples=20,
            conf_level=0.9,
            ci_method=ci_method,
            key=jax.random.key(123),
        )
        assert result.S1_conf is not None
        assert result.ST_conf is not None
        assert result.S2_conf is not None
        ds = result.to_dataset()

        assert "S1_lower" in ds
        assert "S1_upper" in ds
        assert "ST_lower" in ds
        assert "ST_upper" in ds
        assert "S2_lower" in ds
        assert "S2_upper" in ds

        assert list(ds.S1_lower.dims) == ["param"]
        assert list(ds.ST_upper.dims) == ["param"]
        assert list(ds.S2_lower.dims) == ["param_i", "param_j"]

        np.testing.assert_allclose(ds.S1.values, np.asarray(result.S1))
        np.testing.assert_allclose(ds.S1_lower.values, np.asarray(result.S1_conf[0]))
        np.testing.assert_allclose(ds.S1_upper.values, np.asarray(result.S1_conf[1]))
        np.testing.assert_allclose(ds.ST_lower.values, np.asarray(result.ST_conf[0]))
        np.testing.assert_allclose(ds.ST_upper.values, np.asarray(result.ST_conf[1]))
        np.testing.assert_allclose(
            ds.S2_lower.values,
            np.asarray(result.S2_conf[0]),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            ds.S2_upper.values,
            np.asarray(result.S2_conf[1]),
            equal_nan=True,
        )
        assert list(ds.coords["param"].values) == ["x1", "x2", "x3"]
        assert list(ds.coords["param_i"].values) == ["x1", "x2", "x3"]
        assert list(ds.coords["param_j"].values) == ["x1", "x2", "x3"]


# ── HDMRResult tests ────────────────────────────────────────────────────────


class TestHDMRResultToDataset:
    def test_basic(self, problem):
        """Term-indexed Sa/Sb/S + param-indexed ST."""
        terms = ("x1", "x2", "x3", "x1/x2")
        n_terms = len(terms)
        D = 3
        r = HDMRResult(
            Sa=jnp.ones(n_terms),
            Sb=jnp.zeros(n_terms),
            S=jnp.ones(n_terms),
            ST=jnp.ones(D),
            problem=problem,
            terms=terms,
            invalid=_CLEAN_INVALID_ROW,
        )
        ds = r.to_dataset()
        assert list(ds.Sa.dims) == ["term"]
        assert list(ds.ST.dims) == ["param"]
        assert list(ds.coords["term"].values) == list(terms)
        assert list(ds.coords["param"].values) == ["x1", "x2", "x3"]
