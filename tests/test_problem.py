import numpy as np
import pytest

from jaxgsa.problem import (
    GaussianInputSpec,
    Problem,
    UniformInputSpec,
    _normalized_input_to_dict,
)


def test_from_dict_tuple_shorthand_still_produces_uniform_bounds():
    p = Problem.from_dict({"x1": (0.0, 1.0), "x2": (-1.0, 1.0)})
    assert p.names == ("x1", "x2")
    assert p.bounds == ((0.0, 1.0), (-1.0, 1.0))
    assert p.num_vars == 2
    assert p.has_non_uniform_inputs is False


def test_from_dict_accepts_uniform_and_gaussian_typed_dict_specs():
    p = Problem.from_dict(
        {
            "x1": UniformInputSpec(dist="uniform", low=-2.0, high=3.0),
            "x2": GaussianInputSpec(dist="gaussian", mean=1.5, variance=2.25),
            "x3": GaussianInputSpec(
                dist="gaussian",
                mean=0.0,
                variance=1.0,
                low=-1.0,
                high=2.0,
            ),
        }
    )

    assert p.names == ("x1", "x2", "x3")
    assert p.bounds is None
    assert p.num_vars == 3
    assert p.has_non_uniform_inputs is True


@pytest.mark.parametrize(
    "spec",
    [
        GaussianInputSpec(dist="gaussian", mean=0.0, variance=1.0, low=-1.0),
        GaussianInputSpec(dist="gaussian", mean=0.0, variance=1.0, high=1.0),
        GaussianInputSpec(dist="gaussian", mean=0.0, variance=1.0, low=-1.0, high=1.0),
    ],
)
def test_from_dict_accepts_one_or_two_sided_gaussian_truncation(spec):
    p = Problem.from_dict({"x": spec})
    assert p.bounds is None
    assert p.has_non_uniform_inputs is True


@pytest.mark.parametrize(
    ("params", "match"),
    [
        (
            {"x": UniformInputSpec(dist="uniform", low=1.0, high=1.0)},
            "low < high",
        ),
        (
            {"x": GaussianInputSpec(dist="gaussian", mean=0.0, variance=0.0)},
            "variance > 0",
        ),
        (
            {
                "x": GaussianInputSpec(
                    dist="gaussian",
                    mean=0.0,
                    variance=1.0,
                    low=2.0,
                    high=1.0,
                )
            },
            "low < high",
        ),
        (
            {"x": {"dist": "beta", "alpha": 1.0, "beta": 2.0}},
            "Unsupported input distribution",
        ),
    ],
)
def test_invalid_input_specs_raise_value_error(params, match):
    with pytest.raises(ValueError, match=match):
        Problem.from_dict(params)


def test_direct_constructor_remains_uniform_only():
    p = Problem(names=("a",), bounds=((0.0, 1.0),))
    assert p.bounds == ((0.0, 1.0),)
    assert p.has_non_uniform_inputs is False


def test_frozen():
    p = Problem(names=("a",), bounds=((0.0, 1.0),))
    with pytest.raises(AttributeError):
        setattr(p, "names", ("b",))


class TestTruncateGaussians:
    """``Problem.from_dict(..., truncate_gaussians=q)`` bounds every open side."""

    SPECS = {
        "u": (0.0, 1.0),
        "g": {"dist": "gaussian", "mean": 2.0, "variance": 9.0},
        "half": {"dist": "gaussian", "mean": 0.0, "variance": 1.0, "low": -1.0},
        "both": {"dist": "gaussian", "mean": 0.0, "variance": 1.0, "low": -1.0, "high": 1.0},
    }

    def test_default_leaves_gaussians_unbounded(self):
        """Tier T4 (internal consistency): no default truncation is applied.

        An unbounded Gaussian carries neither a ``low`` nor a ``high`` key.
        """
        p = Problem.from_dict(dict(self.SPECS))
        spec = _normalized_input_to_dict(p.input_specs[1])
        assert "low" not in spec
        assert "high" not in spec

    def test_fills_only_open_sides(self):
        """Tier T2 (external library): truncation points match the frozen Gaussian quantiles.

        Provenance:

        * Tier: T2.
        * Oracle: ``scipy.stats.norm.ppf``.
        * Version: scipy 1.18.0.
        * Run on: 2026-08-18.
        * Script: none needed. The three values are
          ``float(norm.ppf(1e-3, loc=2.0, scale=3.0))``,
          ``float(norm.ppf(1 - 1e-3, loc=2.0, scale=3.0))`` and
          ``float(norm.ppf(1 - 1e-3))``, in that order.

        The values are literals and not a live call on purpose.
        ``problem._fill_gaussian_bounds`` calls ``norm.ppf`` itself, so a test
        that called ``norm.ppf`` too would compare scipy against scipy and pass
        for any ``q``, any ``loc`` and any ``scale`` the code chose to pass.
        Frozen numbers make the test fail if the fill ever uses the wrong tail,
        the wrong side, or the wrong location and scale.
        """
        p = Problem.from_dict(dict(self.SPECS), truncate_gaussians=1e-3)
        # uniform untouched
        assert _normalized_input_to_dict(p.input_specs[0]) == {
            "dist": "uniform",
            "low": 0.0,
            "high": 1.0,
        }
        spec = _normalized_input_to_dict(p.input_specs[1])
        assert spec["low"] == pytest.approx(-7.27069691850344)
        assert spec["high"] == pytest.approx(11.27069691850344)
        # A declared side wins; only the open side is filled.
        spec = _normalized_input_to_dict(p.input_specs[2])
        assert spec["low"] == -1.0
        assert spec["high"] == pytest.approx(3.090232306167813)
        assert _normalized_input_to_dict(p.input_specs[3]) == {
            "dist": "gaussian",
            "mean": 0.0,
            "variance": 1.0,
            "low": -1.0,
            "high": 1.0,
        }

    @pytest.mark.parametrize("bad_q", [0.0, 0.5, -0.1, 1.0])
    def test_invalid_q_raises(self, bad_q):
        with pytest.raises(ValueError, match="truncate_gaussians"):
            Problem.from_dict(dict(self.SPECS), truncate_gaussians=bad_q)

    def test_morris_does_not_squash_a_problem_bounded_this_way(self):
        """Tier T4 (internal consistency): a spec bounded by ``truncate_gaussians`` is
        genuinely bounded (A1).

        The Morris design spans the full truncation interval, so the sampler reads
        the same bounds the problem declares.
        """
        import numpy as np

        from jaxgsa import morris

        p = Problem.from_dict(
            {"g": {"dist": "gaussian", "mean": 0.0, "variance": 1.0}}, truncate_gaussians=1e-3
        )
        spec = _normalized_input_to_dict(p.input_specs[0])
        sr = morris.sample(p, n_trajectories=25, seed=1, verbose=False)
        assert float(np.min(sr.samples[:, 0])) == pytest.approx(spec["low"])
        assert float(np.max(sr.samples[:, 0])) == pytest.approx(spec["high"])


# ---------------------------------------------------------------------------
# Problem.correlation
# ---------------------------------------------------------------------------


def test_correlation_defaults_to_none_and_independent():
    p = Problem.from_dict({"x1": (0.0, 1.0), "x2": (0.0, 1.0)})
    assert p.correlation is None
    assert p.has_correlated_inputs is False


def test_from_dict_accepts_latent_correlation():
    R = [[1.0, 0.5], [0.5, 1.0]]
    p = Problem.from_dict({"x1": (0.0, 1.0), "x2": (0.0, 1.0)}, correlation=R)
    assert p.has_correlated_inputs is True
    stored = p.correlation
    assert stored is not None
    np.testing.assert_allclose(stored, R, atol=1e-15)


def test_direct_constructor_accepts_correlation():
    p = Problem(
        names=("a", "b"),
        bounds=((0.0, 1.0), (0.0, 1.0)),
        correlation=[[1.0, -0.3], [-0.3, 1.0]],
    )
    stored = p.correlation
    assert stored is not None
    np.testing.assert_allclose(stored[0, 1], -0.3, atol=1e-15)


def test_spearman_kind_is_converted_to_latent_on_entry():
    rho_s = 0.7
    p = Problem.from_dict(
        {"x1": (0.0, 1.0), "x2": (0.0, 1.0)},
        correlation=[[1.0, rho_s], [rho_s, 1.0]],
        correlation_kind="spearman",
    )
    stored = p.correlation
    assert stored is not None
    np.testing.assert_allclose(stored[0, 1], 2.0 * np.sin(np.pi * rho_s / 6.0), atol=1e-15)


def test_correlation_shape_must_match_names():
    with pytest.raises(ValueError, match=r"must be \(2, 2\)"):
        Problem.from_dict({"x1": (0.0, 1.0), "x2": (0.0, 1.0)}, correlation=np.eye(3))


def test_mildly_indefinite_correlation_warns_at_construction():
    # Smallest eigenvalue -0.01. The repair moves an entry by 5e-3, under the
    # 0.05 threshold that makes a declared matrix an error.
    mild = np.full((3, 3), -0.505)
    np.fill_diagonal(mild, 1.0)
    with pytest.warns(UserWarning, match="not positive definite"):
        p = Problem.from_dict(
            {"a": (0.0, 1.0), "b": (0.0, 1.0), "c": (0.0, 1.0)}, correlation=mild
        )
    stored = p.correlation
    assert stored is not None
    assert np.linalg.eigvalsh(stored).min() > 0


def test_materially_indefinite_correlation_is_rejected_at_construction():
    indefinite = np.array(
        [
            [1.0, 0.9, 0.9],
            [0.9, 1.0, -0.9],
            [0.9, -0.9, 1.0],
        ]
    )
    with pytest.raises(ValueError, match="too far to accept"):
        Problem.from_dict(
            {"a": (0.0, 1.0), "b": (0.0, 1.0), "c": (0.0, 1.0)}, correlation=indefinite
        )


def test_identity_correlation_is_stored_but_not_correlated():
    p = Problem.from_dict({"x1": (0.0, 1.0), "x2": (0.0, 1.0)}, correlation=np.eye(2))
    assert p.correlation is not None
    assert p.has_correlated_inputs is False


def test_with_correlation_returns_new_problem_and_preserves_marginals():
    base = Problem.from_dict(
        {
            "u": (0.0, 2.0),
            "g": GaussianInputSpec(dist="gaussian", mean=1.0, variance=4.0),
        },
        output_names=("response",),
    )
    R = np.array([[1.0, 0.6], [0.6, 1.0]])
    correlated = base.with_correlation(R)

    assert correlated is not base
    assert base.correlation is None  # the original is untouched
    assert correlated.names == base.names
    assert correlated.input_specs == base.input_specs
    assert correlated.output_names == base.output_names
    stored = correlated.correlation
    assert stored is not None
    np.testing.assert_allclose(stored, R, atol=1e-15)


def test_with_correlation_accepts_spearman_kind():
    base = Problem.from_dict({"x1": (0.0, 1.0), "x2": (0.0, 1.0)})
    rho_s = 0.5
    p = base.with_correlation([[1.0, rho_s], [rho_s, 1.0]], kind="spearman")
    stored = p.correlation
    assert stored is not None
    np.testing.assert_allclose(stored[0, 1], 2.0 * np.sin(np.pi * rho_s / 6.0), atol=1e-15)


def test_with_correlation_none_drops_the_matrix():
    R = [[1.0, 0.5], [0.5, 1.0]]
    p = Problem.from_dict({"x1": (0.0, 1.0), "x2": (0.0, 1.0)}, correlation=R)
    dropped = p.with_correlation(None)
    assert dropped.correlation is None
    assert dropped.has_correlated_inputs is False


def test_correlation_property_returns_defensive_copy():
    R = [[1.0, 0.5], [0.5, 1.0]]
    p = Problem.from_dict({"x1": (0.0, 1.0), "x2": (0.0, 1.0)}, correlation=R)
    first = p.correlation
    assert first is not None
    first[0, 1] = 0.0
    second = p.correlation
    assert second is not None
    assert second[0, 1] == 0.5


def test_correlation_participates_in_equality_and_hash():
    params = {"x1": (0.0, 1.0), "x2": (0.0, 1.0)}
    plain = Problem.from_dict(dict(params))
    correlated = Problem.from_dict(dict(params), correlation=[[1.0, 0.5], [0.5, 1.0]])
    correlated_again = Problem.from_dict(dict(params), correlation=[[1.0, 0.5], [0.5, 1.0]])

    assert plain != correlated
    assert correlated == correlated_again
    assert hash(correlated) == hash(correlated_again)


def test_correlated_problem_is_still_frozen():
    p = Problem.from_dict({"x1": (0.0, 1.0), "x2": (0.0, 1.0)}, correlation=np.eye(2))
    with pytest.raises(AttributeError):
        setattr(p, "_correlation", None)
