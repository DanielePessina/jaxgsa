import pytest

from jaxgsa.problem import GaussianInputSpec, Problem, UniformInputSpec


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
        p = Problem.from_dict(dict(self.SPECS))
        assert p.input_specs[1][3] is None
        assert p.input_specs[1][4] is None

    def test_fills_only_open_sides(self):
        from scipy.stats import norm

        p = Problem.from_dict(dict(self.SPECS), truncate_gaussians=1e-3)
        assert p.input_specs[0] == ("uniform", 0.0, 1.0, None, None)  # uniform untouched
        _, _, _, lo, hi = p.input_specs[1]
        assert lo == pytest.approx(float(norm.ppf(1e-3, loc=2.0, scale=3.0)))
        assert hi == pytest.approx(float(norm.ppf(1.0 - 1e-3, loc=2.0, scale=3.0)))
        # A declared side wins; only the open side is filled.
        _, _, _, lo, hi = p.input_specs[2]
        assert lo == -1.0
        assert hi == pytest.approx(float(norm.ppf(1.0 - 1e-3)))
        assert p.input_specs[3] == ("gaussian", 0.0, 1.0, -1.0, 1.0)

    @pytest.mark.parametrize("bad_q", [0.0, 0.5, -0.1, 1.0])
    def test_invalid_q_raises(self, bad_q):
        with pytest.raises(ValueError, match="truncate_gaussians"):
            Problem.from_dict(dict(self.SPECS), truncate_gaussians=bad_q)

    def test_morris_does_not_squash_a_problem_bounded_this_way(self):
        """A spec bounded by ``truncate_gaussians`` is genuinely bounded (A1)."""
        import numpy as np

        from jaxgsa import morris

        p = Problem.from_dict(
            {"g": {"dist": "gaussian", "mean": 0.0, "variance": 1.0}}, truncate_gaussians=1e-3
        )
        sr = morris.sample(p, n_trajectories=25, seed=1, verbose=False)
        assert float(np.min(sr.samples[:, 0])) == pytest.approx(p.input_specs[0][3])
        assert float(np.max(sr.samples[:, 0])) == pytest.approx(p.input_specs[0][4])
