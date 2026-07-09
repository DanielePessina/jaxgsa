import pytest

from gsax.problem import GaussianInputSpec, Problem, UniformInputSpec


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
