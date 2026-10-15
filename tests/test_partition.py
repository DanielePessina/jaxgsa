"""Tests for the equal-frequency class layout in ``jaxgsa._core.partition``."""

from __future__ import annotations

import numpy as np

from jaxgsa._core.partition import _class_layout


def test_n60_m22_uses_exact_integer_edges():
    """``N=60, M=22``: edge 11 is ``11 * 60 // 22 = 30``, not 29.

    ``floor(linspace(0, 60, 23))`` gives 29 there, because the float edge
    rounds just below 30. That moved one row between classes 10 and 11.
    """
    _, sizes = _class_layout(60, 22)
    edges = (np.arange(23) * 60) // 22
    np.testing.assert_array_equal(sizes, np.diff(edges))
    assert np.cumsum(sizes)[10] == 30


def test_sizes_are_an_exact_equal_frequency_split():
    """Edges are ``(m * N) // M``; sizes sum to N and differ by at most one."""
    for N in range(1, 301):
        for M in range(1, min(N, 64) + 1):
            take, sizes = _class_layout(N, M)
            edges = (np.arange(M + 1) * N) // M
            np.testing.assert_array_equal(np.concatenate([[0], np.cumsum(sizes)]), edges)
            assert sizes.sum() == N
            assert sizes.max() - sizes.min() <= 1
            assert take.shape == (M, sizes.max())
            np.testing.assert_array_equal(take[:, 0], edges[:-1])
