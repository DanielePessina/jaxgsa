"""Basis-expansion sensitivity methods (HDMR, PCE).

Each submodule provides ``analyze()`` and ``emulate()`` entry points
that work with arbitrary (X, Y) pairs — no structured sampling required.
"""

from gsax.expansions import hdmr, pce

__all__ = ["hdmr", "pce"]
