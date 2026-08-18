"""The warning class that every jaxgsa warning uses."""

from __future__ import annotations

__all__ = ["JaxgsaWarning"]


class JaxgsaWarning(UserWarning):
    """The category of every warning that jaxgsa raises.

    Each ``warnings.warn`` call in jaxgsa passes this class. You can therefore
    silence or promote jaxgsa warnings alone, without touching the warnings
    that NumPy, SciPy or JAX raise.

    The class is a subclass of :class:`UserWarning`. Code that already filters
    on ``UserWarning`` keeps working.

    Example:
        Turn every jaxgsa warning into an error::

            import warnings
            from jaxgsa import JaxgsaWarning

            warnings.simplefilter("error", JaxgsaWarning)
    """
