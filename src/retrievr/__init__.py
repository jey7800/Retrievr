"""retrievr: find and download documents from websites."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0.dev0"
__all__ = ["HostSession", "Retrievr", "__version__"]

if TYPE_CHECKING:
    from retrievr.api import Retrievr
    from retrievr.fetch import HostSession


def __getattr__(name: str) -> Any:
    # Lazy, so that ``retrievr.config`` can import ``__version__`` without a cycle.
    if name == "Retrievr":
        from retrievr.api import Retrievr

        return Retrievr
    if name == "HostSession":
        from retrievr.fetch import HostSession

        return HostSession
    raise AttributeError(name)
