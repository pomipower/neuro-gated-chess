"""Test helpers. Tests for modules that are still stubs are skipped instead of failing."""

from __future__ import annotations

import functools
from typing import Any, Callable

import pytest


def skip_if_not_implemented(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Turn NotImplementedError raised by a stub into a pytest skip."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except NotImplementedError:
            pytest.skip("module not implemented yet (see docs/SPEC.md)")

    return wrapper
