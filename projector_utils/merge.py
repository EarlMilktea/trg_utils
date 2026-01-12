"""Group and ungroup array legs."""

from __future__ import annotations

import math
import operator
from typing import TYPE_CHECKING, Any, SupportsIndex

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Sequence


def group(arr: npt.ArrayLike, k: int) -> npt.NDArray[Any]:
    """Merge last k legs.

    This does the opposite of `ungroup`.
    """
    arr = np.ascontiguousarray(arr)
    if not (0 < k <= arr.ndim):
        msg = "k must be between 1 and arr.ndim."
        raise ValueError(msg)
    return arr.reshape(*arr.shape[: arr.ndim - k], -1)


def ungroup(arr: npt.ArrayLike, split: Sequence[SupportsIndex]) -> npt.NDArray[Any]:
    """Split last leg to have shape `split`.

    This does the opposite of `group`.
    """
    arr = np.ascontiguousarray(arr)
    split = tuple(operator.index(i) for i in split)
    nl = int(arr.shape[-1])
    if nl != math.prod(split):
        msg = f"Cannot ungroup: {nl} -> {split}."
        raise ValueError(msg)
    return arr.reshape(*arr.shape[:-1], *split)
