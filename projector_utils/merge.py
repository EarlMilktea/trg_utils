"""Utilities to group and ungroup tensor axes.

This module provides helpers to merge the last ``k`` axes into one or to split
the last axis into a target shape.
"""

from __future__ import annotations

import math
import operator
from typing import TYPE_CHECKING, Any, SupportsIndex

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Sequence


def group(arr: npt.ArrayLike, k: int) -> npt.NDArray[Any]:
    """Merge the last ``k`` axes into one.

    Parameters
    ----------
    arr
        Input array.
    k
        Number of trailing axes to merge.

    Returns
    -------
    numpy.ndarray
        Array with the last ``k`` axes flattened into a single axis.

    Raises
    ------
    ValueError
        If ``k`` is not between 1 and ``arr.ndim``.

    Notes
    -----
    The merged axis preserves C-order (row-major) element ordering of the
    original trailing axes.
    """
    arr = np.ascontiguousarray(arr)
    if not (0 < k <= arr.ndim):
        msg = "k must be between 1 and arr.ndim."
        raise ValueError(msg)
    return arr.reshape(*arr.shape[: arr.ndim - k], -1)


def ungroup(arr: npt.ArrayLike, split: Sequence[SupportsIndex]) -> npt.NDArray[Any]:
    """Split the last axis into the given shape.

    Parameters
    ----------
    arr
        Input array.
    split
        Target shape for the last axis.

    Returns
    -------
    numpy.ndarray
        Array with the last axis reshaped to ``split``.

    Raises
    ------
    ValueError
        If the last axis size does not match ``math.prod(split)``.

    Notes
    -----
    The split uses C-order (row-major) when expanding the last axis into
    ``split``.
    """
    arr = np.ascontiguousarray(arr)
    split = tuple(operator.index(i) for i in split)
    nl = int(arr.shape[-1])
    if nl != math.prod(split):  # type: ignore[arg-type]
        msg = f"Cannot ungroup: {nl} -> {split}."
        raise ValueError(msg)
    return arr.reshape(*arr.shape[:-1], *split)
