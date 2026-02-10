"""Grouping and ungrouping of array axes.

This module provides functions to merge or split array axes without changing the C-like memory layout.
"""

from __future__ import annotations

import math
import operator
import typing
from typing import TYPE_CHECKING, Any, SupportsIndex, TypeVar

import numpy as np
import numpy.typing as npt

from trg_utils import _index

if TYPE_CHECKING:
    from collections.abc import Sequence

_T = TypeVar("_T", bound=np.generic)


@typing.overload
def group(arr: npt.NDArray[_T], inds: Sequence[SupportsIndex | Sequence[SupportsIndex]]) -> npt.NDArray[_T]: ...


@typing.overload
def group(arr: npt.ArrayLike, inds: Sequence[SupportsIndex | Sequence[SupportsIndex]]) -> npt.NDArray[Any]: ...


def group(arr: npt.ArrayLike, inds: Sequence[SupportsIndex | Sequence[SupportsIndex]]) -> npt.NDArray[Any]:
    """Reorder and merge axes according to ``inds``.

    Parameters
    ----------
    arr
        Input array.
    inds
        Output-axis specification. Each element is either:
        - a single axis index, copied as-is, or
        - a sequence of axis indices, merged into one axis.
        The specification must cover all input axes exactly once.

    Returns
    -------
    numpy.ndarray
        Array with axes permuted and optionally merged as specified by ``inds``.

    Raises
    ------
    ValueError
        - If an axis index is out of range or duplicated.
        - If an index group is empty.
        - If the specification does not cover all input axes.

    Notes
    -----
    C-like memory layout is preserved on merging for each group.
    """  # noqa: DOC502
    arr = np.asarray(arr)
    d = arr.ndim
    args = [_index.normalize(d, _index.materialize(ind)) for ind in inds]
    _index.assert_span(d, *args)
    s_prev: tuple[int, ...] = arr.shape
    # Compute before reshape
    shapes = (math.prod(s_prev[i] for i in ind) if isinstance(ind, tuple) else s_prev[ind] for ind in args)
    return arr.transpose(*_index.flatten(args)).reshape(*shapes)


def _ungroup_impl(arr: npt.NDArray[_T], target: int, split: tuple[int, ...]) -> npt.NDArray[_T]:
    assert 0 <= target < arr.ndim
    nl = int(arr.shape[target])
    if nl != math.prod(split):
        msg = f"Cannot ungroup: {nl} -> {split}."
        raise ValueError(msg)
    return arr.reshape(*arr.shape[:target], *split, *arr.shape[target + 1 :])


@typing.overload
def ungroup(arr: npt.NDArray[_T], *instr: tuple[SupportsIndex, Sequence[SupportsIndex]]) -> npt.NDArray[_T]: ...


@typing.overload
def ungroup(arr: npt.ArrayLike, *instr: tuple[SupportsIndex, Sequence[SupportsIndex]]) -> npt.NDArray[Any]: ...


def ungroup(arr: npt.ArrayLike, *instr: tuple[SupportsIndex, Sequence[SupportsIndex]]) -> npt.NDArray[Any]:
    """Split one or more axes according to ``instr``.

    Parameters
    ----------
    arr
        Input array.
    *instr
        One or more ``(target, split)`` pairs.
        ``target`` is the axis to split, and ``split`` is the replacement shape for that axis.

    Returns
    -------
    numpy.ndarray
        Array with each target axis replaced by the corresponding ``split`` shape.

    Raises
    ------
    ValueError
        - If a target axis is out of range or duplicated.
        - If a target shape is incompatible with ``arr``.

    Notes
    -----
    All the instructions are applied in parallel to the original array.
    """
    arr = np.asarray(arr)
    d = arr.ndim
    args = [(_index.normalize(d, _index.materialize(l)), _index.materialize(r)) for l, r in instr]
    # Reverse sort by index
    args.sort(key=operator.itemgetter(0), reverse=True)
    known: set[int] = set()
    for i, split in args:
        if i in known:
            msg = "ops must not overlap."
            raise ValueError(msg)
        known.add(i)
        arr = _ungroup_impl(arr, i, split)
    return arr
