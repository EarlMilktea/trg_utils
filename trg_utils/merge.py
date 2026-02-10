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
    arr = np.asarray(arr)
    d = arr.ndim
    args = [_index.normalize(d, ind) for ind in inds]
    _index.assert_span(d, *args)
    arr = arr.transpose(*_index.flatten(args))
    shapes = (math.prod(arr.shape[i] for i in ind) if isinstance(ind, tuple) else int(arr.shape[ind]) for ind in args)
    return arr.reshape(*shapes)


def _ungroup_impl(arr: npt.NDArray[_T], target: int, split: tuple[int, ...]) -> npt.NDArray[_T]:
    if not (0 <= target < arr.ndim):
        msg = "Index out of range after normalization."
        raise ValueError(msg)
    nl = int(arr.shape[target])
    if nl != math.prod(split):
        msg = f"Cannot ungroup: {nl} -> {split}."
        raise ValueError(msg)
    return arr.reshape(*arr.shape[:target], *split, *arr.shape[target + 1 :])


@typing.overload
def ungroup(arr: npt.NDArray[_T], *ops: tuple[SupportsIndex, Sequence[SupportsIndex]]) -> npt.NDArray[_T]: ...


@typing.overload
def ungroup(arr: npt.ArrayLike, *ops: tuple[SupportsIndex, Sequence[SupportsIndex]]) -> npt.NDArray[Any]: ...


def ungroup(arr: npt.ArrayLike, *ops: tuple[SupportsIndex, Sequence[SupportsIndex]]) -> npt.NDArray[Any]:
    arr = np.asarray(arr)
    d = arr.ndim
    args = [(_index.normalize(d, l), _index.normalize(d, r)) for l, r in ops]
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
