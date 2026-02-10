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

if TYPE_CHECKING:
    from collections.abc import Sequence

_T = TypeVar("_T", bound=np.generic)


@typing.overload
def _group_impl(arr: npt.NDArray[_T], begin: SupportsIndex, end: SupportsIndex) -> npt.NDArray[_T]: ...


@typing.overload
def _group_impl(arr: npt.ArrayLike, begin: SupportsIndex, end: SupportsIndex) -> npt.NDArray[Any]: ...


def _group_impl(arr: npt.ArrayLike, begin: SupportsIndex, end: SupportsIndex) -> npt.NDArray[Any]:
    arr = np.asarray(arr)
    begin = operator.index(begin)
    end = operator.index(end)
    if begin < 0:
        begin += arr.ndim
    if end < 0:
        end += arr.ndim
    if not (0 <= begin < end <= arr.ndim):
        msg = "begin and end must satisfy 0 <= begin < end <= arr.ndim after normalization."
        raise ValueError(msg)
    return arr.reshape(*arr.shape[:begin], -1, *arr.shape[end:])


@typing.overload
def _ungroup_impl(arr: npt.NDArray[_T], target: SupportsIndex, split: Sequence[SupportsIndex]) -> npt.NDArray[_T]: ...


@typing.overload
def _ungroup_impl(arr: npt.ArrayLike, target: SupportsIndex, split: Sequence[SupportsIndex]) -> npt.NDArray[Any]: ...


def _ungroup_impl(arr: npt.ArrayLike, target: SupportsIndex, split: Sequence[SupportsIndex]) -> npt.NDArray[Any]:
    arr = np.asarray(arr)
    target = operator.index(target)
    if target < 0:
        target += arr.ndim
    if not (0 <= target < arr.ndim):
        msg = "target must be between 0 and arr.ndim - 1 after normalization."
        raise ValueError(msg)
    split = tuple(operator.index(i) for i in split)
    nl = int(arr.shape[target])
    if nl != math.prod(split):  # type: ignore[arg-type]
        msg = f"Cannot ungroup: {nl} -> {split}."
        raise ValueError(msg)
    return arr.reshape(*arr.shape[:target], *split, *arr.shape[target + 1 :])
