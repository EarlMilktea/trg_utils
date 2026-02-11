from __future__ import annotations

import operator
import typing
from collections.abc import Iterable, Iterator, Sequence
from typing import SupportsIndex, TypeVar


@typing.overload
def materialize(ind: Sequence[SupportsIndex]) -> tuple[int, ...]: ...


@typing.overload
def materialize(ind: SupportsIndex) -> int: ...


def materialize(ind: SupportsIndex | Iterable[SupportsIndex]) -> int | tuple[int, ...]:
    if isinstance(ind, Iterable):
        return tuple(materialize(i) for i in ind)
    return operator.index(ind)


@typing.overload
def normalize(d: int, ind: tuple[int, ...]) -> tuple[int, ...]: ...


@typing.overload
def normalize(d: int, ind: int) -> int: ...


def normalize(d: int, ind: int | tuple[int, ...]) -> int | tuple[int, ...]:
    if isinstance(ind, tuple):
        return tuple(normalize(d, i) for i in ind)

    if ind < 0:
        ind += d
    if not (0 <= ind < d):
        msg = f"Index {ind} is out of range."
        raise ValueError(msg)
    return ind


_T = TypeVar("_T")


def flatten(it: Iterable[_T | Iterable[_T]]) -> Iterator[_T]:
    for x in it:
        if isinstance(x, Iterable):
            yield from flatten(x)
        else:
            yield x


def assert_allunique(*inds: int | tuple[int, ...]) -> None:
    work = list(flatten(inds))
    if len(work) != len(set(work)):
        msg = "Indices must be unique."
        raise ValueError(msg)


def assert_span(d: int, *inds: int | tuple[int, ...]) -> None:
    if any(ind == () for ind in inds):
        msg = "Each index must not be empty."
        raise ValueError(msg)
    ref = list(range(d))
    work = list(flatten(inds))
    work.sort()
    if work != ref:
        msg = "Indices must cover all axes without overlap."
        raise ValueError(msg)
