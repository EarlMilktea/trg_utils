from __future__ import annotations

import operator
import typing
from collections.abc import Iterable, Iterator, Sequence
from typing import SupportsIndex, TypeVar


@typing.overload
def normalize(d: int, ind: Sequence[SupportsIndex]) -> tuple[int, ...]: ...


@typing.overload
def normalize(d: int, ind: SupportsIndex) -> int: ...


def normalize(d: int, ind: SupportsIndex | Sequence[SupportsIndex]) -> int | tuple[int, ...]:
    if isinstance(ind, Sequence):
        return tuple(normalize(d, i) for i in ind)

    i = operator.index(ind)
    if i < 0:
        i += d
    if not (0 <= i < d):
        msg = f"Index {ind} is out of range."
        raise ValueError(msg)
    return i


_T = TypeVar("_T")


def flatten(it: Iterable[_T | Iterable[_T]]) -> Iterator[_T]:
    for x in it:
        if isinstance(x, Iterable):
            yield from flatten(x)
        else:
            yield x


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
