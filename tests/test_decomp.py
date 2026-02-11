from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from trg_utils import decomp


def _perm(arr: npt.NDArray[Any], i0: tuple[int, ...], i1: tuple[int, ...]) -> list[int]:
    d = arr.ndim
    iperm = [i + d if i < 0 else i for i in itertools.chain(i0, i1)]
    return [iperm.index(i) for i in range(len(iperm))]


class TestTSVD:
    @pytest.mark.parametrize(
        ("iu", "iv"),
        [
            ((0, 3), (1, 2)),
            ((-4,), (0, 1, 2)),
            ((0, 1), (2, 3)),
        ],
    )
    def test_ng_oob(self, iu: tuple[int, ...], iv: tuple[int, ...]) -> None:
        with pytest.raises(ValueError, match=r"out of range"):
            decomp.tsvd(np.zeros((2, 3, 4)), iu, iv)

    @pytest.mark.parametrize(
        ("iu", "iv"),
        [
            ((0, 1), (1, 2)),
            ((0,), (1,)),
            ((0, 1, 1), (2,)),
        ],
    )
    def test_ng_overlap(self, iu: tuple[int, ...], iv: tuple[int, ...]) -> None:
        with pytest.raises(ValueError, match=r"without overlap"):
            decomp.tsvd(np.zeros((2, 3, 4)), iu, iv)

    def test_ng_empty(self) -> None:
        with pytest.raises(ValueError, match=r"must not be empty"):
            decomp.tsvd(np.zeros((2, 3, 4)), (), (0, 1, 2))

        with pytest.raises(ValueError, match=r"must not be empty"):
            decomp.tsvd(np.zeros((2, 3, 4)), (0, 1, 2), ())

    @pytest.mark.parametrize(
        ("iu", "iv"),
        [
            ((0, 1), (2, 3, 4)),
            ((1, -1, 2), (3, 0)),
            ((2,), (1, 3, 0, 4)),
            ((-2, 0, 1, -1), (2,)),
        ],
    )
    def test_tsvd(self, rng: np.random.Generator, iu: tuple[int, ...], iv: tuple[int, ...]) -> None:
        arr = rng.normal(size=(1, 2, 3, 4, 5))
        u, s, v = decomp.tsvd(arr, iu, iv)
        us = np.tensordot(u, np.diag(s), axes=(-1, -1))
        usv = np.tensordot(us, v, axes=(-1, -1)).transpose(*_perm(arr, iu, iv))
        np.testing.assert_allclose(arr, usv)


class TestTQR:
    @pytest.mark.parametrize(
        ("iq", "ir"),
        [
            ((0, 3), (1, 2)),
            ((-4,), (0, 1, 2)),
            ((0, 1), (2, 3)),
        ],
    )
    def test_ng_oob(self, iq: tuple[int, ...], ir: tuple[int, ...]) -> None:
        with pytest.raises(ValueError, match=r"out of range"):
            decomp.tqr(np.zeros((2, 3, 4)), iq, ir)

    @pytest.mark.parametrize(
        ("iq", "ir"),
        [
            ((0, 1), (1, 2)),
            ((0,), (1,)),
            ((0, 1, 1), (2,)),
        ],
    )
    def test_ng_overlap(self, iq: tuple[int, ...], ir: tuple[int, ...]) -> None:
        with pytest.raises(ValueError, match=r"without overlap"):
            decomp.tqr(np.zeros((2, 3, 4)), iq, ir)

    def test_ng_empty(self) -> None:
        with pytest.raises(ValueError, match=r"must not be empty"):
            decomp.tqr(np.zeros((2, 3, 4)), (), (0, 1, 2))

        with pytest.raises(ValueError, match=r"must not be empty"):
            decomp.tqr(np.zeros((2, 3, 4)), (0, 1, 2), ())

    @pytest.mark.parametrize(
        ("iq", "ir"),
        [
            ((0, 1), (2, 3, 4)),
            ((1, -1, 2), (3, 0)),
            ((2, 0), (1, 3, 4)),
            ((-2, 0, 1), (-1, 2)),
        ],
    )
    def test_tqr(self, rng: np.random.Generator, iq: tuple[int, ...], ir: tuple[int, ...]) -> None:
        arr = rng.normal(size=(1, 2, 3, 4, 5))
        q, r = decomp.tqr(arr, iq, ir)
        qr = np.tensordot(q, r, axes=(-1, -1)).transpose(*_perm(arr, iq, ir))
        np.testing.assert_allclose(arr, qr)
