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

    def test_herm_ok(self, rng: np.random.Generator) -> None:
        arr = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        arr += arr.T.conj()
        arr = arr.reshape(2, 2, 2, 2)
        u, s, v = decomp.tsvd(arr, (0, 1), (2, 3), hermitian=True)
        usv = np.einsum("abi,i,cdi->abcd", u, s, v)
        np.testing.assert_allclose(arr, usv)

    def test_herm_ng(self, rng: np.random.Generator) -> None:
        arr = rng.normal(size=(2, 2, 2, 2)) + 1j * rng.normal(size=(2, 2, 2, 2))
        with pytest.warns(UserWarning, match=r"not likely to be Hermitian"):
            decomp.tsvd(arr, (0, 1), (2, 3), hermitian=True)


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


class TestHOSVD:
    @pytest.mark.parametrize(
        "iu",
        [
            (0, 3),
            (-4,),
        ],
    )
    def test_ng_oob(self, iu: tuple[int, ...]) -> None:
        with pytest.raises(ValueError, match=r"out of range"):
            decomp.hosvd(np.zeros((2, 3, 4)), iu)

    def test_ng_overlap(self) -> None:
        with pytest.raises(ValueError, match=r"must be unique"):
            decomp.hosvd(np.zeros((2, 3, 4)), (0, 1, 1))

    def test_ng_empty(self) -> None:
        with pytest.raises(ValueError, match=r"must not be empty"):
            decomp.hosvd(np.zeros((2, 3, 4)), ())

        with pytest.raises(ValueError, match=r"must be excluded"):
            decomp.hosvd(np.zeros((2, 3, 4)), (2, 1, 0))

    @pytest.mark.parametrize(
        ("iu", "iv"),
        [
            ((0, 1), (2, 3, 4)),
            ((1, -1, 2), (3, 0)),
            ((2,), (1, 3, 0, 4)),
            ((-2, 0, 1, -1), (2,)),
        ],
    )
    def test_hosvd(self, rng: np.random.Generator, iu: tuple[int, ...], iv: tuple[int, ...]) -> None:
        arr = rng.normal(size=(1, 2, 3, 4, 5))
        s, u = decomp.hosvd(arr, iu)
        u_, s_, _ = decomp.tsvd(arr, iu, iv)
        n = min(s.size, s_.size)
        np.testing.assert_allclose(s[:n], s_[:n])
        for i in range(n):
            vi = u[..., i]
            vi *= np.sign(vi.sum())
            vi_ = u_[..., i]
            vi_ *= np.sign(vi_.sum())
            np.testing.assert_allclose(vi, vi_)
