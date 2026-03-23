from __future__ import annotations

import math
from typing import Any, Literal

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given
from hypothesis.strategies import DrawFn

import trg_utils
from tests import conftest
from trg_utils import projector


def pdot(lhs: npt.NDArray[Any], rhs: npt.NDArray[Any]) -> npt.NDArray[Any]:
    return np.tensordot(lhs, rhs, (range(lhs.ndim - 1), range(rhs.ndim - 1)))


class TestExtend:
    def test_extend_ng(self) -> None:
        with pytest.raises(ValueError, match=r"Inconsistent"):
            projector.extend(np.zeros((9, 1)), np.zeros((9, 2)))

        with pytest.raises(ValueError, match=r"two legs"):
            projector.extend(np.zeros((9,)), np.zeros((9,)))

        with pytest.raises(ValueError, match=r"empty"):
            projector.extend(np.zeros((9, 0)), np.zeros((9, 0)))

        with pytest.raises(ValueError, match=r"smaller"):
            projector.extend(np.zeros((2, 2, 9)), np.zeros((2, 2, 9)))

    @staticmethod
    @st.composite
    def _pq_2d(draw: DrawFn) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
        (d,) = draw(hnp.array_shapes(max_dims=1))
        x = draw(conftest.almost_diagonal(d))
        r = draw(st.integers(1, d))
        ix = np.linalg.inv(x)
        p = x[:, :r]
        q = (ix.T.conj())[:, :r].astype(np.complex128, copy=False)
        assert p.shape == (d, r)
        assert q.shape == (d, r)
        return p, q

    @given(pq=_pq_2d())
    def test_extend_2d(self, pq: tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]) -> None:
        p, q = pq
        d, r = p.shape
        np.testing.assert_allclose(q.T.conj() @ p, np.eye(r), atol=1e-12)
        pex, qex = projector.extend(p, q)
        np.testing.assert_allclose(p, pex[:, :r])
        np.testing.assert_allclose(q, qex[:, :r])
        np.testing.assert_allclose(qex.T.conj() @ pex, np.eye(d), atol=1e-12)
        # Special properties coming from SVD
        p_ = pex[:, r:]
        q_ = qex[:, r:]
        np.testing.assert_allclose(p_.T.conj() @ p_, q_.T.conj() @ q_, atol=1e-12)

    @staticmethod
    @st.composite
    def _pq_3d(draw: DrawFn) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
        d0, d1 = draw(hnp.array_shapes(min_dims=2, max_dims=2))
        d = d0 * d1
        x = draw(conftest.almost_diagonal(d))
        r = draw(st.integers(1, d))
        ix = np.linalg.inv(x)
        p = trg_utils.ungroup(x[:, :r], (0, (d0, d1)))
        q = trg_utils.ungroup((ix.T.conj())[:, :r], (0, (d0, d1))).astype(np.complex128, copy=False)
        assert p.shape == (d0, d1, r)
        assert q.shape == (d0, d1, r)
        return p, q

    @given(pq=_pq_3d())
    def test_extend_3d(self, pq: tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]) -> None:
        p, q = pq
        d0, d1, r = p.shape
        d = d0 * d1
        np.testing.assert_allclose(pdot(p, q.conj()), np.eye(r), atol=1e-12)
        pex, qex = projector.extend(p, q)
        np.testing.assert_allclose(p, pex[..., :r])
        np.testing.assert_allclose(q, qex[..., :r])
        np.testing.assert_allclose(pdot(pex, qex.conj()), np.eye(d), atol=1e-12)


class TestNormalize:
    def test_normalize_ng(self) -> None:
        with pytest.raises(ValueError, match=r"Invalid mode"):
            projector.normalize(np.zeros((9, 1)), np.zeros((9, 1)), mode="invalid")  # type: ignore[arg-type]

    @staticmethod
    @st.composite
    def _pq(draw: DrawFn) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
        shape = draw(hnp.array_shapes(max_side=4))
        extra = math.prod(shape)
        p = draw(conftest.almost_diagonal(extra)).reshape((*shape, extra))
        q = draw(conftest.almost_diagonal(extra)).reshape((*shape, extra))
        return p, q

    @given(pq=_pq())
    def test_normalize_local(self, pq: tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]) -> None:
        # MEMO: Invalid input for dirty testing
        p, q = pq
        orig = np.diagonal(pdot(p, q.conj()))
        p, q = projector.normalize(p, q, mode="local")
        np.testing.assert_allclose(np.diagonal(pdot(p, q.conj())), orig, atol=1e-12)
        for i in range(p.shape[-1]):
            assert np.linalg.norm(p[..., i]) == pytest.approx(np.linalg.norm(q[..., i]))

    @given(pq=_pq())
    def test_normalize_global(self, pq: tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]) -> None:
        # MEMO: Invalid input for dirty testing
        p, q = pq
        orig = np.diagonal(pdot(p, q.conj()))
        p, q = projector.normalize(p, q, mode="global")
        np.testing.assert_allclose(np.diagonal(pdot(p, q.conj())), orig, atol=1e-12)
        assert np.linalg.norm(p) == pytest.approx(np.linalg.norm(q))

    @pytest.mark.parametrize("mode", ["local", "global"])
    def test_normalize_zero(self, mode: Literal["local", "global"]) -> None:
        p = np.zeros((2, 2))
        q = np.zeros((2, 2))
        with pytest.raises(AssertionError):
            projector.normalize(p, q, mode=mode)
