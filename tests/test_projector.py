from __future__ import annotations

from typing import Literal

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given, settings
from hypothesis.strategies import DrawFn, SearchStrategy

from tests import conftest
from trg_utils import projector


def _pq(noise: float = 0.0) -> SearchStrategy[tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]]:

    @st.composite
    def _inner(draw: DrawFn) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
        (d,) = draw(hnp.array_shapes(min_dims=1, max_dims=1))
        x = draw(conftest.almost_diagonal(d))
        r = draw(st.integers(1, d))
        ix = np.linalg.inv(x)
        p = x[:, :r]
        q = ix.T.conj()[:, :r].astype(np.complex128, copy=False)
        assert p.shape == (d, r)
        assert q.shape == (d, r)
        p += draw(conftest.shaped_f128(p.shape)) * noise
        q += draw(conftest.shaped_f128(q.shape)) * noise
        return p, q

    return _inner()


class TestExtend:
    def test_extend_ng(self) -> None:
        with pytest.raises(ValueError, match=r"Inconsistent"):
            projector.extend(np.zeros((9, 1)), np.zeros((9, 2)))

        with pytest.raises(ValueError, match=r"matrices"):
            projector.extend(np.zeros((9,)), np.zeros((9,)))

        with pytest.raises(ValueError, match=r"empty"):
            projector.extend(np.zeros((0, 9)), np.zeros((0, 9)))

        with pytest.raises(ValueError, match=r"larger"):
            projector.extend(np.zeros((4, 9)), np.zeros((4, 9)))

    def test_extend_full(self) -> None:
        p = np.eye(9)
        q = np.eye(9)
        pex, qex = projector.extend(p, q)
        np.testing.assert_allclose(pex, p)
        np.testing.assert_allclose(qex, q)

    def test_extend_empty(self) -> None:
        p = np.zeros((9, 0))
        q = np.zeros((9, 0))
        pex, qex = projector.extend(p, q)
        np.testing.assert_allclose(pex, np.eye(9))
        np.testing.assert_allclose(qex, np.eye(9))

    @given(pq=_pq())
    def test_extend(self, pq: tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]) -> None:
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


class TestNormalize:
    def test_normalize_ng(self) -> None:
        with pytest.raises(ValueError, match=r"Invalid mode"):
            projector.normalize(np.zeros((9, 1)), np.zeros((9, 1)), mode="invalid")  # type: ignore[arg-type]

        with pytest.raises(ValueError, match=r"non-empty"):
            projector.normalize(np.zeros((9, 0)), np.zeros((9, 0)), mode="local")

    @given(pq=_pq())
    def test_normalize_local(self, pq: tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]) -> None:
        # MEMO: Invalid input for dirty testing
        p, q = pq
        orig = q.T.conj() @ p
        p, q = projector.normalize(p, q, mode="local")
        np.testing.assert_allclose(q.T.conj() @ p, orig, atol=1e-12)
        for i in range(p.shape[1]):
            assert np.linalg.norm(p[:, i]) == pytest.approx(np.linalg.norm(q[:, i]))

    @given(pq=_pq())
    def test_normalize_global(self, pq: tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]) -> None:
        # MEMO: Invalid input for dirty testing
        p, q = pq
        orig = q.T.conj() @ p
        p, q = projector.normalize(p, q, mode="global")
        np.testing.assert_allclose(q.T.conj() @ p, orig, atol=1e-12)
        assert np.linalg.norm(p) == pytest.approx(np.linalg.norm(q))

    @pytest.mark.parametrize("mode", ["local", "global"])
    def test_normalize_zero(self, mode: Literal["local", "global"]) -> None:
        p = np.zeros((2, 2))
        q = np.zeros((2, 2))
        with pytest.raises(AssertionError):
            projector.normalize(p, q, mode=mode)


class TestRefine:
    def test_refine_ng(self) -> None:
        with pytest.raises(ValueError, match=r"non-empty"):
            projector.refine(np.zeros((9, 0)), np.zeros((9, 0)))

    def test_no_op(self) -> None:
        p = np.eye(9)
        q = np.eye(9)
        p_ref, q_ref = projector.refine(p, q)
        np.testing.assert_allclose(p_ref, p)
        np.testing.assert_allclose(q_ref, q)

    @pytest.mark.filterwarnings("ignore:Generating overly large repr")
    @settings(deadline=None)
    @given(pq=_pq(1e-3))
    def test_refine(self, pq: tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]) -> None:
        p, q = pq
        _, chi = p.shape
        p_ref, q_ref = projector.refine(p, q)
        assert p_ref.shape == p.shape
        assert q_ref.shape == q.shape
        np.testing.assert_allclose(q_ref.T.conj() @ p_ref, np.eye(chi), atol=1e-12)

    def test_refine_pivot(self) -> None:
        p = np.eye(2)
        q = np.asarray([[0, 1], [1, 0]])
        with pytest.raises(ValueError, match=r"Pivoting required"):
            projector.refine(p, q)
