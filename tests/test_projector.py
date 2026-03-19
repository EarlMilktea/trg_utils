from __future__ import annotations

from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pytest

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

    @pytest.mark.parametrize(("d", "r"), [(3, 1), (3, 2), (3, 3)])
    def test_extend_2d(self, rng: np.random.Generator, d: int, r: int) -> None:
        x = np.eye(d) + 0.1 * conftest.f128_random(rng, (d, d))
        ix = np.linalg.inv(x)
        p = x[:, :r]
        q = (ix.T.conj())[:, :r]
        assert p.shape == (d, r)
        assert q.shape == (d, r)
        np.testing.assert_allclose(q.T.conj() @ p, np.eye(r), atol=1e-12)
        pex, qex = projector.extend(p, q)
        np.testing.assert_allclose(p, pex[:, :r])
        np.testing.assert_allclose(q, qex[:, :r])
        np.testing.assert_allclose(qex.T.conj() @ pex, np.eye(d), atol=1e-12)
        # Special properties coming from SVD
        p_ = pex[:, r:]
        q_ = qex[:, r:]
        np.testing.assert_allclose(p_.T.conj() @ p_, q_.T.conj() @ q_, atol=1e-12)

    @pytest.mark.parametrize(
        ("d0", "d1", "r"),
        [(2, 3, 1), (2, 3, 2), (2, 3, 3), (2, 3, 6)],
    )
    def test_extend_3d(self, rng: np.random.Generator, d0: int, d1: int, r: int) -> None:
        d = d0 * d1
        x = np.eye(d) + 0.1 * conftest.f128_random(rng, (d, d))
        ix = np.linalg.inv(x)
        p = trg_utils.ungroup(x[:, :r], (0, (d0, d1)))
        q = trg_utils.ungroup((ix.T.conj())[:, :r], (0, (d0, d1)))
        assert p.shape == (d0, d1, r)
        assert q.shape == (d0, d1, r)
        np.testing.assert_allclose(pdot(p, q.conj()), np.eye(r), atol=1e-12)
        pex, qex = projector.extend(p, q)
        np.testing.assert_allclose(p, pex[..., :r])
        np.testing.assert_allclose(q, qex[..., :r])
        np.testing.assert_allclose(pdot(pex, qex.conj()), np.eye(d), atol=1e-12)


class TestNormalize:
    def test_normalize_ng(self) -> None:
        with pytest.raises(ValueError, match=r"Invalid mode"):
            projector.normalize(np.zeros((9, 1)), np.zeros((9, 1)), mode="invalid")  # pyright: ignore[reportArgumentType]

    @pytest.mark.parametrize("mode", ["local", "global"])
    @pytest.mark.parametrize(
        "shape",
        [
            (3, 1),
            (3, 2),
            (3, 3),
            (2, 3, 1),
            (2, 3, 2),
            (2, 3, 3),
            (2, 3, 6),
        ],
    )
    def test_normalize(
        self, rng: np.random.Generator, shape: tuple[int, ...], mode: Literal["local", "global"]
    ) -> None:
        # MEMO: Invalid input for dirty testing
        x = conftest.f128_random(rng, shape)
        orig = pdot(x, x.conj())
        p, q = projector.normalize(x, x, mode=mode)
        np.testing.assert_allclose(pdot(p, q.conj()), orig, atol=1e-12)
        np.testing.assert_allclose(
            np.diagonal(pdot(p, p.conj())),
            np.diagonal(pdot(q, q.conj())),
            atol=1e-12,
        )

    @pytest.mark.parametrize("mode", ["local", "global"])
    def test_normalize_zero(self, mode: Literal["local", "global"]) -> None:
        p = np.zeros((2, 2))
        q = np.zeros((2, 2))
        with pytest.raises(AssertionError):
            projector.normalize(p, q, mode=mode)
