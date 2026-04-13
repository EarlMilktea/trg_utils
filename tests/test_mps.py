from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given
from hypothesis import strategies as st

from tests import conftest
from trg_utils import mps
from trg_utils.mps import _CanonicalMPS


@given(ts=conftest.random_mps(4))
def test_ad(ts: list[npt.NDArray[np.complex128]]) -> None:
    orig = copy.deepcopy(ts)
    res = mps._detach_dummy(mps._attach_dummy(ts))
    for t1, t2 in zip(orig, res, strict=True):
        np.testing.assert_allclose(t1, t2)


def test_attach_ng() -> None:
    with pytest.raises(ValueError, match=r".*two tensors.*"):
        mps._attach_dummy([np.zeros((2, 2))])

    with pytest.raises(ValueError, match=r".*2D.*"):
        mps._attach_dummy([np.zeros((2, 2, 2)), np.zeros((2, 2))])

    with pytest.raises(ValueError, match=r".*2D.*"):
        mps._attach_dummy([np.zeros((2, 2)), np.zeros((2, 2, 2))])

    with pytest.raises(ValueError, match=r".*3D.*"):
        mps._attach_dummy([np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2))])

    with pytest.raises(ValueError, match=r".*leg dimension.*"):
        mps._attach_dummy([np.zeros((2, 2)), np.zeros((2, 3))])


class TestCanonicalMPS:
    @staticmethod
    def _is_isometry_u(t: npt.NDArray[Any]) -> bool:
        cont = np.einsum("abi,abj->ij", t, t.conj())
        chi, _ = cont.shape
        return np.allclose(cont, np.eye(chi), atol=1e-12)

    @staticmethod
    def _is_isometry_v(t: npt.NDArray[Any]) -> bool:
        cont = np.einsum("aib,ajb->ij", t, t.conj())
        chi, _ = cont.shape
        return np.allclose(cont, np.eye(chi), atol=1e-12)

    @given(ts=st.one_of(*(conftest.random_mps(n) for n in range(2, 6))))
    def test_from_ts(self, ts: list[npt.NDArray[np.complex128]]) -> None:
        psi = _CanonicalMPS.from_ts(mps._attach_dummy(ts))
        assert psi.n == len(ts)
        assert psi.ts is not ts

    @given(ts=conftest.random_mps(3))
    def test_svd_exact_3(self, ts: list[npt.NDArray[np.complex128]]) -> None:
        ts = mps._attach_dummy(ts)
        orig = np.einsum("ila,jab,kbm->ijklm", *ts)
        psi = _CanonicalMPS.from_ts(ts)

        (u0,), s, (v1, v2) = psi.svd_at(0)
        assert self._is_isometry_u(u0)
        assert self._is_isometry_v(v1)
        assert self._is_isometry_v(v2)
        res = np.einsum("ila,a,jab,kbm->ijklm", u0, s, v1, v2)
        np.testing.assert_allclose(res, orig, atol=1e-12)

        (u0, u1), s, (v2,) = psi.svd_at(1)
        assert self._is_isometry_u(u0)
        assert self._is_isometry_u(u1)
        assert self._is_isometry_v(v2)
        res = np.einsum("ila,jab,b,kbm->ijklm", u0, u1, s, v2)
        np.testing.assert_allclose(res, orig, atol=1e-12)

    @given(ts=conftest.random_mps(4))
    def test_svd_exact_4(self, ts: list[npt.NDArray[np.complex128]]) -> None:
        ts = mps._attach_dummy(ts)
        orig = np.einsum("ima,jab,kbc,lcn->ijklmn", *ts)
        psi = _CanonicalMPS.from_ts(ts)

        (u0,), s, (v1, v2, v3) = psi.svd_at(0)
        assert self._is_isometry_u(u0)
        assert self._is_isometry_v(v1)
        assert self._is_isometry_v(v2)
        assert self._is_isometry_v(v3)
        res = np.einsum("ima,a,jab,kbc,lcn->ijklmn", u0, s, v1, v2, v3)
        np.testing.assert_allclose(res, orig, atol=1e-12)

        (u0, u1), s, (v2, v3) = psi.svd_at(1)
        assert self._is_isometry_u(u0)
        assert self._is_isometry_u(u1)
        assert self._is_isometry_v(v2)
        assert self._is_isometry_v(v3)
        res = np.einsum("ima,jab,b,kbc,lcn->ijklmn", u0, u1, s, v2, v3)
        np.testing.assert_allclose(res, orig, atol=1e-12)

        (u0, u1, u2), s, (v3,) = psi.svd_at(2)
        assert self._is_isometry_u(u0)
        assert self._is_isometry_u(u1)
        assert self._is_isometry_u(u2)
        assert self._is_isometry_v(v3)
        res = np.einsum("ima,jab,kbc,c,lcn->ijklmn", u0, u1, u2, s, v3)
        np.testing.assert_allclose(res, orig, atol=1e-12)

    @given(ts=conftest.random_mps(7))
    def test_shape(self, ts: list[npt.NDArray[np.complex128]]) -> None:
        ts = mps._attach_dummy(ts)
        psi = _CanonicalMPS.from_ts(ts)

        chi = 1
        chis_u = [chi]  # Dummy 1 at the beginning
        for t in ts[:-1]:
            d, _, dp = t.shape
            chi = min(dp, chi * d)
            chis_u.append(chi)

        chi = 1
        chis_v = [chi]  # Dummy 1 at the end
        for cu, t in zip(reversed(chis_u[1:]), reversed(ts[1:]), strict=True):
            d, _, _ = t.shape
            chi = min(cu, chi * d)
            chis_v.append(chi)
        chis_v.reverse()

        for i in range(psi.n - 1):
            cu = chis_u[i + 1]
            cv = chis_v[i]
            assert psi.ss[i].size == cv
            assert cv <= math.prod(t.shape[0] for t in ts[: i + 1])
            assert cv <= math.prod(t.shape[0] for t in ts[i + 1 :])
            assert psi.gauge[i].shape == (cu, cv)
            assert psi.us[i].shape == (ts[i].shape[0], chis_u[i], cu)
            assert psi.vs[i].shape == (ts[i + 1].shape[0], cv, chis_v[i + 1])

    @given(ts=conftest.random_mps(4))
    def test_trunc_lean(self, ts: list[npt.NDArray[np.complex128]]) -> None:
        chi = 1
        ts = mps._attach_dummy(ts)
        psi = _CanonicalMPS.from_ts(ts, chi=chi)

        (u0,), s, (v1, v2, v3) = psi.svd_at(0)
        assert self._is_isometry_u(u0)
        assert self._is_isometry_v(v1)
        assert self._is_isometry_v(v2)
        assert self._is_isometry_v(v3)
        ref = np.einsum("ima,a,jab,kbc,lcn->ijklmn", u0, s, v1, v2, v3)
        v2[:, chi:, :] = 0
        v3[:, chi:, :] = 0
        cmp = np.einsum("ima,a,jab,kbc,lcn->ijklmn", u0, s, v1, v2, v3)
        np.testing.assert_allclose(cmp, ref, atol=1e-12)

        (u0, u1), s, (v2, v3) = psi.svd_at(1)
        assert self._is_isometry_u(u0)
        assert self._is_isometry_u(u1)
        assert self._is_isometry_v(v2)
        assert self._is_isometry_v(v3)
        ref = np.einsum("ima,jab,b,kbc,lcn->ijklmn", u0, u1, s, v2, v3)
        v3[:, chi:, :] = 0
        cmp = np.einsum("ima,jab,b,kbc,lcn->ijklmn", u0, u1, s, v2, v3)
        np.testing.assert_allclose(cmp, ref, atol=1e-12)

        (u0, u1, u2), s, (v3,) = psi.svd_at(2)
        assert self._is_isometry_u(u0)
        assert self._is_isometry_u(u1)
        assert self._is_isometry_u(u2)
        assert self._is_isometry_v(v3)
