"""Tensor decompositions.

This module provides functions to perform tensor decompositions such as SVD and QR.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, SupportsIndex

import numpy as np
import numpy.typing as npt

from trg_utils import _index, merge

if TYPE_CHECKING:
    from collections.abc import Sequence


def tsvd(
    arr: npt.ArrayLike, iu: Sequence[SupportsIndex], iv: Sequence[SupportsIndex]
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
    """Perform tensor SVD.

    Parameters
    ----------
    arr
        Input array to decompose.
    iu
        Axis indices included in ``U``.
    iv
        Axis indices included in ``V``.

    Returns
    -------
    U : numpy.ndarray
        Axes from ``iu`` in the same order plus a new axis appended at the end.
    S : numpy.ndarray
        1D array of singular values.
    V : numpy.ndarray
        Axes from ``iv`` in the same order plus a new axis appended at the end.
    """
    arr = np.asarray(arr)
    d = arr.ndim
    iu = _index.normalize(d, iu)
    iv = _index.normalize(d, iv)
    _index.assert_span(d, iu, iv)
    work = arr.transpose(*iu, *iv)
    nu = len(iu)
    # merge from back
    work = merge._group_impl(work, nu, work.ndim)
    work = merge._group_impl(work, 0, nu)
    u, s, vh = np.linalg.svd(work, full_matrices=False)
    su = tuple(arr.shape[i] for i in iu)
    sv = tuple(arr.shape[i] for i in iv)
    u = merge._ungroup_impl(u, 0, su)
    v = merge._ungroup_impl(vh, -1, sv)
    v = v.transpose(*range(1, v.ndim), 0)
    return u, s, v


def tqr(
    arr: npt.ArrayLike, iq: Sequence[SupportsIndex], ir: Sequence[SupportsIndex]
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Perform tensor QR decomposition.

    Parameters
    ----------
    arr
        Input array to decompose.
    iq
        Axis indices included in ``Q``.
    ir
        Axis indices included in ``R``.

    Returns
    -------
    Q : numpy.ndarray
        Axes from ``iq`` in the same order plus a new axis appended at the end.
    R : numpy.ndarray
        Axes from ``ir`` in the same order plus a new axis appended at the end.
    """
    arr = np.asarray(arr)
    d = arr.ndim
    iq = _index.normalize(d, iq)
    ir = _index.normalize(d, ir)
    _index.assert_span(d, iq, ir)
    work = arr.transpose(*iq, *ir)
    nq = len(iq)
    work = merge._group_impl(work, nq, work.ndim)
    work = merge._group_impl(work, 0, nq)
    q, r = np.linalg.qr(work, mode="reduced")
    sq = tuple(arr.shape[i] for i in iq)
    sr = tuple(arr.shape[i] for i in ir)
    q = merge._ungroup_impl(q, 0, sq)
    r = merge._ungroup_impl(r, -1, sr)
    r = r.transpose(*range(1, r.ndim), 0)
    return q, r
