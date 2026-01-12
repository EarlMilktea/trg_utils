"""Tensor decomposition utilities.

This module provides helpers for tensor SVD and QR decompositions that split
axes into left and right groups.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from projector_utils import merge


def tsvd(
    arr: npt.ArrayLike, nu: int
) -> tuple[
    npt.NDArray[Any],
    npt.NDArray[Any],
    npt.NDArray[Any],
]:
    """Perform tensor SVD.

    Parameters
    ----------
    arr
        Input array to decompose.
    nu
        First ``nu`` axes are included in ``U`` and the rest in ``V``.

    Returns
    -------
    U : numpy.ndarray
        First ``nu`` axes of ``arr`` in the same order plus a new axis appended
        at the end.
    S : numpy.ndarray
        1D array of singular values.
    V : numpy.ndarray
        Last ``arr.ndim - nu`` axes of ``arr`` in the same order plus a new axis
        appended at the end.

    Notes
    -----
    ``arr`` can be reconstructed by an einsum ``"(A)x,x,(B)x->(A)(B)"`` where
    ``(A)`` and ``(B)`` stand for the first ``nu`` axes and the rest in ``arr``,
    respectively.
    """
    arr = np.asarray(arr)
    nuc = arr.ndim - nu
    if not (nu > 0 and nuc > 0):
        msg = "nu must be between 1 and arr.ndim - 1."
        raise ValueError(msg)
    work = merge.group(arr, nuc)
    work = work.transpose(-1, *range(work.ndim - 1))
    work = merge.group(work, nu).T
    u, s, vh = np.linalg.svd(work, full_matrices=False)
    u = merge.ungroup(u.T, arr.shape[:nu])
    u = u.transpose(*range(1, u.ndim), 0)
    v = merge.ungroup(vh, arr.shape[nu:])
    v = v.transpose(*range(1, v.ndim), 0)
    return u, s, v


def tqr(arr: npt.ArrayLike, nq: int) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Perform tensor QR decomposition.

    Parameters
    ----------
    arr
        Input array to decompose.
    nq
        First ``nq`` axes are included in ``Q`` and the rest in ``R``.

    Returns
    -------
    Q : numpy.ndarray
        First ``nq`` axes of ``arr`` in the same order plus a new axis appended
        at the end.
    R : numpy.ndarray
        Last ``arr.ndim - nq`` axes of ``arr`` in the same order plus a new axis
        appended at the end.

    Notes
    -----
    ``arr`` can be reconstructed by an einsum ``"(A)x,(B)x->(A)(B)"`` where
    ``(A)`` and ``(B)`` stand for the first ``nq`` axes and the rest in ``arr``,
    respectively.
    """
    arr = np.asarray(arr)
    nqc = arr.ndim - nq
    if not (nq > 0 and nqc > 0):
        msg = "nq must be between 1 and arr.ndim - 1."
        raise ValueError(msg)
    work = merge.group(arr, nqc)
    work = work.transpose(-1, *range(work.ndim - 1))
    work = merge.group(work, nq).T
    q, r = np.linalg.qr(work, mode="reduced")
    q = merge.ungroup(q.T, arr.shape[:nq])
    q = q.transpose(*range(1, q.ndim), 0)
    r = merge.ungroup(r, arr.shape[nq:])
    r = r.transpose(*range(1, r.ndim), 0)
    return q, r
