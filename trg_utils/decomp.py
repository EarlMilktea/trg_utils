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
    iu = _index.normalize(d, _index.materialize(iu))
    iv = _index.normalize(d, _index.materialize(iv))
    _index.assert_span(d, iu, iv)
    work = arr.transpose(*iu, *iv)
    nu = len(iu)
    work = merge.group(work, (range(nu), range(nu, work.ndim)))
    u, s, vh = np.linalg.svd(work, full_matrices=False)
    su = tuple(arr.shape[i] for i in iu)
    sv = tuple(arr.shape[i] for i in iv)
    u = merge.ungroup(u, (0, su))
    v = merge.ungroup(vh, (-1, sv))
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
    iq = _index.normalize(d, _index.materialize(iq))
    ir = _index.normalize(d, _index.materialize(ir))
    _index.assert_span(d, iq, ir)
    work = arr.transpose(*iq, *ir)
    nq = len(iq)
    work = merge.group(work, (range(nq), range(nq, work.ndim)))
    q, r = np.linalg.qr(work, mode="reduced")
    sq = tuple(arr.shape[i] for i in iq)
    sr = tuple(arr.shape[i] for i in ir)
    q = merge.ungroup(q, (0, sq))
    r = merge.ungroup(r, (-1, sr))
    r = r.transpose(*range(1, r.ndim), 0)
    return q, r


def hosvd(arr: npt.ArrayLike, iu: Sequence[SupportsIndex]) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Perform higher-order SVD.

    Parameters
    ----------
    arr
        Input array to decompose.
    iu
        Axis indices to be included in ``U``.

    Returns
    -------
    S : numpy.ndarray
        1D array of singular values.
    U : numpy.ndarray
        Axes from ``iu`` in the same order plus a new axis appended at the end.

    Raises
    ------
    ValueError
        If all axes are included in ``iu``. See :func:`tsvd` for more details.

    Notes
    -----
    This function is similar to :func:`tsvd`, but it computes only the ``U`` matrix using HOSVD.
    """
    arr = np.asarray(arr)
    d = arr.ndim
    iu = _index.normalize(d, _index.materialize(iu))
    _index.assert_allunique(iu)
    iv = tuple(i for i in range(d) if i not in iu)
    if not iv:
        msg = "At least one axis must be excluded from 'iu' to perform HOSVD."
        raise ValueError(msg)
    work = merge.group(arr, (iu, iv))
    vals, vecs = np.linalg.eigh(work @ work.T)
    perm = np.argsort(vals)[::-1]
    vals = np.maximum(vals, 0)
    su = tuple(arr.shape[i] for i in iu)
    return np.sqrt(vals[perm]), merge.ungroup(vecs[:, perm], (0, su))
