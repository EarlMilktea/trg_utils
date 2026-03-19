"""Manipulate projector tensors."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

import trg_utils
from trg_utils import _index


def extend(p: npt.NDArray[Any], q: npt.NDArray[Any]) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    r"""Extend projectors.

    This function assumes that the input projectors :math:`P` and :math:`Q` satisfy :math:`Q^\dagger P = E`.
    Then it computes :math:`P_{ex}` and :math:`Q_{ex}` satisfying :math:`Q_{ex}^\dagger P_{ex} = E` by basis extension.

    Parameters
    ----------
    p
        Left projector.
    q
        Right projector.

    Returns
    -------
    pex : `numpy.ndarray`
        `p` extended to a full-rank isometry.
    qex : `numpy.ndarray`
        `q` extended to a full-rank isometry.
    """
    sp: tuple[int, ...] = p.shape
    sq: tuple[int, ...] = q.shape
    _index.assert_pshapes(sp, sq)
    p = trg_utils.group(p, (range(p.ndim - 1), -1))
    q = trg_utils.group(q, (range(q.ndim - 1), -1))
    d: int
    r: int
    d, r = p.shape
    rc = d - r
    proj_c = np.eye(d) - p @ q.T.conj()
    # Construct the dual bases of the perpendicular complement
    u, s, vh = np.linalg.svd(proj_c)
    w = np.diag(np.sqrt(s))
    v = vh.T.conj()
    # Satisfies s @ vh @ u @ s = s: implies (w @ vh @ u @ w)[:rc, :rc] = np.eye(rc)
    pex = np.concatenate((p, (u @ w)[:, :rc]), axis=1)
    qex = np.concatenate((q, (v @ w)[:, :rc]), axis=1)
    return trg_utils.ungroup(pex, (0, sp[:-1])), trg_utils.ungroup(qex, (0, sq[:-1]))
