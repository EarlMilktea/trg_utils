"""Manipulate projector tensors."""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from trg_utils import _index, merge


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
        ``p`` extended to a full-rank isometry.
    qex : `numpy.ndarray`
        ``q`` extended to a full-rank isometry.

    Notes
    -----
    Biorthonormality is not validated.
    If both projectors are empty, returns identity matrices.
    """
    sp: tuple[int, ...] = p.shape
    sq: tuple[int, ...] = q.shape
    _index.assert_pshapes(sp, sq, allow_empty=True)
    p = merge.group(p, (range(p.ndim - 1), -1))
    q = merge.group(q, (range(q.ndim - 1), -1))
    d: int
    r: int
    d, r = p.shape
    rc = d - r
    proj_c = np.eye(d) - p @ q.T.conj()
    if rc == 0:
        proj_c[...] = 0  # Eliminate numerical noise
    # Construct the dual bases of the perpendicular complement
    u, s, vh = np.linalg.svd(proj_c)
    w = np.diag(np.sqrt(s))
    v = vh.T.conj()
    # Satisfies s @ vh @ u @ s = s: implies (w @ vh @ u @ w)[:rc, :rc] = np.eye(rc)
    pex = np.concatenate((p, (u @ w)[:, :rc]), axis=1)
    qex = np.concatenate((q, (v @ w)[:, :rc]), axis=1)
    return merge.ungroup(pex, (0, sp[:-1])), merge.ungroup(qex, (0, sq[:-1]))


def _normalize_local(p: npt.NDArray[Any], q: npt.NDArray[Any]) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    chi: int
    *_, chi = p.shape
    ps: list[npt.NDArray[Any]] = []
    qs: list[npt.NDArray[Any]] = []
    for i in range(chi):
        cp = np.linalg.norm(p[..., i])
        cq = np.linalg.norm(q[..., i])
        c = math.sqrt(cp * cq)
        if c == 0:
            msg = "Basis overlap is zero."
            raise AssertionError(msg)
        ps.append((c / cp) * p[..., i])
        qs.append((c / cq) * q[..., i])
    return np.stack(ps, axis=-1), np.stack(qs, axis=-1)


def _normalize_global(p: npt.NDArray[Any], q: npt.NDArray[Any]) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    cp = np.linalg.norm(p)
    cq = np.linalg.norm(q)
    c = math.sqrt(cp * cq)
    if c == 0:
        msg = "Basis overlap is zero."
        raise AssertionError(msg)
    return (c / cp) * p, (c / cq) * q


def normalize(
    p: npt.NDArray[Any], q: npt.NDArray[Any], mode: Literal["local", "global"]
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Adjust norms of projector pairs without affecting the biorthogonality.

    Parameters
    ----------
    p
        Left projector.
    q
        Right projector.
    mode
        Normalization mode.
        If ``"local"``, each pair of vectors will have the same norm. If ``"global"``, the entire projectors are used.

    Returns
    -------
    p : `numpy.ndarray`
        Normalized left projector.
    q : `numpy.ndarray`
        Normalized right projector.

    Raises
    ------
    ValueError
        If ``mode`` is invalid.

    Notes
    -----
    Biorthonormality is not validated.
    """
    _index.assert_pshapes(p.shape, q.shape)
    match mode:
        case "local":
            return _normalize_local(p, q)
        case "global":
            return _normalize_global(p, q)
        case _:
            msg = "Invalid mode."
            raise ValueError(msg)
