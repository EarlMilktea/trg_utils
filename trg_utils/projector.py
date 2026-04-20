"""Manipulate projector tensors."""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from scipy import linalg

from trg_utils import _index, merge


def extend(p: npt.NDArray[Any], q: npt.NDArray[Any]) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    r"""Extend projectors.

    This function assumes that the input projectors :math:`P` and :math:`Q` satisfy :math:`Q^\dagger P = E`.
    Then it computes :math:`P_{ex}` and :math:`Q_{ex}` satisfying :math:`Q_{ex}^\dagger P_{ex} = E` by basis extension.

    Parameters
    ----------
    p
        Left dual basis.
    q
        Right dual basis.

    Returns
    -------
    pex : `numpy.ndarray`
        ``p`` extended to a full-rank dual basis.
    qex : `numpy.ndarray`
        ``q`` extended to a full-rank dual basis.

    Notes
    -----
    Biorthonormality is not validated.
    If both projectors are empty, returns identity.
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
    """Adjust norms of dual bases without affecting the biorthogonality.

    Parameters
    ----------
    p
        Left dual basis.
    q
        Right dual basis.
    mode
        Normalization mode.
        If ``"local"``, each pair of vectors will have the same norm. If ``"global"``, the entire bases will.

    Returns
    -------
    p : `numpy.ndarray`
        Normalized left dual basis.
    q : `numpy.ndarray`
        Normalized right dual basis.

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


def refine(p: npt.NDArray[Any], q: npt.NDArray[Any]) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    r"""Refine projectors by LU decomposition.

    Given ill-conditioned projectors, this function improves the orthonormality :math:`Q^\dagger P \simeq E`.

    Parameters
    ----------
    p
        Left basis.
    q
        Right basis.

    Returns
    -------
    p : `numpy.ndarray`
        Refined ``p`` close to the original.
    q : `numpy.ndarray`
        Refined ``q`` close to the original.

    Raises
    ------
    ValueError
        If pivoting is required: this should not happen as long as :math:`Q^\dagger P` is close enough to identity.
    """
    _index.assert_pshapes(p.shape, q.shape)
    spq = p.shape
    p = merge.group(p, (range(p.ndim - 1), -1))
    q = merge.group(q, (range(q.ndim - 1), -1))
    x = q.T.conj() @ p
    lu, piv = linalg.lu_factor(x)
    if np.any(piv != range(piv.size)):
        msg = "Pivoting required: maybe too ill-conditioned."
        raise ValueError(msg)
    *_, d = spq
    up = linalg.solve_triangular(lu, np.eye(d))  # Use U as-is
    l = np.tril(lu, k=-1) + np.eye(d)  # Restore L
    uq = linalg.solve_triangular(l.T.conj(), np.eye(d))
    return merge.ungroup(p @ up, (0, spq[:-1])), merge.ungroup(q @ uq, (0, spq[:-1]))
