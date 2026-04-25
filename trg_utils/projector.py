"""Manipulate projector tensors."""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Literal, TypeVar

import numpy as np
import numpy.typing as npt
from scipy import linalg

from trg_utils import _index, merge

_T = TypeVar("_T", bound=np.generic)


@dataclasses.dataclass
class _ToMatrix:
    tshape: tuple[int, ...]

    @staticmethod
    def encode(t: npt.NDArray[_T]) -> tuple[npt.NDArray[_T], _ToMatrix]:
        return merge.group(t, (range(t.ndim - 1), -1)), _ToMatrix(t.shape)

    def decode(self, m: npt.NDArray[_T]) -> npt.NDArray[_T]:
        return merge.ungroup(m, (0, self.tshape[:-1]))


def extend(p: npt.NDArray[Any], q: npt.NDArray[Any]) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    r"""Extend projector bases.

    This function assumes that the input bases :math:`P` and :math:`Q` satisfy :math:`Q^\dagger P = E`.
    Then it computes :math:`P_{\mathrm{ex}}` and :math:`Q_{\mathrm{ex}}`
    satisfying :math:`Q_{\mathrm{ex}}^\dagger P_{\mathrm{ex}} = E` by basis extension.

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
    If both inputs are empty, returns identity.

    Examples
    --------
    >>> import numpy as np
    >>> from trg_utils import projector
    >>> p = np.array([[1, 0], [0, 1], [0, 0]])
    >>> q = np.array([[1, 0], [0, 1], [1, 0]])
    >>> pex, qex = projector.extend(p, q)
    >>> assert np.allclose(qex.T.conj() @ pex, np.eye(3))
    """
    _index.assert_pshapes(p.shape, q.shape, allow_empty=True)
    p, dec = _ToMatrix.encode(p)
    q, _ = _ToMatrix.encode(q)
    d, r = p.shape
    rc = d - r
    proj_c = np.eye(d) - p @ q.T.conj()
    if rc == 0:
        proj_c[...] = 0  # Eliminate numerical noise
    # Construct the dual bases of the perpendicular complement
    u, s, vh = np.linalg.svd(proj_c)
    w = np.diag(np.sqrt(s))
    # Satisfies s @ vh @ u @ s = s: implies (w @ vh @ u @ w)[:rc, :rc] = np.eye(rc)
    pex = np.concatenate((p, (u @ w)[:, :rc]), axis=1)
    qex = np.concatenate((q, (vh.T.conj() @ w)[:, :rc]), axis=1)
    return dec.decode(pex), dec.decode(qex)


def _normalize_local(p: npt.NDArray[Any], q: npt.NDArray[Any]) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    _, chi = p.shape
    ps: list[npt.NDArray[Any]] = []
    qs: list[npt.NDArray[Any]] = []
    for i in range(chi):
        cp = np.linalg.norm(p[:, i])
        cq = np.linalg.norm(q[:, i])
        c = math.sqrt(cp * cq)
        if c == 0:
            msg = "Basis overlap is zero."
            raise AssertionError(msg)
        ps.append((c / cp) * p[:, i])
        qs.append((c / cq) * q[:, i])
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
    r"""Adjust norms of dual bases without affecting the biorthonormality.

    This function adjusts the norms of :math:`P` and :math:`Q` while :math:`Q^\dagger P = E` is maintained.

    Parameters
    ----------
    p
        Left dual basis.
    q
        Right dual basis.
    mode
        Normalization mode.
        If ``"local"``, each pair of vectors will have the same norm. If ``"global"``, the entire bases will be used.

    Returns
    -------
    p : `numpy.ndarray`
        Normalized left dual basis.
    q : `numpy.ndarray`
        Normalized right dual basis.

    Notes
    -----
    Biorthonormality is not validated.

    Examples
    --------
    >>> import numpy as np
    >>> from trg_utils import projector
    >>> p0 = np.array([[1, 0], [0, 1], [0, 0]])
    >>> q0 = np.array([[1, 0], [0, 1], [1, 0]])
    >>> p, q = projector.normalize(p0, q0, mode="local")
    >>> assert np.allclose(np.linalg.norm(p[..., 0]), np.linalg.norm(q[..., 0]))
    >>> assert np.allclose(np.linalg.norm(p[..., 1]), np.linalg.norm(q[..., 1]))
    >>> p, q = projector.normalize(p0, q0, mode="global")
    >>> assert np.allclose(np.linalg.norm(p), np.linalg.norm(q))
    """
    _index.assert_pshapes(p.shape, q.shape)
    p, dec = _ToMatrix.encode(p)
    q, _ = _ToMatrix.encode(q)
    match mode:
        case "local":
            p, q = _normalize_local(p, q)
        case "global":
            p, q = _normalize_global(p, q)
        case _:
            msg = "Invalid mode."
            raise ValueError(msg)
    return dec.decode(p), dec.decode(q)


def refine(p: npt.NDArray[Any], q: npt.NDArray[Any]) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    r"""Refine dual bases by LU decomposition.

    Given ill-conditioned dual bases, this function improves the orthonormality :math:`Q^\dagger P \simeq E`.

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

    Examples
    --------
    >>> import numpy as np
    >>> from trg_utils import projector
    >>> p = np.eye(3) + 1e-2 * np.random.rand(3, 3)
    >>> q = np.eye(3) + 1e-2 * np.random.rand(3, 3)
    >>> p, q = projector.refine(p, q)
    >>> assert np.allclose(q.T.conj() @ p, np.eye(3))
    """
    _index.assert_pshapes(p.shape, q.shape)
    p, dec = _ToMatrix.encode(p)
    q, _ = _ToMatrix.encode(q)
    x = q.T.conj() @ p
    lu, piv = linalg.lu_factor(x)
    if np.any(piv != range(piv.size)):
        msg = "Pivoting required: maybe too ill-conditioned."
        raise ValueError(msg)
    *_, d = dec.tshape
    up = linalg.solve_triangular(lu, np.eye(d))  # Use U as-is
    l = np.tril(lu, k=-1) + np.eye(d)  # Restore L
    uq = linalg.solve_triangular(l.T.conj(), np.eye(d))
    return dec.decode(p @ up), dec.decode(q @ uq)
