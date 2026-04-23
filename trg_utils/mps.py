"""MPS optimization."""

from __future__ import annotations

import copy
import dataclasses
import itertools
import math
from collections.abc import Iterable, Sequence
from typing import Any, NamedTuple, TypeVar

import numpy as np
import numpy.typing as npt
from scipy import stats

from trg_utils import decomp, projector

_T = TypeVar("_T", bound=np.generic)


def _attach_dummy(ts: Sequence[npt.NDArray[_T]]) -> list[npt.NDArray[_T]]:
    if len(ts) < 2:
        msg = "At least two tensors are required."
        raise ValueError(msg)
    head, *mid, tail = ts
    if head.ndim != 2 or tail.ndim != 2:
        msg = "Edge tensors must be 2D."
        raise ValueError(msg)
    if any(t.ndim != 3 for t in mid):
        msg = "Bulk tensors must be 3D."
        raise ValueError(msg)
    ret = [
        head[:, np.newaxis, :],
        *mid,
        tail[:, :, np.newaxis],
    ]
    for left, right in itertools.pairwise(ret):
        if left.shape[2] != right.shape[1]:
            msg = "Inconsistent closed bond dimensions."
            raise ValueError(msg)
    return ret


def _detach_dummy(ts: Iterable[npt.NDArray[_T]]) -> list[npt.NDArray[_T]]:
    head, *mid, tail = ts
    return [
        head[:, 0, :],
        *mid,
        tail[:, :, 0],
    ]


class ProjectorResult(NamedTuple):
    """Projector dual basis and associated weights.

    Attributes
    ----------
    s
        Relative contribution of each basis to the contraction result. Can have zeros.
    p
        Left dual basis biorthonormal to ``q``.
        Should be attached to ``ts[i]`` and not to ``ts[i + 1]``.
    q
        Right dual basis biorthonormal to ``p``.
        Should be attached to ``ts[i + 1]`` and not to ``ts[i]``.
    """

    s: npt.NDArray[Any]
    p: npt.NDArray[Any]
    q: npt.NDArray[Any]

    @property
    def rank(self) -> int:
        """Number of nonzero weights."""
        return int(np.count_nonzero(self.s > 0))


_CHI_INF = 10**17


@dataclasses.dataclass
class _CanonicalMPS:
    ts: list[npt.NDArray[Any]]
    chi: int
    ss: list[npt.NDArray[Any]] = dataclasses.field(default_factory=list)
    us: list[npt.NDArray[Any]] = dataclasses.field(default_factory=list)
    vs: list[npt.NDArray[Any]] = dataclasses.field(default_factory=list)
    # Gauge transformation applied just after U
    gauge: list[npt.NDArray[Any]] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        assert self.n >= 2
        if self.chi <= 0:
            msg = "chi must be positive."
            raise ValueError(msg)

    def zerofill(self, arr: npt.NDArray[Any]) -> npt.NDArray[Any]:
        arr = arr.copy()
        arr[..., self.chi :] = 0
        return arr

    @property
    def n(self) -> int:
        return len(self.ts)

    def _forward(self) -> npt.NDArray[Any]:
        work = self.ts[0]
        s: npt.NDArray[Any] | None = None
        for i in range(self.n - 1):
            u, s, v = decomp.tsvd(work, (0, 1), (2,))
            self.us.append(u)
            work = np.einsum("b,ib,aic->abc", s, v, self.ts[i + 1], optimize=True)
        return work

    def _backward(self, work: npt.NDArray[Any]) -> None:
        for i in reversed(range(self.n - 1)):
            gauge, s, v = decomp.tsvd(work, (1,), (0, 2))
            self.gauge.append(gauge)
            self.ss.append(s)  # Store the raw singular values
            self.vs.append(v.transpose(0, 2, 1))  # Adjust leg order
            work = np.einsum("abi,ij,j->abj", self.us[i], gauge, self.zerofill(s), optimize=True)
        self.gauge.reverse()
        self.ss.reverse()
        self.vs.reverse()

    @staticmethod
    def from_ts(ts_3: Sequence[npt.NDArray[Any]], chi: int | None = None) -> _CanonicalMPS:
        if chi is None:
            chi = _CHI_INF
        mps = _CanonicalMPS([t.copy() for t in ts_3], chi)
        work = mps._forward()
        mps._backward(work)
        return mps

    def svd_at(self, i: int) -> tuple[list[npt.NDArray[Any]], npt.NDArray[Any], list[npt.NDArray[Any]]]:
        nu = i + 1
        nv = self.n - nu
        assert nu > 0
        assert nv > 0
        us = copy.deepcopy(self.us[:nu])
        vs = copy.deepcopy(self.vs[-nv:])
        s = self.ss[i].copy()
        us[-1] = np.einsum("abi,ic->abc", us[-1], self.gauge[i])
        return us, s, vs

    @staticmethod
    def _safe_isqrt(s: npt.NDArray[Any], rtol: float = 1e-14, atol: float = 1e-15) -> npt.NDArray[Any]:
        assert s.ndim == 1
        rank_abs = int(np.count_nonzero(s > atol))
        rank_rel = int(np.count_nonzero(s > rtol * s.max()))
        rank = min(rank_abs, rank_rel)
        return np.asarray(1 / np.sqrt(s[:rank]))

    def _prefix(self) -> list[npt.NDArray[Any]]:
        prefix: list[npt.NDArray[Any]] = [np.eye(1)]
        for t, u in zip(self.ts, self.us, strict=False):
            prefix.append(np.einsum("abi,bc,acj->ij", t.conj(), prefix[-1], u, optimize=True))
        for i, g in enumerate(self.gauge, start=1):
            prefix[i] = prefix[i] @ g  # noqa: PLR6104 (shape change required)
        return prefix

    def _suffix(self) -> list[npt.NDArray[Any]]:
        suffix: list[npt.NDArray[Any]] = [np.eye(1)]
        for t, v in zip(reversed(self.ts), reversed(self.vs), strict=False):
            suffix.append(np.einsum("aib,bc,ajc->ij", t.conj(), suffix[-1], v, optimize=True))
        return suffix

    def projectors(self) -> list[ProjectorResult]:
        ret: list[ProjectorResult] = []
        # MEMO: No truncation required for T (only V)
        prefix = self._prefix()
        suffix = self._suffix()
        for lp, s in enumerate(self.ss, start=1):
            iw = self._safe_isqrt(s)
            rank = iw.size
            p = suffix[self.n - lp].conj()[:, :rank] * iw
            q = prefix[lp][:, :rank] * iw
            ret.append(ProjectorResult(s[:rank], p, q))
        return ret


def _preprocess(ts: Sequence[npt.NDArray[Any]]) -> list[npt.NDArray[Any]]:
    work = np.eye(1)
    lcum = 0.0
    for t in ts:
        work = np.einsum("cbj,ab,cai->ij", t.conj(), work, t, optimize=True)
        norm = np.linalg.norm(work)
        if norm == 0:
            return list(ts)
        work /= norm
        lcum += math.log(norm)
    lcum += math.log(np.real(np.trace(work)))
    lcum /= 2 * len(ts)
    norms = np.asarray([np.linalg.norm(t) for t in ts])
    co = stats.gmean(norms) / math.exp(lcum)
    ret: list[npt.NDArray[Any]] = []
    for nt, t in zip(norms.flat, ts, strict=True):
        ret.append(co / nt * t)
    return ret


def projective_svd(ts: Sequence[npt.NDArray[Any]], chi: int | None = None) -> list[ProjectorResult]:
    r"""Compute oblique projections that SVD-canonicalize the input MPS tensors.

    Parameters
    ----------
    ts
        The input MPS tensors.
            - Tensors are given as a sequence of `numpy.ndarray` objects.
            - At least two tensors are required.
            - ``ts[i].ndim`` must be ``2`` for ``i == 0`` and ``i == len(ts) - 1``, and must be ``3`` otherwise.
            - Tensors are indexed as ``ts[i][open, minus, plus]`` where ``open`` is the open leg, ``minus`` is the closed leg tied to ``ts[i - 1]``, and ``plus`` is tied to ``ts[i + 1]``. ``minus`` or ``plus`` are ignored in the edge tensors.
            - All the closed legs must have coherent dimensions.
    chi
        The maximum bond dimension. If `None`, treated as infinity. See below for the details.

    Returns
    -------
    projectors
        ``projectors[i]`` corresponds to the projection to be placed at the bond between ``ts[i]`` and ``ts[i + 1]``.

    Notes
    -----
    When ``projectors[i..]`` is applied to the MPS, the result is SVD-canonicalized with the canonical center at the bond between ``ts[i]`` and ``ts[i + 1]``.
    ``projectors[(i + 1)..]`` can be truncated up to bond dimension ``chi`` without changing the final contraction result.
    """  # noqa: E501
    ts_3 = _preprocess(_attach_dummy(ts))
    mps = _CanonicalMPS.from_ts(ts_3, chi)
    chi = mps.chi
    projectors: list[ProjectorResult] = []
    spq = mps.projectors()
    for s, p, q in spq:
        d, _ = p.shape
        projectors.append(ProjectorResult(np.pad(s, (0, d - s.size)), *projector.extend(p, q)))
    return projectors
