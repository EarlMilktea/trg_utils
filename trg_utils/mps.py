"""MPS optimization."""

from __future__ import annotations

import copy
import dataclasses
import itertools
from collections.abc import Iterable, Sequence
from typing import Any, NamedTuple, TypeVar

import numpy as np
import numpy.typing as npt

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
            msg = "Inconsistent closed leg dimensions."
            raise ValueError(msg)
    return ret


def _detach_dummy(ts: Iterable[npt.NDArray[_T]]) -> list[npt.NDArray[_T]]:
    head, *mid, tail = ts
    return [
        head[:, 0, :],
        *mid,
        tail[:, :, 0],
    ]


class _ProjectorResult(NamedTuple):
    s: npt.NDArray[Any]
    p: npt.NDArray[Any]
    q: npt.NDArray[Any]


_CHI_INF = 10**17


@dataclasses.dataclass
class _CanonicalMPS:
    """Left/right canonical MPS with dummy legs."""

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
            work = np.einsum("b,ib,aic->abc", s, v, self.ts[i + 1])
        return work

    def _backward(self, work: npt.NDArray[Any]) -> None:
        for i in reversed(range(self.n - 1)):
            gauge, s, v = decomp.tsvd(work, (1,), (0, 2))
            self.gauge.append(gauge)
            self.ss.append(s)  # Store the raw singular values
            self.vs.append(v.transpose(0, 2, 1))  # Adjust leg order
            work = np.einsum("abi,ij,j->abj", self.us[i], gauge, self.zerofill(s))
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
            prefix.append(np.einsum("abi,bc,acj->ij", t.conj(), prefix[-1], u))
        for i, g in enumerate(self.gauge, start=1):
            prefix[i] = prefix[i] @ g  # noqa: PLR6104 (shape change required)
        return prefix

    def _suffix(self) -> list[npt.NDArray[Any]]:
        suffix: list[npt.NDArray[Any]] = [np.eye(1)]
        for t, v in zip(reversed(self.ts), reversed(self.vs), strict=False):
            suffix.append(np.einsum("aib,bc,ajc->ij", t.conj(), suffix[-1], v))
        return suffix

    def projectors(self) -> list[_ProjectorResult]:
        ret: list[_ProjectorResult] = []
        # MEMO: No truncation required for T (only V)
        prefix = self._prefix()
        suffix = self._suffix()
        for lp, s in enumerate(self.ss, start=1):
            iw = self._safe_isqrt(s)
            rank = iw.size
            p = suffix[self.n - lp].conj()[:, :rank] * iw
            q = prefix[lp][:, :rank] * iw
            ret.append(_ProjectorResult(s[:rank], p, q))
        return ret


def optimize(
    ts: Sequence[npt.NDArray[Any]], chi: int | None = None
) -> tuple[list[npt.NDArray[Any]], list[_ProjectorResult]]:
    """Optimize MPS by SVD-compatible oblique projection.

    Parameters
    ----------
    ts
        The input MPS tensors.
            - Tensors are given as a sequence of `numpy.ndarray` objects.
            - At least two tensors are required.
            - ``ts[i].ndim`` must be ``2`` for ``i == 0`` and ``i == len(ts) - 1``, and must be ``3`` otherwise.
            - Tensors are indexed as ``ts[i][open, minus, plus]`` where ``open`` is the open leg, \
            ``minus`` is the closed leg tied to ``ts[i - 1]``, and ``plus`` is tied to ``ts[i + 1]``. \
            ``minus`` or ``plus`` are ignored in the edge tensors.
            - All the closed legs must have coherent dimensions.
    chi
        The maximum bond dimension allowed. If `None`, treated as infinity.

    Returns
    -------
    compressed
        The compressed MPS tensors.
        Its closed bond dimensions are truncated to at most ``chi``.
        Its open bond dimensions are unchanged.
    projectors
        Projector information for each closed bond.

    Notes
    -----
    The optimization is performed by SVD canonicalization.
    """
    ts_3 = _attach_dummy(ts)
    mps = _CanonicalMPS.from_ts(ts_3, chi)
    chi = mps.chi
    projectors: list[_ProjectorResult] = []
    spq = mps.projectors()
    for s, p, q in spq:
        d, _ = p.shape
        projectors.append(_ProjectorResult(np.pad(s, (0, d - s.size)), *projector.extend(p, q)))
    dummy: npt.NDArray[Any] = np.eye(1)
    ps = [*(val.p for val in spq), dummy]
    qs = [dummy, *(val.q for val in spq)]
    compressed = [
        np.einsum("iab,aj,bk->ijk", t, q.conj(), p)[:, :chi, :chi] for (t, p, q) in zip(ts_3, ps, qs, strict=True)
    ]
    return _detach_dummy(compressed), projectors
