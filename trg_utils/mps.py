"""Optimize MPS by locally-optimal global SVDs.

Thoroughout this module, MPSs are assumed to have these conventions:

- Tensors are given as a sequence of `numpy.ndarray` objects (denoted as ``ts``).
- At least two tensors are required.
- ``ts[i].ndim`` must be ``2`` for ``i == 0`` and ``i == len(ts) - 1``, and must be ``3`` otherwise.
- Tensors are indexed as ``ts[i][open, minus, plus]`` where ``open`` is the open leg, \
  ``minus`` is the closed leg tied to ``ts[i - 1]``, and ``plus`` is tied to ``ts[i + 1]``. \
  ``minus`` or ``plus`` are ignored in the edge tensors.
- All the closed legs must have coherent dimensions.
"""

from __future__ import annotations

import copy
import dataclasses
import itertools
from collections.abc import Sequence
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt

import trg_utils
from trg_utils import projector

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


def _detach_dummy(ts: Sequence[npt.NDArray[_T]]) -> list[npt.NDArray[_T]]:
    head, *mid, tail = ts
    return [
        head[:, 0, :],
        *mid,
        tail[:, :, 0],
    ]


@dataclasses.dataclass
class _CanonicalMPS:
    """Left/right canonical MPS with dummy legs."""

    ts: list[npt.NDArray[Any]]
    chi: int | None
    ss: list[npt.NDArray[Any]] = dataclasses.field(default_factory=list)
    us: list[npt.NDArray[Any]] = dataclasses.field(default_factory=list)
    vs: list[npt.NDArray[Any]] = dataclasses.field(default_factory=list)
    # Gauge transformation applied just after U
    gauge: list[npt.NDArray[Any]] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        assert self.n >= 2
        assert self.chi is None or self.chi > 0

    def _trunc(self, s: npt.NDArray[Any]) -> npt.NDArray[Any]:
        if self.chi is not None:
            s = s.copy()
            s[self.chi :] = 0
        return s

    @property
    def n(self) -> int:
        return len(self.ts)

    def _forward(self) -> npt.NDArray[Any]:
        work = self.ts[0]
        s: npt.NDArray[Any] | None = None
        for i in range(self.n - 1):
            u, s, v = trg_utils.tsvd(work, (0, 1), (2,))
            self.us.append(u)
            work = np.einsum("bj,ij,aic->abc", np.diag(s), v, self.ts[i + 1])
        return work

    def _backward(self, work: npt.NDArray[Any]) -> None:
        for i in reversed(range(self.n - 1)):
            gauge, s, v = trg_utils.tsvd(work, (1,), (0, 2))
            self.gauge.append(gauge)
            self.ss.append(s)  # Store the raw singular values
            self.vs.append(v.transpose(0, 2, 1))  # Adjust leg order
            work = np.einsum("abi,ij,jc->abc", self.us[i], gauge, np.diag(self._trunc(s)))
        self.gauge.reverse()
        self.ss.reverse()
        self.vs.reverse()

    @staticmethod
    def from_ts(ts_3: Sequence[npt.NDArray[Any]], chi: int | None = None) -> _CanonicalMPS:
        ts_3 = [t.copy() for t in ts_3]
        mps = _CanonicalMPS(ts_3, chi)
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
    def _isqrt(s: npt.NDArray[Any], rtol: float = 1e-14) -> tuple[int, npt.NDArray[Any]]:
        assert s.ndim == 1
        rank = int(np.count_nonzero(s > rtol * s.max()))
        ret = np.zeros_like(s)
        if rank > 0:
            ret[:rank] = 1 / np.sqrt(s[:rank])
        return rank, ret

    def _prefix(self) -> list[npt.NDArray[Any]]:
        prefix: list[npt.NDArray[Any]] = [np.eye(1)]
        for t, u in zip(self.ts, self.us, strict=False):
            prefix.append(np.einsum("abi,bc,acj->ij", t.conj(), prefix[-1], u))
        for i, g in enumerate(self.gauge, start=1):
            prefix[i] = prefix[i] @ g  # noqa: PLR6104 (shape change required)
        return prefix

    def projectors(self) -> tuple[list[npt.NDArray[Any]], list[npt.NDArray[Any]]]:
        ps: list[npt.NDArray[Any]] = []
        qs: list[npt.NDArray[Any]] = []
        prefix = self._prefix()
        work = np.eye(1)
        suffix = [work]
        for ls, (t, v) in enumerate(zip(reversed(self.ts), reversed(self.vs), strict=False), start=1):
            work = np.einsum("aib,bc,ajc->ij", t.conj(), work, v)
            suffix.append(work)
            rank, iw = self._isqrt(self.ss[-ls])
            if rank > 0:
                iw = np.diag(iw)
                p = (suffix[ls].conj() @ iw)[:, :rank]
                q = (prefix[self.n - ls] @ iw)[:, :rank]
                p, q = projector.extend(p, q)
            else:
                d, _ = suffix[ls].shape
                p = np.eye(d)
                q = np.eye(d)
            ps.append(p)
            qs.append(q)
            if self.chi is not None:
                keep = min(self.chi, rank)  # MEMO: Eliminate the artifacts from parital null-space bases
                p = p[:, :keep]
                q = q[:, :keep]
                work = p.conj() @ q.T @ work
        ps.reverse()
        qs.reverse()
        return ps, qs
