r"""Simple utilities for tensor network renormalization groups.

This package provides various utilities for tensor manipulations, which are slightly beyond basic NumPy operations.

Terminology
-----------
Dual bases
    A pair of matrices :math:`P` and :math:`Q` of shape :math:`(d,\ r)`,
    or their reshaped counterparts of shape :math:`(d_1,\ \ldots,\ d_k,\ r)`.
    Must satisfy :math:`d \geq r` and :math:`Q^\dagger P = E`.
    Can be used to construct an oblique projector of rank :math:`r`.
Oblique projector
    :math:`P Q^\dagger` constructed from dual bases :math:`P` and :math:`Q`.
    Not necessarily Hermitian but idempotent: :math:`P Q^\dagger P Q^\dagger = P Q^\dagger`.
"""

from __future__ import annotations

from trg_utils.decomp import hosvd, tqr, tsvd
from trg_utils.merge import group, ungroup
from trg_utils.mps import optimize
from trg_utils.projector import extend, normalize, refine

__all__ = ["extend", "group", "hosvd", "normalize", "optimize", "refine", "tqr", "tsvd", "ungroup"]
