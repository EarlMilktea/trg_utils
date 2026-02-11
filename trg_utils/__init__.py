"""Simple utilities for working with tensor renormalization groups.

This package provides various utilities for tensor manipulations, which are slightly beyond basic NumPy operations.
"""

from __future__ import annotations

from trg_utils.decomp import hosvd, tqr, tsvd
from trg_utils.merge import group, ungroup

__all__ = ["group", "hosvd", "tqr", "tsvd", "ungroup"]
