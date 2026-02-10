"""Simple utilities for working with tensor renormalization groups.

This package provides various utilities for tensor manipulations, which are slightly beyond basic NumPy operations.
"""

from __future__ import annotations

from trg_utils.decomp import tqr, tsvd
from trg_utils.merge import _group_impl, _ungroup_impl

__all__ = ["_group_impl", "tqr", "tsvd", "_ungroup_impl"]
