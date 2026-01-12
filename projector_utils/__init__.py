"""Utilities for tensor axis manipulation and decomposition.

This package provides NumPy-based helpers for reshaping tensor axes and
performing common decompositions used in tensor workflows.
"""

from __future__ import annotations

from projector_utils.decomp import tqr, tsvd
from projector_utils.merge import group, ungroup

__all__ = [
    "group",
    "tqr",
    "tsvd",
    "ungroup",
]
