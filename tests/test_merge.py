from __future__ import annotations

import math

import numpy as np
import pytest

from projector_utils import merge


class TestGroup:
    @pytest.mark.parametrize(("begin", "end"), [(-4, 1), (2, 2), (0, 4)])
    def test_group_ng(self, begin: int, end: int) -> None:
        with pytest.raises(ValueError, match=r"begin and end must satisfy"):
            merge.group(np.zeros((2, 3, 4)), begin, end)

    @pytest.mark.parametrize("target", [-4, 3])
    def test_ungroup_ng_dim(self, target: int) -> None:
        with pytest.raises(ValueError, match=r"target must be between"):
            merge.ungroup(np.zeros((2, 3, 4)), target, (2, 2))

    def test_ungroup_ng_shape(self) -> None:
        with pytest.raises(ValueError, match=r"Cannot ungroup: 4 -> \(2, 3\)\."):
            merge.ungroup(np.zeros((2, 3, 4)), 2, (2, 3))

    @pytest.mark.parametrize(
        ("begin", "end"),
        [(0, 2), (1, 4), (-3, -1), (4, 5), (0, 1), (0, 5)],
    )
    def test_group(self, begin: int, end: int) -> None:
        arr = np.arange(120).reshape(1, 2, 3, 4, 5)
        res = merge.group(arr, begin, end)
        eshape: list[int] = list(arr.shape)
        begin = begin + arr.ndim if begin < 0 else begin
        end = end + arr.ndim if end < 0 else end
        eshape[begin:end] = [math.prod(arr.shape[begin:end])]
        assert res.shape == tuple(eshape)
        np.testing.assert_array_equal(res.ravel(), arr.ravel())

    @pytest.mark.parametrize(
        ("target", "split"),
        [(0, (1, 1)), (1, (1, 2)), (2, (3,)), (3, (2, 1, 2)), (4, (5,)), (-1, (1, 5))],
    )
    def test_ungroup(self, target: int, split: tuple[int, ...]) -> None:
        arr = np.arange(120).reshape(1, 2, 3, 4, 5)
        res = merge.ungroup(arr, target, split)
        eshape: list[int] = list(arr.shape)
        target = target + arr.ndim if target < 0 else target
        eshape[target : target + 1] = list(split)
        assert res.shape == tuple(eshape)
        np.testing.assert_array_equal(res.ravel(), arr.ravel())
