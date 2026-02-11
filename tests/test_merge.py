from __future__ import annotations

import math

import numpy as np
import pytest

from trg_utils import merge


class TestGroup:
    def test_ng_dim(self) -> None:
        with pytest.raises(ValueError, match=r"out of range"):
            merge.group(np.zeros((2, 3, 4)), (0, (10, 11)))

        with pytest.raises(ValueError, match=r"out of range"):
            merge.group(np.zeros((2, 3, 4)), (-10, (1, 2)))

    def test_ng_empty(self) -> None:
        with pytest.raises(ValueError, match=r"must not be empty"):
            merge.group(np.zeros((2, 3, 4)), (0, ()))

    def test_ng_deficit(self) -> None:
        with pytest.raises(ValueError, match=r"must cover all axes"):
            merge.group(np.zeros((2, 3, 4)), (0, (1,)))

    def test_ng_overlap(self) -> None:
        with pytest.raises(ValueError, match=r"without overlap"):
            merge.group(np.zeros((2, 3, 4)), (0, (1, 2), 2))

    @pytest.mark.parametrize(
        "instr",
        [
            (0, 1, 2, 3, 4),
            (0, (1, 2), 3, 4),
            ((0, 1), (2, 3, 4)),
            ((0,), 1, (2, 3), (4,)),
            ((0, 1, 2, 3, 4),),
            (-5, 1, -3, 3, -1),
        ],
    )
    def test_noperm(self, instr: tuple[int, tuple[int, ...]]) -> None:
        arr = np.arange(120).reshape(1, 2, 3, 4, 5)
        res = merge.group(arr, instr)
        np.testing.assert_array_equal(res.ravel(), arr.ravel())

    @pytest.mark.parametrize(
        "instr",
        [
            (4, 3, 2, 1, 0),
            (4, (3, 2), 1, 0),
            ((4, 3), (2, 1, 0)),
            ((4,), 3, (2, 1), (0,)),
            ((4, 3, 2, 1, 0),),
            (-1, -3, -2, -4, -5),
        ],
    )
    def test_arbitrary(self, instr: tuple[int, tuple[int, ...]]) -> None:
        arr = np.arange(120).reshape(1, 2, 3, 4, 5)
        res = merge.group(arr, instr)
        eshape = tuple(
            math.prod(arr.shape[i] for i in ind) if isinstance(ind, tuple) else arr.shape[ind] for ind in instr
        )
        assert res.shape == eshape


class TestUngroup:
    @pytest.mark.parametrize("target", [-4, 3])
    def test_ng_dim(self, target: int) -> None:
        with pytest.raises(ValueError, match=r"out of range"):
            merge.ungroup(np.zeros((2, 3, 4)), (target, (2, 2)))

    def test_ng_shape(self) -> None:
        with pytest.raises(ValueError, match=r"Cannot ungroup: 4 -> \(2, 3\)\."):
            merge.ungroup(np.zeros((2, 3, 4)), (2, (2, 3)))

    def test_ng_overlap(self) -> None:
        with pytest.raises(ValueError, match=r"must not overlap"):
            merge.ungroup(np.zeros((2, 3, 4)), (2, (2, 2)), (2, (1, 4)))

    @pytest.mark.parametrize(
        ("target", "split"),
        [(0, (1, 1)), (1, (1, 2)), (2, (3,)), (3, (2, 1, 2)), (4, (5,)), (-1, (1, 5))],
    )
    def test_single(self, target: int, split: tuple[int, ...]) -> None:
        arr = np.arange(120).reshape(1, 2, 3, 4, 5)
        res = merge.ungroup(arr, (target, split))
        eshape: list[int] = list(arr.shape)
        target = target + arr.ndim if target < 0 else target
        eshape[target : target + 1] = list(split)
        assert res.shape == tuple(eshape)
        np.testing.assert_array_equal(res.ravel(), arr.ravel())

    def test_multiple(self) -> None:
        arr = np.arange(120).reshape(1, 2, 3, 4, 5)
        op1 = (1, (1, 2))
        op2 = (3, (2, 2))
        res = merge.ungroup(arr, op1, op2)
        assert res.shape == (1, 1, 2, 3, 2, 2, 5)
        res_ = merge.ungroup(arr, op2, op1)
        np.testing.assert_array_equal(res, res_)
