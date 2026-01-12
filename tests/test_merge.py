from __future__ import annotations

import numpy as np
import pytest

from projector_utils import merge


class TestGroup:
    def test_group_ng(self) -> None:
        with pytest.raises(ValueError, match=r".*between 1.*"):
            merge.group(np.eye(3), 0)

    def test_ungroup_ng(self) -> None:
        with pytest.raises(ValueError, match=r".*3 -> \(2, 2\).*"):
            merge.ungroup(np.eye(3), (2, 2))

    @pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
    def test_gug(self, rng: np.random.Generator, k: int) -> None:
        shape = (2, 3, 4, 5, 6)
        arr0 = rng.random(shape)
        arr1 = merge.group(arr0, k)
        arr2 = merge.ungroup(arr1, shape[-k:])
        np.testing.assert_array_equal(arr0, arr2)
        arr3 = merge.group(arr2, k)
        np.testing.assert_array_equal(arr1, arr3)

    @pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
    def test_group_fortran(self, rng: np.random.Generator, k: int) -> None:
        shape = (2, 3, 4, 5, 6)
        arr_c = rng.random(shape)
        arr_f = np.asfortranarray(arr_c)
        np.testing.assert_array_equal(merge.group(arr_c, k), merge.group(arr_f, k))

    @pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
    def test_upgroup_fortran(self, rng: np.random.Generator, k: int) -> None:
        shape = (2, 3, 4, 5, 6)
        arr_c = merge.group(rng.random(shape), k)
        arr_f = np.asfortranarray(arr_c)
        shape_ = shape[-k:]
        np.testing.assert_array_equal(merge.ungroup(arr_c, shape_), merge.ungroup(arr_f, shape_))
