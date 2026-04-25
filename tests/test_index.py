from __future__ import annotations

import pytest

from trg_utils import _index


class TestNormalize:
    def test_oob(self) -> None:
        with pytest.raises(ValueError, match=r"out of range"):
            _index.normalize(2, 3)

        with pytest.raises(ValueError, match=r"out of range"):
            _index.normalize(2, -100)

        with pytest.raises(ValueError, match=r"out of range"):
            _index.normalize(2, (0, 3))


class TestSpan:
    def test_empty(self) -> None:
        with pytest.raises(ValueError, match=r"must not be empty"):
            _index.assert_span(3, (), (0, 1, 2))

    @pytest.mark.parametrize(
        "inds",
        [
            (0, (1,)),
            ((0, 1), (1, 2)),
            ((0, 1, 1), (2,)),
        ],
    )
    def test_invalid(self, inds: tuple[int | tuple[int, ...], ...]) -> None:
        with pytest.raises(ValueError, match=r"must cover all axes without overlap"):
            _index.assert_span(3, *inds)


class TestAllUnique:
    def test_duplicate(self) -> None:
        with pytest.raises(ValueError, match=r"must be unique"):
            _index.assert_allunique(0, (1, 1))


class TestPShapes:
    def test_inconsistent(self) -> None:
        with pytest.raises(ValueError, match=r"Inconsistent"):
            _index.assert_pshapes((9, 1), (9, 2))

    def test_not_matrix(self) -> None:
        with pytest.raises(ValueError, match=r"matrices"):
            _index.assert_pshapes((9,), (9,))

    def test_empty_original(self) -> None:
        with pytest.raises(ValueError, match=r"non-empty"):
            _index.assert_pshapes((0, 9), (0, 9), allow_empty=True)

    def test_too_wide(self) -> None:
        with pytest.raises(ValueError, match=r"larger"):
            _index.assert_pshapes((4, 9), (4, 9), allow_empty=True)

    def test_empty_ng(self) -> None:
        with pytest.raises(ValueError, match=r"non-empty"):
            _index.assert_pshapes((9, 0), (9, 0))

    def test_empty_ok(self) -> None:
        _index.assert_pshapes((9, 0), (9, 0), allow_empty=True)
