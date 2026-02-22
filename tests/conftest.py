from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


def f128_random(rng: np.random.Generator, shape: tuple[int, ...]) -> npt.NDArray[np.complex128]:
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)
