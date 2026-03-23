from __future__ import annotations

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import numpy.typing as npt
from hypothesis.strategies import DrawFn, SearchStrategy


def shaped_f128(shape: tuple[int, ...]) -> SearchStrategy[npt.NDArray[np.complex128]]:
    return hnp.arrays(
        dtype=np.complex128,
        shape=shape,
        elements=st.complex_numbers(max_magnitude=1),
    )


def almost_diagonal(d: int) -> SearchStrategy[npt.NDArray[np.complex128]]:
    @st.composite
    def _inner(draw: DrawFn) -> npt.NDArray[np.complex128]:
        arr = draw(shaped_f128((d, d)))
        rho, *_ = np.linalg.svdvals(arr).flat
        return np.eye(d) + 0.25 * (arr / max(rho, 1))

    return _inner()
