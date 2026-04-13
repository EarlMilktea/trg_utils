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
        rho = float(np.linalg.svdvals(arr)[0])
        co = 0.25 / max(rho, 1.0)
        return np.eye(d) + co * arr

    return _inner()


def random_mps(n: int) -> SearchStrategy[list[npt.NDArray[np.complex128]]]:
    assert n >= 2

    @st.composite
    def _inner(draw: DrawFn) -> list[npt.NDArray[np.complex128]]:
        idim = draw(hnp.array_shapes(min_dims=n - 1, max_dims=n - 1))
        odim = draw(hnp.array_shapes(min_dims=n, max_dims=n))
        ts = [draw(shaped_f128((odim[0], idim[0])))]
        ts.extend(draw(shaped_f128((odim[i], idim[i - 1], idim[i]))) for i in range(1, n - 1))
        ts.append(draw(shaped_f128((odim[-1], idim[-1]))))
        return ts

    return _inner()
