import numpy as np
import pytest

from stencilpal import conservative_interpolation_stencil


@pytest.mark.parametrize("p", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
@pytest.mark.parametrize("pos", ["l", "c", "r"])
def test_rational_float_equivalence_for_conservative_interpolation(p: int, pos: str):
    rational_stencil = conservative_interpolation_stencil(p, pos)
    float_stencil = conservative_interpolation_stencil(
        p, {"l": -1.0, "c": 0.0, "r": 1.0}[pos]
    )
    assert np.abs(np.sum(rational_stencil.w) - np.sum(float_stencil.w)) < 1e-14
