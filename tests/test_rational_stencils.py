import numpy as np
import pytest
from stencilpal import conservative_interpolation_stencil, uniform_quadrature


@pytest.mark.parametrize("p", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
@pytest.mark.parametrize("pos", ["l", "c", "r"])
def test_rational_float_equivalence_for_conservative_interpolation(p: int, pos: str):
    rational_stencil = conservative_interpolation_stencil(p, pos, rational=True)
    float_stencil = conservative_interpolation_stencil(p, pos, rational=False)
    assert np.abs(np.sum(rational_stencil.w) - np.sum(float_stencil.w)) < 1e-14


@pytest.mark.parametrize("p", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
def test_rational_float_equivalance_for_uniform_quadrature(p: int):
    rational_quadrature = uniform_quadrature(p, rational=True)
    float_quadrature = uniform_quadrature(p, rational=False)
    assert np.abs(np.sum(rational_quadrature.w) - np.sum(float_quadrature.w)) < 1e-14
