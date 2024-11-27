import numpy as np

from stencilpal import conservative_interpolation_stencil, uniform_quadrature


def test_known_conservative_interpolation_stencil_degree_3():
    left = conservative_interpolation_stencil(3, "l")
    assert np.array_equal(left.asnumpy(), np.array([-1, 10, 20, -6, 1]))

    center = conservative_interpolation_stencil(3, "c")
    assert np.array_equal(center.asnumpy(), np.array([-1, 26, -1]))

    right = conservative_interpolation_stencil(3, "r")
    assert np.array_equal(right.asnumpy(), np.array([1, -6, 20, 10, -1]))


def test_known_conservative_interpolation_stencil_degree_4():
    left = conservative_interpolation_stencil(4, "l")
    assert np.array_equal(left.asnumpy(), np.array([-3, 27, 47, -13, 2]))

    center = conservative_interpolation_stencil(4, "c")
    assert np.array_equal(center.asnumpy(), np.array([9, -116, 2134, -116, 9]))

    right = conservative_interpolation_stencil(4, "r")
    assert np.array_equal(right.asnumpy(), np.array([2, -13, 47, 27, -3]))


def test_known_uniform_quadrature_degree_3():
    stencil = uniform_quadrature(3)
    assert np.array_equal(stencil.asnumpy(), np.array([1, 22, 1]))


def test_known_uniform_quadrature_degree_4():
    stencil = uniform_quadrature(4)
    assert np.array_equal(stencil.asnumpy(), np.array([-17, 308, 5178, 308, -17]))
