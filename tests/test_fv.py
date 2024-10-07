import numpy as np
from stencilpal import conservative_interpolation_stencil, uniform_quadrature


def test_known_conservative_interpolation_stencil_degree_3():
    left = conservative_interpolation_stencil(3, "l")
    left.weights_as_ints()
    assert np.array_equal(left.w, np.array([-1, 10, 20, -6, 1]))

    center = conservative_interpolation_stencil(3, "c")
    center.weights_as_ints()
    assert np.array_equal(center.w, np.array([-1, 26, -1]))

    right = conservative_interpolation_stencil(3, "r")
    right.weights_as_ints()
    assert np.array_equal(right.w, np.array([1, -6, 20, 10, -1]))


def test_known_conservative_interpolation_stencil_degree_4():
    left = conservative_interpolation_stencil(4, "l")
    left.weights_as_ints()
    assert np.array_equal(left.w, np.array([-3, 27, 47, -13, 2]))

    center = conservative_interpolation_stencil(4, "c")
    center.weights_as_ints()
    assert np.array_equal(center.w, np.array([9, -116, 2134, -116, 9]))

    right = conservative_interpolation_stencil(4, "r")
    right.weights_as_ints()
    assert np.array_equal(right.w, np.array([2, -13, 47, 27, -3]))


def test_known_uniform_quadrature_degree_3():
    stencil = uniform_quadrature(3)
    stencil.weights_as_ints()
    assert np.array_equal(stencil.w, np.array([1, 22, 1]))


def test_known_uniform_quadrature_degree_4():
    stencil = uniform_quadrature(4)
    stencil.weights_as_ints()
    assert np.array_equal(stencil.w, np.array([-17, 308, 5178, 308, -17]))
