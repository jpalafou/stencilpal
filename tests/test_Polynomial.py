import numpy as np
import pytest
import rationalpy as rp

from stencilpal.polynomial import Polynomial, binomial_product


def test_invalid_polynomial_initialization():
    # Test invalid initialization cases
    with pytest.raises(ValueError):
        Polynomial(np.array([[1, 2], [3, 4]]))  # not a 1D array

    with pytest.raises(ValueError):
        Polynomial(np.array([]))  # empty array

    with pytest.raises(ValueError):
        Polynomial(np.array([1, 2, "a"]))  # non-numeric dtype


def test_polynomial_initialization_with_numpy_array():
    # Test polynomial initialization with numpy array
    coeffs = np.array([1, 2, 3])  # corresponds to 1*x^2 + 2*x + 3
    poly = Polynomial(coeffs)
    assert isinstance(poly.coeffs, np.ndarray)
    assert poly.p == 2
    assert np.array_equal(poly.coeffs, np.array([1, 2, 3]))


def test_polynomial_initialization_with_rational_array():
    # Test polynomial initialization with RationalArray
    coeffs = rp.rational_array([1, 2], [3, 4])  # corresponds to (1/3)*x + (2/4)
    poly = Polynomial(coeffs)
    assert isinstance(poly.coeffs, rp.RationalArray)
    assert poly.p == 1
    assert np.array_equal(poly.coeffs.numerator, np.array([1, 1]))
    assert np.array_equal(poly.coeffs.denominator, np.array([3, 2]))


def test_polynomial_call_with_numpy_coeffs():
    # Test polynomial evaluation with numpy coefficients
    coeffs = np.array([1, 2, 3])  # corresponds to 1*x^2 + 2*x + 3
    poly = Polynomial(coeffs)
    assert poly(1) == 6  # 1*(1^2) + 2*1 + 3 = 6
    assert poly(2) == 11  # 1*(2^2) + 2*2 + 3 = 11


def test_polynomial_call_with_rational_coeffs():
    # Test polynomial evaluation with RationalArray coefficients
    coeffs = rp.rational_array([1, 2], [3, 4])  # corresponds to (1/3)*x + (2/4)
    poly = Polynomial(coeffs)
    assert poly(1) == rp.rational_array(5, 6)  # (1/3)*1 + (2/4) = 5/6
    assert poly(2) == rp.rational_array(7, 6)  # (1/3)*2 + (2/4) = 5/3


def test_polynomial_differentiate_with_numpy_coeffs():
    # Test differentiation with numpy coefficients
    coeffs = np.array([3, 0, 2, 1])  # corresponds to 3*x^3 + 2*x + 1
    poly = Polynomial(coeffs)
    dpdx = poly.differentiate()
    assert dpdx.p == 2
    assert np.array_equal(dpdx.coeffs, np.array([9, 0, 2]))  # derivative: 9*x^2 + 2


def test_polynomial_differentiate_with_rational_coeffs():
    # Test differentiation with RationalArray coefficients
    coeffs = rp.rational_array([3, 0, 2, 1])  # corresponds to 3*x^3 + 2*x + 1
    poly = Polynomial(coeffs)
    dpdx = poly.differentiate()
    assert dpdx.p == 2
    assert np.array_equal(
        dpdx.coeffs.numerator, np.array([9, 0, 2])
    )  # derivative: 9*x^2 + 2
    assert np.all(dpdx.coeffs.denominator == 1)


def test_polynomial_antidifferentiate_with_numpy_coeffs():
    # Test antidifferentiation with numpy coefficients
    coeffs = np.array([4, 3, 2])  # corresponds to 4*x^2 + 3*x + 2
    poly = Polynomial(coeffs)
    antiderivative = poly.antidifferentiate(5)
    assert antiderivative.p == 3
    assert np.array_equal(antiderivative.coeffs.numerator, np.array([4, 3, 2, 5]))
    assert np.array_equal(antiderivative.coeffs.denominator, np.array([3, 2, 1, 1]))


def test_polynomial_antidifferentiate_with_rational_coeffs():
    # Test antidifferentiation with RationalArray coefficients
    coeffs = rp.rational_array([4, 3, 2])  # corresponds to 4*x^2 + 3*x + 2
    poly = Polynomial(coeffs)
    antiderivative = poly.antidifferentiate(5)
    assert antiderivative.p == 3
    assert np.array_equal(antiderivative.coeffs.numerator, np.array([4, 3, 2, 5]))
    assert np.array_equal(antiderivative.coeffs.denominator, np.array([3, 2, 1, 1]))


def test_polynomial_coeff_simplify():
    # Test coefficient simplification for RationalArray
    coeffs = rp.rational_array([4, 2], [6, 4])  # (4/6)*x + (2/4)
    poly = Polynomial(coeffs)
    poly.coeffs_simplify()
    assert np.array_equal(poly.coeffs.numerator, np.array([2, 1]))
    assert np.array_equal(poly.coeffs.denominator, np.array([3, 2]))


def test_polynomial_zero_handling():
    # Test that a zero polynomial is correctly initialized
    coeffs = np.array([0, 0, 0])
    poly = Polynomial(coeffs)
    assert poly.p == 0
    assert np.array_equal(poly.coeffs, np.array([0]))


def test_binomial_product():
    # Test binomial product for the polynomial (x+1)(x+2)
    binomials = np.array([1, 2])
    product_poly = Polynomial(binomial_product(binomials))
    assert product_poly.p == 2
    assert np.array_equal(
        product_poly.coeffs, np.array([1, 3, 2])
    )  # corresponds to x^2 + 3*x + 2
