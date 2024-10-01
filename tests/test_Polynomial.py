import pytest
import numpy as np
from stencilpal.math_utils import FractionArray, Polynomial


# Tests for Polynomial class
def test_polynomial_initialization_with_ndarray():
    # Test initialization with numpy array
    p = Polynomial(np.array([0, 0, 3, 2, 1]))
    assert np.array_equal(p.coeffs, np.array([3, 2, 1]))  # Leading zeros stripped
    assert p.p == 2  # Degree of polynomial


def test_polynomial_initialization_with_fraction_array():
    # Test initialization with FractionArray
    num = np.array([0, 0, 3, 2])
    denom = np.array([1, 1, 1, 1])
    f_arr = FractionArray(num, denom)
    p = Polynomial(f_arr)
    assert p.coeffs == FractionArray(
        np.array([3, 2]), np.array([1, 1])
    )  # Leading zeros stripped
    assert p.p == 1  # Degree of polynomial


def test_zero_polynomial():
    # Test initialization with zero polynomial
    p = Polynomial(np.array([0, 0, 0]))
    assert np.array_equal(p.coeffs, np.array([0]))  # Zero polynomial should be [0]
    assert p.p == 0  # Degree should be 0


def test_polynomial_evaluation_with_ndarray():
    # Test evaluation with numpy array coefficients
    p = Polynomial(np.array([4, 2, -2, 0, 1]))  # 4x^4 + 2x^3 - 2x^2 + 1
    assert p(1) == 5  # p(1) = 4 + 2 - 2 + 1 = 5
    assert p(0) == 1  # p(0) = 1
    assert p(2) == 41  # p(2) = 32 + 16 - 8 + 1 = 41


def test_polynomial_evaluation_with_fraction_array():
    # Test evaluation with FractionArray coefficients
    num = np.array([1, 4])
    denom = np.array([2, 1])
    f_arr = FractionArray(num, denom)
    p = Polynomial(f_arr)  # (1/2)x + 4
    assert FractionArray.all_equal(p(1), FractionArray(9, 2))  # p(1) = 1/2 + 4 = 9/2
    assert FractionArray.all_equal(p(0), FractionArray(4, 1))  # p(0) = 4
    assert FractionArray.all_equal(p(2), FractionArray(5, 1))  # p(2) = (2/2) + 4 = 5


def test_polynomial_differentiation_with_ndarray():
    # Test differentiation with numpy array
    p = Polynomial(np.array([4, 2, -2, 0, 1]))  # 4x^4 + 2x^3 - 2x^2 + 1
    dp = p.differentiate()  # should return 16x^3 + 6x^2 - 4x
    assert np.array_equal(dp.coeffs, np.array([16, 6, -4, 0]))
    assert dp.p == 3  # Degree of derivative


@pytest.mark.parametrize("simplify", [False, True])
def test_polynomial_differentiation_with_fraction_array(simplify: bool):
    # Test differentiation with FractionArray
    num = np.array([4, 2, -2, 0, 1])
    denom = 2
    f_arr = FractionArray(num, denom)
    p = Polynomial(f_arr)  # (4/2)x^4 + (2/2)x^3 - (2/2)x^2 + 1/2
    dp = p.differentiate()  # should return (16/2)x^3 + (6/2)x^2 - (4/2)x
    if simplify:
        dp.simplify_coeffs()
        assert FractionArray.all_equal(
            dp.coeffs, FractionArray(np.array([8, 3, -2, 0]), np.array([1, 1, 1, 1]))
        )
    else:
        assert FractionArray.all_equal(
            dp.coeffs, FractionArray(np.array([16, 6, -4, 0]), np.array([2, 2, 2, 2]))
        )
    assert dp.p == 3  # Degree of derivative


@pytest.mark.parametrize("integration_constant", [0, 1, 2, 3])
def test_polynomial_antidifferentiate_with_ndarray(integration_constant: int):
    # Test antidifferentiation with numpy array
    p = Polynomial(np.array([4, 2, -2, 0, 1]))  # 4x^4 + 2x^3 - 2x^2 + 1
    ap = p.antidifferentiate(
        integration_constant
    )  # Should return (4/5)x^5 + (2/4)x^4 - (2/3)x^3 + (1/1)x + C
    print(ap.coeffs, type(ap.coeffs))
    assert FractionArray.all_equal(
        ap.coeffs,
        FractionArray(
            np.array([4, 2, -2, 0, 1, integration_constant]),
            np.array([5, 4, 3, 1, 1, 1]),
        ),
    )
    assert ap.p == 3  # Degree should be increased by 1


def test_polynomial_antidifferentiate_with_fraction_array():
    # Test antidifferentiation with FractionArray
    num = np.array([3, 2, 1])
    denom = np.array([1, 1, 1])
    f_arr = FractionArray(num, denom)
    p = Polynomial(f_arr)  # 3x^2 + 2x + 1
    ap = p.antidifferentiate()  # Should return x^3 + x^2 + x + C
    assert ap.coeffs.numerator[0] == 1  # Leading term should be 1/3
    assert ap.p == 3  # Degree should be increased by 1


def test_polynomial_with_leading_zero():
    # Test leading zeros in the polynomial coefficients
    p = Polynomial(np.array([0, 0, 0, 1, 2, 3]))
    assert np.array_equal(p.coeffs, np.array([1, 2, 3]))  # Leading zeros stripped


def test_polynomial_invalid_input():
    # Test invalid input for Polynomial class
    with pytest.raises(ValueError):
        Polynomial(np.array([]))  # Empty array

    with pytest.raises(ValueError):
        Polynomial(np.array([1, 2]).reshape(2, 1))  # Not 1D array

    with pytest.raises(ValueError):
        Polynomial(np.array([0]))  # Only zero coefficient should be allowed


def test_zero_polynomial_evaluation():
    # Test evaluation of zero polynomial
    p = Polynomial(np.array([0]))
    assert p(1) == 0  # Zero polynomial should evaluate to 0 for all inputs
    assert p(100) == 0
