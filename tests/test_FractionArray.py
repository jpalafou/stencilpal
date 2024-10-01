import numpy as np
import pytest
from stencilpal.math_utils import FractionArray


def test_fraction_array_init():
    num = np.array([1, 2, 3])
    denom = np.array([4, 5, 6])
    farr = FractionArray(num, denom)
    assert np.array_equal(farr.numerator, num)
    assert np.array_equal(farr.denominator, denom)


def test_fraction_array_add():
    num1 = np.array([1, 2])
    denom1 = np.array([3, 4])
    num2 = np.array([1, 3])
    denom2 = np.array([2, 5])

    farr1 = FractionArray(num1, denom1)
    farr2 = FractionArray(num2, denom2)

    farr3 = farr1 + farr2

    expected_num = np.array([5, 22])
    expected_denom = np.array([6, 20])

    assert np.array_equal(farr3.numerator, expected_num)
    assert np.array_equal(farr3.denominator, expected_denom)


def test_fraction_array_sub():
    num1 = np.array([3, 5])
    denom1 = np.array([4, 6])
    num2 = np.array([1, 3])
    denom2 = np.array([2, 5])

    farr1 = FractionArray(num1, denom1)
    farr2 = FractionArray(num2, denom2)

    farr3 = farr1 - farr2

    expected_num = np.array([1, 7])
    expected_denom = np.array([4, 30])

    assert np.array_equal(farr3.numerator, expected_num)
    assert np.array_equal(farr3.denominator, expected_denom)


def test_fraction_array_mul():
    num1 = np.array([1, 3])
    denom1 = np.array([4, 7])
    num2 = np.array([2, 5])
    denom2 = np.array([3, 6])

    farr1 = FractionArray(num1, denom1)
    farr2 = FractionArray(num2, denom2)

    farr3 = farr1 * farr2

    expected_num = np.array([2, 15])
    expected_denom = np.array([12, 42])

    assert np.array_equal(farr3.numerator, expected_num)
    assert np.array_equal(farr3.denominator, expected_denom)


def test_fraction_array_div():
    num1 = np.array([3, 8])
    denom1 = np.array([4, 9])
    num2 = np.array([2, 5])
    denom2 = np.array([3, 6])

    farr1 = FractionArray(num1, denom1)
    farr2 = FractionArray(num2, denom2)

    farr3 = farr1 / farr2

    expected_num = np.array([9, 48])
    expected_denom = np.array([8, 45])

    assert np.array_equal(farr3.numerator, expected_num)
    assert np.array_equal(farr3.denominator, expected_denom)


def test_fraction_array_reciprocal():
    num = np.array([2, 3])
    denom = np.array([5, 4])

    farr = FractionArray(num, denom)
    rec_farr = farr.reciprocal()

    assert np.array_equal(rec_farr.numerator, denom)
    assert np.array_equal(rec_farr.denominator, num)


def test_fraction_array_multiply_by_scalar():
    num = np.array([1, 2, 3])
    denom = np.array([4, 5, 6])
    farr = FractionArray(num, denom)

    farr2 = farr * 2

    expected_num = np.array([2, 4, 6])
    expected_denom = np.array([4, 5, 6])

    assert np.array_equal(farr2.numerator, expected_num)
    assert np.array_equal(farr2.denominator, expected_denom)


def test_fraction_array_simplify():
    num = np.array([10, 20])
    denom = np.array([15, 25])
    farr = FractionArray(num, denom)

    simplified_farr = farr.simplify()

    expected_num = np.array([2, 4])
    expected_denom = np.array([3, 5])

    assert np.array_equal(simplified_farr.numerator, expected_num)
    assert np.array_equal(simplified_farr.denominator, expected_denom)


def test_fraction_array_find_common_denominator():
    num = np.array([1, 2])
    denom = np.array([3, 4])
    farr = FractionArray(num, denom)

    common_farr = farr.find_common_denominator()

    expected_num = np.array([4, 6])
    expected_denom = np.array([12, 12])

    assert np.array_equal(common_farr.numerator, expected_num)
    assert np.array_equal(common_farr.denominator, expected_denom)


# Test for get_sum method in FractionArray class
def test_fraction_array_get_sum_with_common_denominator():
    # Test summing fractions with a common denominator
    numerators = np.array([1, 2, 3])
    denominators = np.array([4, 4, 4])
    f_arr = FractionArray(numerators, denominators)
    f_sum = f_arr.get_sum()

    # The expected result is (1+2+3)/4 = 6/4 = 3/2
    assert f_sum.numerator == 3
    assert f_sum.denominator == 2


def test_fraction_array_get_sum_with_different_denominators():
    # Test summing fractions with different denominators
    numerators = np.array([1, 1, 1])
    denominators = np.array([2, 3, 6])
    f_arr = FractionArray(numerators, denominators)
    f_sum = f_arr.get_sum()

    # The expected result is 1/2 + 1/3 + 1/6 = 1
    assert f_sum == FractionArray(1, 1)


def test_fraction_array_get_sum_with_mixed_signs():
    # Test summing fractions with mixed signs
    numerators = np.array([1, -1, 1])
    denominators = np.array([2, 2, 1])
    f_arr = FractionArray(numerators, denominators)
    f_sum = f_arr.get_sum()

    # The expected result is 1/2 - 1/2 + 1 = 1
    assert f_sum == FractionArray(1, 1)


def test_fraction_array_get_sum_with_reduction():
    # Test summing fractions that require reduction
    numerators = np.array([2, 4])
    denominators = np.array([6, 3])
    f_arr = FractionArray(numerators, denominators)
    f_sum = f_arr.get_sum()

    # The expected result is 1/3 + 4/3 = 5/3
    assert f_sum == FractionArray(5, 3)


def test_fraction_array_invalid_inputs():
    # Case 1: Zero denominator
    with pytest.raises(ValueError, match="The denominator cannot be zero."):
        FractionArray(np.array([1, 2]), np.array([0, 0]))

    # Case 2: Non-integer numerator
    with pytest.raises(
        ValueError, match="The numerator and denominator must be integers."
    ):
        FractionArray(np.array([1.5, 2.5]), np.array([3, 4]))

    # Case 3: Non-integer denominator
    with pytest.raises(
        ValueError, match="The numerator and denominator must be integers."
    ):
        FractionArray(np.array([1, 2]), np.array([3.5, 4.5]))

    # Case 4: Mismatched numerator and denominator lengths
    with pytest.raises(
        ValueError, match="The number of numerators and denominators must be the same."
    ):
        FractionArray(np.array([1, 2]), np.array([3]))

    # Case 5: Empty numerator array
    with pytest.raises(
        ValueError, match="The numerator and denominator must be non-empty 1D arrays."
    ):
        FractionArray(np.array([], dtype=int)), np.array([1, 1])

    # Case 6: Empty denominator array
    with pytest.raises(
        ValueError, match="The numerator and denominator must be non-empty 1D arrays."
    ):
        FractionArray(np.array([1, 1]), np.array([], dtype=int))

    # Case 7: Non-1D arrays
    with pytest.raises(
        ValueError, match="The numerator and denominator must be non-empty 1D arrays."
    ):
        FractionArray(np.array([[1, 2]]), np.array([[3, 4]]))
