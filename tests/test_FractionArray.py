import numpy as np
from stencilpal.math_utils import FractionArray


def test_fraction_array_init():
    numerators = np.array([3, 4, 5, 6, 7])
    denominators = np.array([9, 12, 15, 18, 21])
    farr = FractionArray(numerators, denominators)
    assert np.all(farr.numerator == numerators)
    assert np.all(farr.denominator == denominators)


def test_fraction_array_init_with_integer_denominator():
    numerators = np.array([2, 4, 6, 8, 10])
    farr = FractionArray(numerators, 5)
    assert np.all(farr.numerator == numerators)
    assert np.all(farr.denominator == np.full_like(numerators, 5))


def test_simplify():
    numerators = np.array([10, 8, 12, 18, 20])
    denominators = np.array([20, 16, 24, 36, 40])
    farr = FractionArray(numerators, denominators)
    simplified_farr = farr.simplify()
    expected_numerators = np.array([1, 1, 1, 1, 1])
    expected_denominators = np.array([2, 2, 2, 2, 2])
    assert np.all(simplified_farr.numerator == expected_numerators)
    assert np.all(simplified_farr.denominator == expected_denominators)


def test_find_common_denominator():
    numerators = np.array([1, 2, 3, 4, 5])
    denominators = np.array([2, 3, 4, 5, 6])
    farr = FractionArray(numerators, denominators)
    common_farr = farr.find_common_denominator()
    expected_denominator = np.lcm.reduce(denominators)
    assert np.all(common_farr.denominator == expected_denominator)


def test_to_numpy():
    numerators = np.array([3, 6, 9, 12, 15])
    denominators = np.array([6, 12, 18, 24, 30])
    farr = FractionArray(numerators, denominators)
    numpy_array = farr.to_numpy()
    expected_array = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    assert np.allclose(numpy_array, expected_array)


def test_get_sum():
    numerators = np.array([1, 2, 3, 4, 5])
    denominators = np.array([2, 3, 4, 5, 6])
    farr = FractionArray(numerators, denominators)
    fraction_sum = farr.get_sum()
    assert fraction_sum == (71, 20)  # Expected sum in fraction form


def test_addition():
    numerators1 = np.array([1, 2, 3, 4, 5])
    denominators1 = np.array([2, 3, 4, 5, 6])
    numerators2 = np.array([1, 1, 1, 1, 1])
    denominators2 = np.array([2, 3, 4, 5, 6])
    farr1 = FractionArray(numerators1, denominators1)
    farr2 = FractionArray(numerators2, denominators2)
    result = farr1 + farr2
    assert np.all(result.numerator == np.array([2, 3, 4, 5, 6]))
    assert np.all(result.denominator == np.array([2, 3, 4, 5, 6]))
    simplified = result.simplify()
    assert np.all(simplified.numerator == np.array([1, 1, 1, 1, 1]))  # Simplified
    assert np.all(simplified.denominator == np.array([1, 1, 1, 1, 1]))


def test_multiplication():
    numerators = np.array([1, 2, 3, 4, 5])
    denominators = np.array([2, 3, 4, 5, 6])
    farr = FractionArray(numerators, denominators)
    result = farr * 2
    assert np.all(result.numerator == numerators * 2)
    assert np.all(result.denominator == denominators)
