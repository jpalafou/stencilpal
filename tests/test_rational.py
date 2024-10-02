import numpy as np
import pytest
from stencilpal.rational import RationalArray


def test_rationalarray_init_int():
    ra = RationalArray(3, 4)
    assert ra.numerator == np.array([3])
    assert ra.denominator == np.array([4])


def test_rationalarray_init_array():
    ra = RationalArray(np.array([1, 2]), np.array([3, 4]))
    assert np.array_equal(ra.numerator, np.array([1, 2]))
    assert np.array_equal(ra.denominator, np.array([3, 4]))


def test_rationalarray_invalid_denominator_zero():
    with pytest.raises(ValueError, match="Denominator elements cannot be 0."):
        RationalArray(np.array([1, 2]), np.array([0, 4]))


def test_rationalarray_invalid_numerator_dtype():
    with pytest.raises(ValueError, match="Numerator must be of integer type."):
        RationalArray(np.array([1.5, 2.0]), np.array([3, 4]))


def test_rationalarray_invalid_shape_mismatch():
    with pytest.raises(
        ValueError, match="Numerator and denominator must have the same shape."
    ):
        RationalArray(np.array([1, 2]), np.array([3]))


def test_rationalarray_simplify():
    ra = RationalArray(np.array([6, 8]), np.array([9, 12]))
    simplified = ra.simplify()
    assert np.array_equal(simplified.numerator, np.array([2, 2]))
    assert np.array_equal(simplified.denominator, np.array([3, 3]))


def test_rationalarray_form_common_denominator():
    ra = RationalArray(np.array([1, 1]), np.array([2, 3]))
    result = ra.form_common_denominator()
    assert np.array_equal(result.numerator, np.array([3, 2]))
    assert np.array_equal(result.denominator, np.array([6, 6]))


def test_rationalarray_add():
    """
    1/3 + 2/4 = 5/6
    2/4 + 3/5 = 11/10
    """
    ra1 = RationalArray(np.array([1, 2]), np.array([3, 4]))
    ra2 = RationalArray(np.array([2, 3]), np.array([4, 5]))
    result = ra1 + ra2
    print(result)
    assert np.array_equal(result.numerator, np.array([5, 11]))
    assert np.array_equal(result.denominator, np.array([6, 10]))


def test_rationalarray_sub():
    """
    1/3 - 2/4 = -1/6
    2/4 - 3/5 = -1/10
    """
    ra1 = RationalArray(np.array([1, 2]), np.array([3, 4]))
    ra2 = RationalArray(np.array([2, 3]), np.array([4, 5]))
    result = ra1 - ra2
    assert np.array_equal(result.numerator, np.array([-1, -1]))
    assert np.array_equal(result.denominator, np.array([6, 10]))


def test_rationalarray_mul():
    """
    1/3 * 2/4 = 1/6
    2/4 * 3/5 = 3/10
    """
    ra1 = RationalArray(np.array([1, 2]), np.array([3, 4]))
    ra2 = RationalArray(np.array([2, 3]), np.array([4, 5]))
    result = ra1 * ra2
    assert np.array_equal(result.numerator, np.array([1, 3]))
    assert np.array_equal(result.denominator, np.array([6, 10]))


def test_rationalarray_div():
    """
    1/3 / 2/4 = 2/3
    2/4 / 3/5 = 5/6
    """
    ra1 = RationalArray(np.array([1, 2]), np.array([3, 4]))
    ra2 = RationalArray(np.array([2, 3]), np.array([4, 5]))
    result = ra1 / ra2
    assert np.array_equal(result.numerator, np.array([2, 5]))
    assert np.array_equal(result.denominator, np.array([3, 6]))


def test_rationalarray_negative():
    ra = RationalArray(np.array([1, 2]), np.array([3, 4]))
    result = -ra
    assert np.array_equal(result.numerator, np.array([-1, -2]))
    assert np.array_equal(result.denominator, np.array([3, 4]))


def test_rationalarray_reciprocal():
    ra = RationalArray(np.array([1, 2]), np.array([3, 4]))
    result = ra.reciprocal()
    assert np.array_equal(result.numerator, np.array([3, 4]))
    assert np.array_equal(result.denominator, np.array([1, 2]))


def test_rationalarray_asnumpy():
    ra = RationalArray(np.array([1, 2]), np.array([3, 4]))
    result = ra.asnumpy()
    assert np.allclose(result, np.array([1 / 3, 2 / 4]))


def test_rationalarray_numpy_append():
    ra1 = RationalArray(np.array([1, 2]), np.array([3, 4]))
    ra2 = RationalArray(np.array([2, 3]), np.array([4, 5]))
    result = np.append(ra1, ra2)
    assert np.array_equal(result.numerator, np.array([1, 2, 2, 3]))
    assert np.array_equal(result.denominator, np.array([3, 4, 4, 5]))


def test_rationalarray_numpy_mean():
    ra = RationalArray(np.array([1, 2]), np.array([3, 4]))
    result = np.mean(ra)
    assert np.array_equal(result.numerator, np.array([5]))
    assert np.array_equal(result.denominator, np.array([12]))


def test_rationalarray_numpy_sum():
    """
    1/3 + 2/4 = 5/6
    """
    ra = RationalArray(np.array([1, 2]), np.array([3, 4]))
    result = np.sum(ra)
    assert np.array_equal(result.numerator, np.array([5]))
    assert np.array_equal(result.denominator, np.array([6]))


def test_rationalarray_numpy_concatenate():
    ra1 = RationalArray(np.array([1, 2]), np.array([3, 4]))
    ra2 = RationalArray(np.array([2, 3]), np.array([4, 5]))
    result = np.concatenate([ra1, ra2])
    assert np.array_equal(result.numerator, np.array([1, 2, 2, 3]))
    assert np.array_equal(result.denominator, np.array([3, 4, 4, 5]))


def test_rationalarray_numpy_insert():
    ra1 = RationalArray(np.array([1, 2]), np.array([3, 4]))
    ra2 = RationalArray(np.array([2]), np.array([4]))
    result = np.insert(ra1, 1, ra2)
    assert np.array_equal(result.numerator, np.array([1, 2, 2]))
    assert np.array_equal(result.denominator, np.array([3, 4, 4]))


def test_rationalarray_mul_with_inty_numpy_array():
    """
    1/3 * 1 = 1/3
    2/4 * 2 = 1
    """
    ra = RationalArray(np.array([1, 2]), np.array([3, 4]))
    result = ra * np.array([1, 2])
    assert np.array_equal(result.numerator, np.array([1, 1]))
    assert np.array_equal(result.denominator, np.array([3, 1]))


def test_rationalarray_mul_with_floaty_numpy_array():
    """
    1/3 * 1.0 = 1/3
    2/4 * 2.0 = 1
    """
    ra = RationalArray(np.array([1, 2]), np.array([3, 4]))
    result = ra * np.array([1.0, 2.0])
    assert np.array_equal(result, np.array([1 / 3, 1.0]))


def test_rationalarray_mul_with_inty_scalar():
    """
    1/3 * 2 = 2/3
    2/4 * 2 = 1
    """
    ra = RationalArray(np.array([1, 2]), np.array([3, 4]))
    result = ra * 2
    print(result)
    assert np.array_equal(result.numerator, np.array([2, 1]))
    assert np.array_equal(result.denominator, np.array([3, 1]))


def test_rationalarray_mul_with_floaty_scalar():
    """
    1/3 * 2.0 = 2/3
    2/4 * 2.0 = 1
    """
    ra = RationalArray(np.array([1, 2]), np.array([3, 4]))
    result = ra * 2.0
    assert np.array_equal(result, np.array([2 / 3, 1.0]))


@pytest.mark.parametrize(
    "arg2",
    [RationalArray(np.array([2, 3]), np.array([4, 5])), np.array([1.0, 2.0]), 2, 2.0],
)
def test_rationalarray_mul_commutativity(arg2):
    arg1 = RationalArray(np.array([1, 2]), np.array([3, 4]))
    result1 = arg1 * arg2
    result2 = arg2 * arg1
    assert np.all(result1 == result2)
