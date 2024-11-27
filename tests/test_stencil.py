import numpy as np
import pytest
import rationalpy as rp

from stencilpal.stencil import Stencil, StencilError


# === FIXTURES ===
@pytest.fixture
def integer_stencil():
    x = [-2, 0, 2]
    w = [0.2, 0.5, 0.3]
    return Stencil(x, w, h=1)


@pytest.fixture
def rational_stencil():
    x = [-2, 0, 2]
    w = rp.rational_array([1, 3, 1], [5, 5, 10])  # [1/5, 3/5, 1/10]
    return Stencil(x, w, h=1)


# === TEST CASES ===


# --- Validation ---
def test_validation_integer_stencil():
    stencil = Stencil([1, 2, 3], [0.1, 0.2, 0.3])
    assert np.all(stencil.x == np.array([1, 2, 3]))
    assert np.all(stencil.w == np.array([0.1, 0.2, 0.3]))
    assert stencil.h == 1


def test_validation_rational_stencil():
    w = rp.rational_array([1, 2], [3, 3])  # [1/3, 2/3]
    stencil = Stencil([0, 1], w)
    assert isinstance(stencil.w, rp.RationalArray)
    assert stencil.w[0].numerator == 1
    assert stencil.w[0].denominator == 3


def test_validation_errors():
    with pytest.raises(ValueError, match="x must be integers."):
        Stencil([1.0, 2.0], [0.1, 0.2])
    with pytest.raises(ValueError, match="x and w must be 1D arrays."):
        Stencil([[1, 2]], [0.1, 0.2])
    with pytest.raises(ValueError, match="x and w must have the same length."):
        Stencil([1, 2], [0.1])


# --- Rescoping ---
def test_rescope_no_clipping(integer_stencil):
    integer_stencil.rescope(-3, 3)
    assert np.all(integer_stencil.x == np.array([-3, -2, -1, 0, 1, 2, 3]))
    assert np.all(integer_stencil.w == np.array([0, 0.2, 0, 0.5, 0, 0.3, 0]))


def test_rescope_with_different_h(integer_stencil):
    integer_stencil.rescope(-2, 2, h=2)
    assert np.all(integer_stencil.x == np.array([-2, 0, 2]))
    assert np.all(integer_stencil.w == np.array([0.2, 0.5, 0.3]))


def test_rescope_error_clipping(integer_stencil):
    with pytest.raises(
        StencilError, match="Rescoping cannot clip existing stencil positions."
    ):
        integer_stencil.rescope(-1, 1)


def test_rescope_does_not_change_weights_sum(integer_stencil):
    original_sum = np.sum(np.abs(integer_stencil.w))
    integer_stencil.rescope(-4, 4)
    assert np.sum(np.abs(integer_stencil.w)) == original_sum


def test_rescope_rational(rational_stencil):
    rational_stencil.rescope(-3, 3)
    assert np.all(
        rational_stencil.w
        == rp.rational_array([0, 1, 0, 3, 0, 1, 0], [1, 5, 1, 5, 1, 10, 1])
    )


# --- Trimming ---
def test_trim_zeros(integer_stencil):
    integer_stencil.rescope(-3, 3)
    integer_stencil.trim_zeros()
    assert np.all(integer_stencil.x == np.array([-2, 0, 2]))
    assert np.all(integer_stencil.w == np.array([0.2, 0.5, 0.3]))


def test_trim_leading_and_trailing_zeros(integer_stencil):
    integer_stencil.rescope(-3, 3)
    integer_stencil.trim_leading_and_trailing_zeros()
    assert np.all(integer_stencil.x == np.array([-2, -1, 0, 1, 2]))
    assert np.all(integer_stencil.w == np.array([0.2, 0, 0.5, 0, 0.3]))


def test_trim_zeros_error_empty_stencil(integer_stencil):
    integer_stencil.rescope(-3, 3)
    integer_stencil.w.fill(0)
    with pytest.raises(ValueError, match="Cannot trim all zeros."):
        integer_stencil.trim_zeros()


# --- Arithmetic Operations ---
def test_add_stencils(integer_stencil):
    stencil1 = Stencil([-1, 0, 1], [0.2, 0.5, 0.3])
    stencil2 = Stencil([-2, 0, 2], [0.1, 0.4, 0.2])
    result = stencil1 + stencil2
    assert np.all(result.x == np.array([-2, -1, 0, 1, 2]))
    assert np.all(result.w == np.array([0.1, 0.2, 0.9, 0.3, 0.2]))


def test_multiply_stencil_with_scalar(integer_stencil):
    result = integer_stencil * 2
    assert np.all(result.w == np.array([0.4, 1.0, 0.6]))


def test_negate_stencil(integer_stencil):
    result = -integer_stencil
    assert np.all(result.w == np.array([-0.2, -0.5, -0.3]))


# --- Rational Operations ---
def test_simplify_rational_stencil(rational_stencil):
    rational_stencil.simplify()
    assert np.all(rational_stencil.w == rp.rational_array([1, 3, 1], [5, 5, 10]))


def test_common_denominator_rational_stencil(rational_stencil):
    rational_stencil.form_common_denominator()
    numerators = rational_stencil.w.numerator.tolist()
    denominators = rational_stencil.w.denominator.tolist()
    assert numerators == [2, 6, 1]
    assert all(d == 10 for d in denominators)


def test_asnumpy_rational_array_with_mode_numerator(rational_stencil):
    padded_stencil = rational_stencil.rescope(inplace=False)
    result = rational_stencil.asnumpy(mode="numerator")
    assert np.all(result == np.array([2, 0, 6, 0, 1]))
    assert np.all(rp.rational_array(result, 10) == padded_stencil.w)


def test_asnumpy_numpy_array_with_mode_numerator(integer_stencil):
    with pytest.raises(
        ValueError, match="'numerator' mode can only be used with rational stencils."
    ):
        integer_stencil.asnumpy(mode="numerator")


def test_asnumpy_any_array_with_mode_float(integer_stencil, rational_stencil):
    integer_result = integer_stencil.asnumpy(mode="float")
    assert np.all(integer_result == np.array([0.2, 0, 0.5, 0, 0.3]))

    rational_result = rational_stencil.asnumpy(mode="float")
    assert np.all(rational_result == np.array([0.2, 0, 0.6, 0, 0.1]))
