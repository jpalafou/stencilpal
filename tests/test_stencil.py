import pytest
import numpy as np
from stencilpal.stencil import Stencil
from stencilpal.rational import RationalArray

# ---- Test Initialization ---- #


def test_empty_stencil():
    stencil = Stencil()
    assert stencil.size == 0
    assert repr(stencil) == "Stencil(empty)"


def test_valid_initialization_with_np_array():
    x = np.array([0, 1, 2])
    w = np.array([1.0, 2.0, 3.0])
    stencil = Stencil(x, w)
    assert stencil.size == 3
    assert np.array_equal(stencil.x, x)
    assert np.array_equal(stencil.w, w)
    assert repr(stencil) == f"Stencil({stencil.to_dict()})"


def test_valid_initialization_with_rational_array():
    x = np.array([0, 1, 2])
    w = RationalArray(np.array([1, 2, 3]), np.array([1, 1, 1]))
    stencil = Stencil(x, w)
    assert stencil.size == 3
    assert np.array_equal(stencil.x, x)
    assert np.array_equal(stencil.w.numerator, w.numerator)
    assert repr(stencil) == f"Stencil({stencil.to_dict()})"


def test_invalid_initialization_mismatched_x_and_w():
    x = np.array([0, 1])
    w = np.array([1.0])
    with pytest.raises(ValueError):
        Stencil(x, w)


def test_invalid_initialization_non_unique_x():
    x = np.array([0, 1, 1])
    w = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        Stencil(x, w)


# ---- Test Node Addition/Removal ---- #


def test_add_node_to_stencil_with_np_array():
    x = np.array([0, 1])
    w = np.array([1.0, 2.0])
    stencil = Stencil(x, w)
    stencil.add_node(2, 3.0)
    assert np.array_equal(stencil.x, np.array([0, 1, 2]))
    assert np.array_equal(stencil.w, np.array([1.0, 2.0, 3.0]))


def test_add_node_to_empty_stencil():
    stencil = Stencil()
    stencil.add_node(0, 1.0)
    assert np.array_equal(stencil.x, np.array([0]))
    assert np.array_equal(stencil.w, np.array([1.0]))


def test_remove_node_from_stencil():
    x = np.array([0, 1, 2])
    w = np.array([1.0, 2.0, 3.0])
    stencil = Stencil(x, w)
    stencil.rm_node(1)
    assert np.array_equal(stencil.x, np.array([0, 1, 2]))
    assert np.array_equal(stencil.w, np.array([1.0, 0.0, 3.0]))


def test_remove_nonexistent_node():
    x = np.array([0, 1, 2])
    w = np.array([1.0, 2.0, 3.0])
    stencil = Stencil(x, w)
    with pytest.raises(ValueError):
        stencil.rm_node(3)


# ---- Test Arithmetic Operations ---- #


def test_negate_stencil():
    x = np.array([0, 1, 2])
    w = np.array([1.0, -2.0, 3.0])
    stencil = Stencil(x, w)
    neg_stencil = -stencil
    assert np.array_equal(neg_stencil.x, x)
    assert np.array_equal(neg_stencil.w, -w)


def test_multiply_stencil_by_scalar():
    x = np.array([0, 1, 2])
    w = np.array([1.0, -2.0, 3.0])
    stencil = Stencil(x, w)
    mul_stencil = stencil * 2
    assert np.array_equal(mul_stencil.x, x)
    assert np.array_equal(mul_stencil.w, w * 2)


def test_add_stencils():
    x1 = np.array([0, 1])
    w1 = np.array([1.0, 2.0])
    stencil1 = Stencil(x1, w1)

    x2 = np.array([1, 2])
    w2 = np.array([3.0, 4.0])
    stencil2 = Stencil(x2, w2)

    result_stencil = stencil1 + stencil2

    assert np.array_equal(result_stencil.x, np.array([0, 1, 2]))
    assert np.array_equal(result_stencil.w, np.array([1.0, 5.0, 4.0]))


def test_subtract_stencils():
    x1 = np.array([0, 1])
    w1 = np.array([1.0, 2.0])
    stencil1 = Stencil(x1, w1)

    x2 = np.array([1, 2])
    w2 = np.array([3.0, 4.0])
    stencil2 = Stencil(x2, w2)

    result_stencil = stencil1 - stencil2

    assert np.array_equal(result_stencil.x, np.array([0, 1, 2]))
    assert np.array_equal(result_stencil.w, np.array([1.0, -1.0, -4.0]))


# ---- Test Weight Type Conversion ---- #


def test_convert_weights_to_rationals():
    x = np.array([-2, -1, 0, 1, 2])
    w = np.array([-1, 10, 20, -6, 1])
    stencil = Stencil(x, w)
    stencil.weights_as_rationals()
    assert isinstance(stencil.w, RationalArray)
    assert np.all(stencil.w == RationalArray([-1, 5, 5, -1, 1], [24, 12, 6, 4, 24]))


def test_convert_weights_to_ints():
    x = np.array([-2, -1, 0, 1, 2])
    w = RationalArray([-1, 5, 5, -1, 1], [24, 12, 6, 4, 24])
    stencil = Stencil(x, w)
    stencil.weights_as_ints()
    assert isinstance(stencil.w, np.ndarray)
    assert np.array_equal(stencil.w, np.array([-1, 10, 20, -6, 1]))
