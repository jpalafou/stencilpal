import numpy as np
import pytest
import rationalpy as rp

from stencilpal.stencil import Stencil, StencilError


# === FIXTURES ===
@pytest.fixture
def empty_stencil():
    return Stencil([], [])


@pytest.fixture
def float_stencil():
    return Stencil([-2, 0, 2], [0.2, 0.5, 0.3])


@pytest.fixture
def rational_stencil():
    x = [-2, 0, 2]
    w = rp.rational_array([1, 3, 1], [5, 5, 10])  # [1/5, 3/5, 1/10]
    return Stencil(x, w)


# === TEST CASES ===


# --- Node Operations ---
def test_add_node_float_stencil(float_stencil):
    float_stencil.add_node(-1, 0.4)
    assert np.all(float_stencil.x == np.array([-2, -1, 0, 2]))
    assert np.all(float_stencil.w == np.array([0.2, 0.4, 0.5, 0.3]))


def test_add_node_rational_stencil(rational_stencil):
    rational_stencil.add_node(-1, rp.rational_array(1, 10))
    assert np.all(rational_stencil.x == np.array([-2, -1, 0, 2]))
    assert np.all(rational_stencil.w == rp.rational_array([1, 1, 3, 1], [5, 10, 5, 10]))


def test_remove_node_float_stencil(float_stencil):
    float_stencil.remove_node(0)
    assert np.all(float_stencil.x == np.array([-2, 2]))
    assert np.all(float_stencil.w == np.array([0.2, 0.3]))


def test_remove_node_rational_stencil(rational_stencil):
    with pytest.raises(
        NotImplementedError, match="Cannot remove node from rational stencil."
    ):
        rational_stencil.remove_node(2)


# --- Rescoping ---
def test_invalid_rescope_of_empty_stencil_with_h(empty_stencil):
    with pytest.raises(
        ValueError,
        match="Cannot rescope stencil with less than 2 positions without specifying x.",
    ):
        empty_stencil.rescope()


def test_trivial_rescope_of_float_stencil_with_h(float_stencil):
    float_stencil.rescope(h=2)
    assert np.all(float_stencil.x == np.array([-2, 0, 2]))
    assert np.all(float_stencil.w == np.array([0.2, 0.5, 0.3]))


def test_trivial_rescope_of_ratinal_stencil_with_h(rational_stencil):
    rational_stencil.rescope(h=2)
    assert np.all(rational_stencil.x == np.array([-2, 0, 2]))
    assert np.all(rational_stencil.w == rp.rational_array([1, 3, 1], [5, 5, 10]))


def test_nontrivial_rescope_of_float_stencil_with_h(float_stencil):
    float_stencil.rescope()
    assert np.all(float_stencil.x == np.array([-2, -1, 0, 1, 2]))
    assert np.all(float_stencil.w == np.array([0.2, 0.0, 0.5, 0.0, 0.3]))


def test_nontrivial_rescope_of_rational_stencil_with_h(rational_stencil):
    rational_stencil.rescope()
    assert np.all(rational_stencil.x == np.array([-2, -1, 0, 1, 2]))
    assert np.all(
        rational_stencil.w == rp.rational_array([1, 0, 3, 0, 1], [5, 1, 5, 1, 10])
    )


def test_rescope_of_empty_stencil_with_x(empty_stencil):
    empty_stencil.rescope(np.array([-2, -1, 0, 1, 2]))
    assert np.all(empty_stencil.x == np.array([-2, -1, 0, 1, 2]))
    assert np.all(empty_stencil.w == 0)


def test_rescope_of_float_stencil_with_x(float_stencil):
    float_stencil.rescope(np.array([-2, -1, 0, 1, 2]))
    assert np.all(float_stencil.x == np.array([-2, -1, 0, 1, 2]))
    assert np.all(float_stencil.w == np.array([0.2, 0.0, 0.5, 0.0, 0.3]))


def test_rescope_of_rational_stencil_with_x(rational_stencil):
    rational_stencil.rescope(np.array([-2, -1, 0, 1, 2]))
    assert np.all(rational_stencil.x == np.array([-2, -1, 0, 1, 2]))
    assert np.all(
        rational_stencil.w == rp.rational_array([1, 0, 3, 0, 1], [5, 1, 5, 1, 10])
    )


def test_invalid_rescope_of_float_stencil_with_x(float_stencil):
    with pytest.raises(
        StencilError,
        match="Rescoping would exclude existing stencil positions with non-zero weights.",
    ):
        float_stencil.rescope(np.array([-1, 0, 1, 2]))


def test_invalid_rescope_of_rational_stencil_with_x(rational_stencil):
    with pytest.raises(
        StencilError,
        match="Rescoping would exclude existing stencil positions with non-zero weights.",
    ):
        rational_stencil.rescope(np.array([-1, 0, 1, 2]))


# --- Copy ---
def test_copy_empty_stencil(empty_stencil):
    stencil_copy = empty_stencil.copy()
    assert len(stencil_copy.x) == 0
    assert len(stencil_copy.w) == 0
    assert stencil_copy is not empty_stencil


def test_copy_float_stencil(float_stencil):
    stencil_copy = float_stencil.copy()
    assert np.all(stencil_copy.x == float_stencil.x)
    assert np.all(stencil_copy.w == float_stencil.w)
    assert stencil_copy is not float_stencil


def test_copy_rational_stencil(rational_stencil):
    stencil_copy = rational_stencil.copy()
    assert np.all(stencil_copy.x == rational_stencil.x)
    assert np.all(stencil_copy.w == rational_stencil.w)
    assert stencil_copy is not rational_stencil


# --- Arithmetic Operations ---
def test_add_rational_stencils():
    stencil1 = Stencil([-1, 1], rp.rational_array([1, 1], [5, 5]))
    stencil2 = Stencil([-2, 0, 2], rp.rational_array([1, 1, 1], [5, 5, 5]))
    result = stencil1 + stencil2
    assert np.all(result.x == np.array([-2, -1, 0, 1, 2]))
    assert np.all(result.w == rp.rational_array(1, 5))


def test_multiply_rational_stencil_with_scalar(rational_stencil):
    result = rational_stencil * rp.rational_array(5, 2)
    assert np.all(result.x == np.array([-2, 0, 2]))
    assert np.all(result.w == rp.rational_array([1, 3, 1], [2, 2, 4]))


def test_negate_rational_stencil(rational_stencil):
    result = -rational_stencil
    assert np.all(result.x == rational_stencil.x)
    assert np.all(result.w == -rational_stencil.w)
