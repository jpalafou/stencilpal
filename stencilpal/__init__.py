"""
module for computing stencils useful for finite volume computations.

- conservative_interpolation_stencil: compute the symmetric conservative interpolation
    stencil.
- uniform_quadrature: compute the symmetric uniform quadrature stencil.
"""

from numbers import Number
from typing import Union

import numpy as np
import rationalpy as rp

from stencilpal.polynomial import binomial_product
from stencilpal.stencil import Stencil


def _validate_centers(centers: np.ndarray):
    if not isinstance(centers, np.ndarray):
        raise ValueError("centers must be a numpy array")
    if not np.issubdtype(centers.dtype, np.integer):
        raise ValueError("centers must be integers")
    if centers.ndim != 1:
        raise ValueError("centers must be a 1D array")
    if centers.size == 0:
        raise ValueError("centers must have at least one element")
    elif centers.size > 1:
        if not np.all(np.diff(centers) == 2):
            raise ValueError("centers must be consecutive integers spaced by 2")


def _compute_conservative_interpolation_weights(
    centers: np.ndarray, x: Union[str, int, float], rational: bool = True
) -> Stencil:
    """
    compute the conservative interpolation weights for a series of finite-volume cells.

    args:
        centers (np.ndarray): array of cell centers, which must be consecutive integers
            spaced by 2.
        x (Union[str, int, float]): interpolation point on the interval [-1, 1] (the
            cell with center 0).
            - "l": alias for the leftmost point of the cell (-1).
            - "c": alias for the center of the cell (0).
            - "r": alias for the rightmost point of the cell (1).
        rational (bool): if True, return weights as rational numbers; if False, return
            as floating-point numbers.

    returns:
        stencil (Stencil): the computed stencil object containing the interpolation
            weights, normalized to the cell width.
    """

    # validate input
    _validate_centers(centers)
    if not isinstance(x, (str, Number)):
        raise ValueError("x must be a string or number")

    # format x
    if x in ["l", "c", "r"]:
        x = {"l": -1, "c": 0, "r": 1}[x]
    if rational and not np.issubdtype(type(x), np.integer):
        raise ValueError("x must be an integer for rational interpolation")

    # compute interfaces
    interfaces = np.append(centers[0] - 1, centers + 1)

    # preallocate lagrange polynomial evaluations and initialize stencil
    if rational:
        li_prime_x = rp.rational_array(np.zeros_like(interfaces))
    else:
        li_prime_x = np.zeros_like(interfaces, dtype=np.float64)
        interfaces = interfaces.astype(np.float64)

    # construct lagrange polynomials and derivatives
    for i in range(1, len(interfaces)):
        p = binomial_product(-np.delete(interfaces, i))
        li = p / p(interfaces[i])
        li_prime_x[i] = li.differentiate()(x)

    # gather stencil weights as a cumulative sum of the derivative evaluations
    weights = []
    for i in range(len(centers)):
        weights.append(np.sum(li_prime_x[(i + 1) :])[np.newaxis])

    # convert to stencil
    stencil = Stencil(centers // 2, np.concatenate(weights) * 2)
    stencil.trim_leading_and_trailing_zeros()

    return stencil


def _compute_uniform_quadrature_weights(
    centers: np.ndarray, rational: bool = True
) -> Stencil:
    """
    compute the quadrature weights for a series of equally-spaced nodes.

    args:
        centers (np.ndarray): array of node positions, which must be consecutive
            integers spaced by 2.
        rational (bool): if true, return weights as rational numbers; if false, return
            as floating-point numbers.

    returns:
        stencil (Stencil): the computed stencil object containing the quadrature
            weights, normalized to the node spacing.
    """

    # validate input
    _validate_centers(centers)

    # construct lagrange polynomials and antiderivatives and gather weights
    weights = []
    if centers.size > 1:
        for i, x in enumerate(centers):
            p = binomial_product(-np.delete(centers, i))
            if not rational:
                p = p.convert_coeffs_to_numpy()
            li = p / p(centers[i])
            Li = li.antidifferentiate()
            weights.append((Li(1) - Li(-1))[np.newaxis])
    elif centers.size == 1:
        weights.append(np.array([2]))
    else:
        raise ValueError("centers must have at least one element")

    # convert to stencil
    stencil = Stencil(centers // 2, np.concatenate(weights) // 2)
    stencil.trim_leading_and_trailing_zeros()

    return stencil


def conservative_interpolation_stencil(
    p: int, x: Union[str, int, float], rational: bool = True
) -> Stencil:
    """
    compute the symmetric conservative interpolation stencil for a given polynomial
        degree

    args:
        p (int): the polynomial degree.
        x (Union[str, int, float]): interpolation point on the interval [-1, 1].
            - "l": alias for the leftmost point of the cell (-1).
            - "c": alias for the center of the cell (0).
            - "r": alias for the rightmost point of the cell (1).
        rational (bool): if true, return weights as rational numbers; if false, return
            as floating-point numbers.

    returns:
        stencil (Stencil): the computed stencil object containing the interpolation
            weights, normalized to the cell width.
    """

    centers = np.arange(-(-2 * (-p // 2)), -2 * (-p // 2) + 2, 2)
    if p % 2 == 0:
        stencil = _compute_conservative_interpolation_weights(
            centers, x, rational=rational
        )
    else:
        left_biased_stencil = _compute_conservative_interpolation_weights(
            centers[1:], x, rational=rational
        )
        right_biased_stencil = _compute_conservative_interpolation_weights(
            centers[:-1], x, rational=rational
        )
        stencil = (
            rp.rational_array(1, 2) * left_biased_stencil
            + rp.rational_array(1, 2) * right_biased_stencil
        )
    return stencil


def uniform_quadrature(p: int, rational: bool = True) -> Stencil:
    """
    compute the symmetric uniform quadrature stencil for a given polynomial degree

    args:
        p (int): the polynomial degree.
        rational (bool): if true, return weights as rational numbers; if false, return
            as floating-point numbers.

    returns:
        stencil (Stencil): the computed stencil object containing the quadrature
            weights, normalized to the node spacing.
    """

    centers = np.arange(-(-2 * (-(p - 1) // 2)), -2 * (-(p - 1) // 2) + 2, 2)
    stencil = _compute_uniform_quadrature_weights(centers, rational=rational)
    return stencil
