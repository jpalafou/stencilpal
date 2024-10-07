from numbers import Number
import numpy as np
from stencilpal.polynomial import binomial_product
from stencilpal.rational import _int_like, RationalArray
from stencilpal.stencil import Stencil
from typing import Union


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


def compute_conservative_interpolation_weights(
    centers: np.ndarray, x: Union[str, int, float], rational: bool = True
) -> Stencil:
    # validate input
    _validate_centers(centers)
    if not isinstance(x, (str, Number)):
        raise ValueError("x must be a string or number")

    # format x
    if x in ["l", "c", "r"]:
        x = {"l": -1, "c": 0, "r": 1}[x]
    if rational and not _int_like(x):
        raise ValueError("x must be an integer for rational interpolation")

    # compute interfaces
    interfaces = np.append(centers[0] - 1, centers + 1)

    # preallocate lagrange polynomial evaluations and initialize stencil
    if rational:
        li_prime_x = RationalArray(np.zeros_like(interfaces))
    else:
        li_prime_x = np.zeros_like(interfaces, dtype=np.float64)
        interfaces = interfaces.astype(np.float64)
    stencil = Stencil()

    # construct lagrange polynomials and derivatives
    for i in range(1, len(interfaces)):
        p = binomial_product(-np.delete(interfaces, i))
        li = p / p(interfaces[i])
        li_prime_x[i] = li.differentiate()(x)

    # add up stencil weights
    for i, x in enumerate(centers):
        stencil.add_node(x // 2, np.sum(li_prime_x[i + 1 :]) * 2)

    return stencil


def conservative_interpolation_stencil(
    p: int, x: Union[str, int, float], rational: bool = True
) -> Stencil:
    centers = np.arange(-(-2 * (-p // 2)), -2 * (-p // 2) + 2, 2)
    if p % 2 == 0:
        stencil = compute_conservative_interpolation_weights(
            centers, x, rational=rational
        )
    else:
        one_half = RationalArray(1, 2)
        left_biased_stencil = compute_conservative_interpolation_weights(
            centers[1:], x, rational=rational
        )
        right_biased_stencil = compute_conservative_interpolation_weights(
            centers[:-1], x, rational=rational
        )
        stencil = one_half * left_biased_stencil + one_half * right_biased_stencil
    return stencil


def compute_uniform_quadrature_weights(
    centers: np.ndarray, rational: bool = True
) -> Stencil:
    # validate input
    _validate_centers(centers)

    # initialize stencil
    stencil = Stencil()

    # construct lagrange polynomials and antiderivatives
    for i, x in enumerate(centers):
        p = binomial_product(-np.delete(centers, i))
        if rational:
            p = p.convert_coeffs_to_numpy()
        li = p / p(centers[i])
        Li = li.antidifferentiate()
        stencil.add_node(x // 2, (Li(1) - Li(-1)) // 2)

    return stencil


def uniform_quadrature_stencil(p: int, rational: bool = True) -> Stencil:
    centers = np.arange(-(-2 * (-p // 2)), -2 * (-p // 2) + 2, 2)
    return compute_uniform_quadrature_weights(centers, rational=rational)
