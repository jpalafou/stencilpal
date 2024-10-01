from numbers import Number
import numpy as np
from stencilpal.math_utils import (
    add_fractions,
    binomial_product,
    form_common_denominator,
    mul_fractions,
    polynomial_differentiate,
    polynomial_evaluate,
    simplify_fractions,
    sum_fractions,
)
from typing import Union


def compute_conservative_interpolation_stencil_weights(
    centers: np.ndarray, x: Union[str, int, float], return_rational: bool = False
) -> np.ndarray:
    """
    construct the conservative interpolation stencil for a given set of stencil indices.
    args:
        centers: np.ndarray, positions of finite-volume cell-centers relative to the
            central point. for a uniform finite-volume grid, cells-centers are given by
            x=[0,+/-2,+/-4,...]. must be sorted in ascending order and have int type.
        x: Union[int, float], interpolation point relative to the central point. must
            be in the interval [-1,1]. "l" and "r" correspond to x=-1 and x=1,
            respectively, while "c" corresponds to x=0.
        return_rational: bool, if True, return stencil weights as fractions. if False,
            return stencil weights as either integers or floating-point numbers.
    returns:
        out: np.ndarray, stencil weights. if x is an integer, out is an array of un-normalized integer weights.
    """
    # validate input
    int_mode = isinstance(x, int) or np.issubdtype(type(x), np.integer)
    if not np.issubdtype(centers.dtype, np.integer) or (
        centers.size > 1 and not np.all(np.diff(centers) == 2)
    ):
        raise ValueError(f"invalid {centers=}. must be integers evenly spaced by 2")
    if (isinstance(x, Number) and (x < -1 or x > 1)) or (
        isinstance(x, str) and x not in ["l", "c", "r"]
    ):
        raise ValueError(f"invalid interpolation point {x=}")
    if return_rational and not int_mode:
        raise ValueError(
            "rational output is only supported for integer interpolation points"
        )
    if x in ["l", "c", "r"]:
        x = {"l": -1, "c": 0, "r": 1}[x]

    # compute uniform interfaces
    interfaces = np.append(centers[0] - 1, centers + 1)

    # preallocate numpy representation of lagrange polynomial evaluations (rational numbers)
    liprime_numerator = np.zeros_like(interfaces, dtype=type(x))
    liprime_denominator = np.ones_like(interfaces)

    # construct lagrange polynomials
    for i in range(1, len(interfaces)):
        li_numerator_coeffs = binomial_product(-np.delete(interfaces, i))
        li_denominator = polynomial_evaluate(li_numerator_coeffs, interfaces[i])
        liprime_numerator[i] = polynomial_evaluate(
            polynomial_differentiate(li_numerator_coeffs), x
        )
        # denominator is unchanged by differentiation
        liprime_denominator[i] = li_denominator

    # simplify
    if int_mode:
        liprime_numerator, liprime_denominator = simplify_fractions(
            liprime_numerator, liprime_denominator
        )

    # compute stencil weights
    if return_rational:
        numerators = np.empty_like(centers, dtype=type(x))
        denominators = np.empty_like(centers, dtype=type(x))
        for i in range(len(centers)):
            numerators[i], denominators[i] = sum_fractions(
                liprime_numerator[i + 1 :], liprime_denominator[i + 1 :]
            )
        weights = numerators, denominators
    else:
        liprime_numerator, liprime_denominator = form_common_denominator(
            liprime_numerator,
            liprime_denominator,
            allow_floating_numerator=not int_mode,
        )

        # normalize if not in integer mode
        if not int_mode:
            liprime_numerator /= liprime_denominator.astype(liprime_numerator.dtype)

        # compute stencil weights (numerator only)
        weights = np.empty_like(centers, dtype=type(x))
        for i in range(len(centers)):
            weights[i] = np.sum(liprime_numerator[i + 1 :])

    # return stencil weights
    return weights


def conservative_interpolation_stencil(p: int, x: Union[str, int, float]) -> np.ndarray:
    """
    construct the conservative interpolation stencil for a given polynomial degree and
    interpolation point.
    args:
        p: int, polynomial order
        x: Union[int, float], interpolation point relative to the central point. must
            be in the interval [-1,1]. "l" and "r" correspond to x=-1 and x=1,
            respectively, while "c" corresponds to x=0.
    returns:
        weights: np.ndarray, stencil weights. if x is an integer, out is an array of un-normalized integer weights.
    """
    int_mode = isinstance(x, int) or np.issubdtype(type(x), np.integer)

    # find stencil grid points
    centers = np.arange(-(-2 * (-p // 2)), -2 * (-p // 2) + 2, 2)

    # compute stencil weights
    if p % 2 == 0:
        weights = compute_conservative_interpolation_stencil_weights(centers, x)
    else:
        # compute weights for left- and right-biased stencils, then combine
        left_biased_centers = centers[:-1]
        left_biased_weights = compute_conservative_interpolation_stencil_weights(
            left_biased_centers, x, return_rational=int_mode
        )
        right_biased_centers = centers[1:]
        right_biased_weights = compute_conservative_interpolation_stencil_weights(
            right_biased_centers, x, return_rational=int_mode
        )

        # preserve fractional representation if in integer mode
        if int_mode:
            left_biased_weights_numerator, left_biased_weights_denominator = (
                left_biased_weights
            )
            left_biased_weights_numerator = np.append(left_biased_weights_numerator, 0)
            left_biased_weights_denominator = np.append(
                left_biased_weights_denominator, 1
            )

            right_biased_weights_numerator, right_biased_weights_denominator = (
                right_biased_weights
            )
            right_biased_weights_numerator = np.append(
                0, right_biased_weights_numerator
            )
            right_biased_weights_denominator = np.append(
                1, right_biased_weights_denominator
            )

            weights_numerator, weights_denominator = add_fractions(
                *mul_fractions(
                    np.array([1]),
                    np.array([2]),
                    left_biased_weights_numerator,
                    left_biased_weights_denominator,
                ),
                *mul_fractions(
                    np.array([1]),
                    np.array([2]),
                    right_biased_weights_numerator,
                    right_biased_weights_denominator,
                ),
            )
            weights, _ = form_common_denominator(weights_numerator, weights_denominator)
        else:
            left_biased_weights = np.append(left_biased_weights, 0)
            right_biased_weights = np.append(0, right_biased_weights)
            weights = 0.5 * (left_biased_weights + right_biased_weights)

    # return stencil weights
    return weights
