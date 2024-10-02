from dataclasses import dataclass
from functools import lru_cache
from numbers import Number
import numpy as np
from stencilpal.rational import RationalArray
from typing import Union


@lru_cache(None)
def _binomial_product(i: int, j: int, binomial_coeffs: np.ndarray):
    """
    Compute the product of binomial coefficients from index i to j.

    Args:
        i (int): The starting index of the binomial coefficients to include.
        j (int): The ending index of the binomial coefficients to include.
        binomial_coeffs (np.ndarray): An array of coefficients [a1, a2, ..., an],
            representing the binomial product (x + a1)(x + a2)...(x + an).

    Returns:
        np.ndarray: The resulting polynomial coefficients [bn, ..., b1], where bn
            corresponds to the highest power of x.
    """

    if i == j:
        return np.array([1, binomial_coeffs[i]])

    mid = (i + j) // 2
    left_product = _binomial_product(i, mid, binomial_coeffs)
    right_product = _binomial_product(mid + 1, j, binomial_coeffs)

    return np.convolve(left_product, right_product)


def binomial_product(binomial_coeffs):
    """
    Compute the product of binomial coefficients using dynamic programming.

    Args:
        i (int): The starting index of the binomial coefficients to include.
        j (int): The ending index of the binomial coefficients to include.
        binomial_coeffs (np.ndarray): An array of coefficients [a1, a2, ..., an],
            representing the binomial product (x + a1)(x + a2)...(x + an).

    Returns:
        np.ndarray: The resulting polynomial coefficients [bn, ..., b1], where bn
            corresponds to the highest power of x.
    """
    return Polynomial(
        _binomial_product(0, len(binomial_coeffs) - 1, tuple(binomial_coeffs))
    )


@dataclass
class Polynomial:
    """
    A class for representing a polynomial
    args:
        coeffs: Union[np.ndarray, RationalArray], [an, ..., a0] corresponding to the
            polynomial an*x^n + ... + a0*x^0
    """

    coeffs: Union[np.ndarray, RationalArray]

    def __post_init__(self):
        # determine array type
        self.as_fraction = isinstance(self.coeffs, RationalArray)

        # validate input
        if not self.as_fraction:
            if not isinstance(self.coeffs, np.ndarray):
                raise ValueError(
                    "The coefficients must be a numpy array or RationalArray."
                )
            if not np.issubdtype(self.coeffs.dtype, np.number):
                raise ValueError("The coefficients must be of numeric type.")
        if self.coeffs.ndim != 1:
            raise ValueError("The coefficients must be a 1D array.")
        if self.coeffs.size == 0:
            raise ValueError("The coefficients must not be empty.")

        # handle leading zeros
        non_zero_idx = np.nonzero(self.coeffs)[0]
        if non_zero_idx.size == 0:
            # in case all coefficients are zero, retain a single zero (for zero polynomial)
            self.coeffs = self.coeffs[:1]
        else:
            self.coeffs = self.coeffs[non_zero_idx[0] :]

        # assign remaining attributes
        self.p = self.coeffs.size - 1

    def __call__(self, x: Union[int, float]) -> Union[int, float, RationalArray]:
        """
        Evaluate the polynomial at x
        args:
            x: Union[int, float]
        returns:
            float if x is a float or coeffs are floats
            int if x is an int and coeffs are ints
            RationalArray if x is an int and coeffs are RationalArray
        """
        return np.sum(self.coeffs * np.power(x, np.arange(self.p, -1, -1)))

    def coeffs_simplify(self):
        """
        Simplify the coefficients of the polynomial
        """
        if self.as_fraction:
            self.coeffs = self.coeffs.simplify()

    def coeffs_form_common_denominator(self):
        if self.as_fraction:
            self.coeffs = self.coeffs.form_common_denominator()

    def differentiate(self, simplify: bool = True) -> "Polynomial":
        """
        Compute the derivative of the polynomial
        args:
            simplify: bool, if True, simplify the coefficients of the derivative
        returns:
            dpdx, Polynomial
        """
        if self.p == 0:
            if self.as_fraction:
                return self.__class__(RationalArray(0))
            return self.__class__(np.array([0]))
        dpdx = self.__class__(self.coeffs[:-1] * np.arange(self.p, 0, -1))
        if simplify:
            dpdx.coeffs_simplify()
        return dpdx

    def _check_valid_integration_constant(
        self, constant: Union[int, float, np.ndarray, RationalArray]
    ) -> Union[int, float, np.ndarray, RationalArray]:
        if isinstance(constant, (np.ndarray, RationalArray)):
            if constant.size != 1:
                raise ValueError("The constant must be a scalar.")
        elif not isinstance(constant, Number):
            raise ValueError("The constant must be a number.")
        if not isinstance(constant, RationalArray):
            return RationalArray(constant)
        return constant

    def antidifferentiate(
        self,
        constant: Union[int, float, np.ndarray, RationalArray] = 0,
        simplify: bool = True,
    ) -> "Polynomial":
        """
        Compute the antiderivative of the polynomial
        args:
            constant: Union[int, float, RationalArray], the constant of integration
            simplify: bool, if True, simplify the coefficients of the antiderivative
        returns:
            P, Polynomial
        """
        constant = self._check_valid_integration_constant(constant)
        # perform integration, include constant, and create new Polynomial
        P = self.__class__(
            np.append(
                self.coeffs * RationalArray(1, np.arange(self.p + 1, 0, -1)), constant
            )
        )
        if simplify:
            P.coeffs_simplify()
        return P
