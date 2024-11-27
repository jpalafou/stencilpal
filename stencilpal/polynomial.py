from dataclasses import dataclass
from functools import lru_cache
from numbers import Number
from typing import Union

import numpy as np
import rationalpy as rp


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


@dataclass(eq=False)
class Polynomial:
    """
    A class for representing a polynomial
    args:
        coeffs: Union[np.ndarray, rp.RationalArray], [an, ..., a0] corresponding to the
            polynomial an*x^n + ... + a0*x^0
    """

    coeffs: Union[np.ndarray, rp.RationalArray]

    def __post_init__(self):
        # determine array type
        self.as_fraction = isinstance(self.coeffs, rp.RationalArray)

        # validate input
        if not self.as_fraction:
            if not isinstance(self.coeffs, np.ndarray):
                raise ValueError(
                    "The coefficients must be a numpy array or rp.RationalArray."
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

    def __call__(self, x: Union[int, float]) -> Union[int, float, rp.RationalArray]:
        """
        Evaluate the polynomial at x
        args:
            x: Union[int, float]
        returns:
            float if x is a float or coeffs are floats
            int if x is an int and coeffs are ints
            rp.RationalArray if x is an int and coeffs are rp.RationalArray
        """
        return np.sum(self.coeffs * np.power(x, np.arange(self.p, -1, -1)))

    def coeffs_simplify(self):
        """
        Simplify the coefficients of the polynomial
        """
        if self.as_fraction:
            self.coeffs.simplify()

    def coeffs_form_common_denominator(self):
        if self.as_fraction:
            self.coeffs.form_common_denominator()

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
                return self.__class__(rp.rational_array(0))
            return self.__class__(np.array([0]))
        dpdx = self.__class__(self.coeffs[:-1] * np.arange(self.p, 0, -1))
        if simplify:
            dpdx.coeffs_simplify()
        return dpdx

    def _check_valid_integration_constant(
        self, constant: Union[int, float, np.ndarray, rp.RationalArray]
    ) -> Union[int, float, np.ndarray, rp.RationalArray]:
        if isinstance(constant, np.ndarray) or isinstance(constant, rp.RationalArray):
            if constant.size != 1:
                raise ValueError("The constant must be a scalar.")
        elif not isinstance(constant, Number):
            raise ValueError("The constant must be a number.")
        if not isinstance(constant, rp.RationalArray):
            return rp.rational_array(constant)
        return constant

    def antidifferentiate(
        self,
        constant: Union[int, float, np.ndarray, rp.RationalArray] = 0,
        simplify: bool = True,
    ) -> "Polynomial":
        """
        Compute the antiderivative of the polynomial
        args:
            constant: Union[int, float, rp.RationalArray], the constant of integration
            simplify: bool, if True, simplify the coefficients of the antiderivative
        returns:
            P, Polynomial
        """
        constant = self._check_valid_integration_constant(constant)
        # perform integration, include constant, and create new Polynomial
        P = self.__class__(
            np.append(
                self.coeffs * rp.rational_array(1, np.arange(self.p + 1, 0, -1)),
                constant,
            )
        )
        if simplify:
            P.coeffs_simplify()
        return P

    def convert_coeffs_to_rational(self) -> "Polynomial":
        """
        Convert the coefficients of the polynomial to rp.RationalArray
        """
        if self.as_fraction:
            return self
        return self.__class__(rp.rational_array(self.coeffs))

    def convert_coeffs_to_numpy(self) -> "Polynomial":
        """
        Convert the coefficients of the polynomial to numpy array
        """
        if not self.as_fraction:
            return self
        return self.__class__(self.coeffs.asnumpy())

    def __mul__(
        self, other: Union[int, float, np.ndarray, rp.RationalArray]
    ) -> "Polynomial":
        """
        Multiply the polynomial by a scalar or another polynomial
        args:
            other: Union[int, float, np.ndarray, rp.RationalArray]
        returns:
            Polynomial
        """
        if isinstance(other, (int, float, np.ndarray, rp.RationalArray)):
            return self.__class__(self.coeffs * other)
        raise ValueError("Invalid type for multiplication.")

    def __truediv__(
        self, other: Union[int, float, np.ndarray, rp.RationalArray]
    ) -> "Polynomial":
        """
        Divide the polynomial by a scalar or another polynomial
        args:
            other: Union[int, float, np.ndarray, rp.RationalArray]
        returns:
            Polynomial
        """
        if np.issubdtype(type(other), np.integer):
            return self.__class__(self.coeffs * rp.rational_array(1, other))
        elif isinstance(other, (np.ndarray, rp.RationalArray)) or np.issubdtype(
            type(other), np.floating
        ):
            return self.__class__(self.coeffs / other)
        raise ValueError("Invalid type for division.")
