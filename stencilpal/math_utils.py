from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from typing import Tuple, Union

RationalArray = Tuple[np.ndarray, np.ndarray]


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
    return _binomial_product(0, len(binomial_coeffs) - 1, tuple(binomial_coeffs))


@dataclass
class Polynomial:
    """
    A class for representing a polynomial
    args:
        coeffs: Union[np.ndarray, RationalArray]
    """

    coeffs: Union[np.ndarray, RationalArray]

    def __post_init__(self):
        # validate input
        if not isinstance(self.coeffs, (np.ndarray, RationalArray)):
            raise ValueError("The coefficients must be a numpy array or RationalArray.")

        # case 1: np.ndarray
        if isinstance(self.coeffs, np.ndarray):
            # check 1D and nonempty
            if self.coeffs.size == 0 or self.coeffs.ndim != 1:
                raise ValueError("The coefficients must be a non-empty 1D array.")
            # strip leading zeros
            non_zero_idx = np.nonzero(self.coeffs)[0]
            if non_zero_idx.size == 0:
                # in case all coefficients are zero, retain a single zero (for zero polynomial)
                self.coeffs = self.coeffs[:1]
            else:
                self.coeffs = self.coeffs[non_zero_idx[0] :]
            # assign attributes
            self.as_int = np.issubdtype(self.coeffs.dtype, np.integer)
            self.as_fraction = False

        # case 2: RationalArray
        if isinstance(self.coeffs, RationalArray):
            # check non-empty
            if self.coeffs.size == 0:
                raise ValueError("The coefficients must be a non-empty 1D array.")
            # strip leading zeros
            non_zero_idx = np.nonzero(self.coeffs.numerator)[0]
            if non_zero_idx.size == 0:
                # in case all coefficients are zero, retain a single zero (for zero polynomial)
                self.coeffs = RationalArray(
                    self.coeffs.numerator[:1], self.coeffs.denominator[:1]
                )
            else:
                self.coeffs = RationalArray(
                    self.coeffs.numerator[non_zero_idx[0] :],
                    self.coeffs.denominator[non_zero_idx[0] :],
                )
            # assign attributes
            self.as_int = False
            self.as_fraction = True

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
        polysum = self.coeffs * np.power(x, np.arange(self.p, -1, -1))
        if isinstance(polysum, RationalArray):
            return polysum.get_sum()
        return np.sum(polysum)

    def simplify_coeffs(self):
        """
        Simplify the coefficients of the polynomial
        """
        if self.as_fraction:
            self.coeffs = self.coeffs.simplify()

    def differentiate(self) -> "Polynomial":
        """
        Compute the derivative of the polynomial
        returns:
            Polynomial
        """
        if self.p == 0:
            return self.__class__(np.array([0]))  # The derivative of a constant is 0.
        coeffs = (
            RationalArray(self.coeffs.numerator[:-1], self.coeffs.denominator[:-1])
            if self.as_fraction
            else self.coeffs[:-1]
        )
        return self.__class__(coeffs * np.arange(self.p, 0, -1))

    def antidifferentiate(self, constant: int = 0) -> "Polynomial":
        """
        Compute the antiderivative of the polynomial
        args:
            constant: int, the constant of integration
        returns:
            Polynomial, coeffs become RationalArray if the original coeffs has integer dtype or is a RationalArray
        """
        if self.as_fraction:
            if not isinstance(constant, RationalArray):
                constant = RationalArray(constant)
            appended_coeffs = RationalArray.concatenate((self.coeffs, constant))
        else:
            appended_coeffs = np.append(self.coeffs, constant)
        antiderivative_multiplier = RationalArray(1, np.arange(self.p + 1, 0, -1))
        print(antiderivative_multiplier)
        return self.__class__(appended_coeffs * antiderivative_multiplier)
