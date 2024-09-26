from dataclasses import dataclass
import numpy as np
from typing import List, Tuple, Union


@dataclass
class FractionArray:
    """
    a class for representing an array of fractions
    args:
        numerator: np.ndarray
        denominator: Union[int, np.ndarray]
        skip_post_init: bool = False
    """

    numerator: np.ndarray
    denominator: Union[int, np.ndarray] = 1

    def __post_init__(self):
        # validate input
        if isinstance(self.denominator, (int, np.integer)):
            self.denominator = np.full_like(self.numerator, self.denominator)
        if np.any(self.denominator == 0):
            raise ValueError("The denominator cannot be zero.")
        if not (
            np.issubdtype(self.numerator.dtype, np.integer)
            and np.issubdtype(self.denominator.dtype, np.integer)
        ):
            raise ValueError("The numerator and denominator must be integers.")
        if self.denominator.ndim != 1 or self.numerator.ndim != 1:
            raise ValueError("The numerator and denominator must be 1D arrays.")
        if len(self.numerator) != len(self.denominator):
            raise ValueError(
                "The number of numerators and denominators must be the same."
            )

        # compute the size of the array
        self.size = len(self.numerator)

    @classmethod
    def concatenate(
        cls, arrays: Union[List["FractionArray"], Tuple["FractionArray"]]
    ) -> "FractionArray":
        """
        concatenate multiple FractionArrays
        args:
            arrays: Union[List["FractionArray"], Tuple["FractionArray"]]
        returns:
            FractionArray
        """
        return cls(
            np.concatenate([array.numerator for array in arrays]),
            np.concatenate([array.denominator for array in arrays]),
        )

    def simplify(self, inplace: bool = True):
        """
        independently simplify each fraction in the fraction array
        """
        gcd = np.gcd(self.numerator, self.denominator)
        return self.__class__(
            self.numerator // gcd,
            self.denominator // gcd,
        )

    def find_common_denominator(self):
        """
        find the common denominator of the fraction array
        """
        common_denominator = np.lcm.reduce(self.denominator)
        return self.__class__(
            self.numerator * common_denominator // self.denominator,
            self.denominator * common_denominator // self.denominator,
        )

    def to_numpy(self) -> np.ndarray:
        """
        convert the fraction array to a numpy array
        returns:
            np.ndarray
        """
        return self.numerator / self.denominator

    def get_sum(self) -> Tuple[int, int]:
        """
        compute the sum of the fraction array
        returns:
            Tuple[int, int]
        """
        farr = self.find_common_denominator()
        farr = self.__class__(
            np.sum(farr.numerator, keepdims=True), farr.denominator[0]
        )
        farr = farr.simplify()
        return farr.numerator.item(), farr.denominator.item()

    def check_same_length(self, other):
        if len(self.numerator) != len(other.numerator):
            raise ValueError("The arrays must have the same length.")

    def reciprocal(self):
        return self.__class__(self.denominator, self.numerator)

    def __add__(self, other):
        if isinstance(other, self.__class__):
            self.check_same_length(other)
            gcd = np.gcd(self.denominator, other.denominator)
            reivsed_self = self.__class__(
                self.numerator * (other.denominator // gcd),
                self.denominator * (other.denominator // gcd),
            )
            revised_other = self.__class__(
                other.numerator * (self.denominator // gcd),
                other.denominator * (self.denominator // gcd),
            )
            return self.__class__(
                reivsed_self.numerator + revised_other.numerator,
                reivsed_self.denominator,
            )
        raise ValueError(f"Invalid type for addition: {type(other)}")

    def __mul__(self, other):
        if isinstance(other, (int, np.integer)):
            return self.__class__(self.numerator * other, self.denominator)
        if isinstance(other, (float, np.floating)):
            return self.to_numpy() * other
        if isinstance(other, np.ndarray):
            if np.issubdtype(other.dtype, np.integer):
                return self.__class__(self.numerator * other, self.denominator)
            elif np.issubdtype(other.dtype, np.floating):
                return self.to_float() * other
            raise ValueError(
                f"Invalid type for multiplication: np.ndarray.dtype={other.dtype}"
            )
        if isinstance(other, self.__class__):
            self.check_same_length(other)
            return self.__class__(
                self.numerator * other.numerator, self.denominator * other.denominator
            )
        raise ValueError(f"Invalid type for multiplication: {type(other)}")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, self.__class__):
            self.check_same_length(other)
            if np.any(other.numerator == 0):
                raise ValueError("The numerator of the divisor cannot be zero.")
            return self.__class__(
                self.numerator * other.denominator, self.denominator * other.numerator
            )
        raise ValueError(f"Invalid type for true division: {type(other)}")

    def __neg__(self):
        return self.__class__(-self.numerator, self.denominator)

    def __sub__(self, other):
        return self.__add__(-other)


@dataclass
class Polynomial:
    """
    A class for representing a polynomial
    args:
        coeffs: Union[np.ndarray, FractionArray]
    """

    coeffs: Union[np.ndarray, FractionArray]

    def __post_init__(self):
        self.as_int = (
            isinstance(self.coeffs, np.ndarray) and self.coeffs.dtype == np.integer
        )
        self.as_fraction = isinstance(self.coeffs, FractionArray)
        self.p = len(self.coeffs) - 1

        if self.coeffs.ndim != 1:
            raise ValueError("The coefficients must be a 1D array.")
        if not (isinstance(self.coeffs, np.ndarray) or self.as_fraction):
            raise ValueError("The coefficients must be a numpy array or FractionArray.")
        if self.p < 0:
            raise ValueError("The polynomial must have at least one coefficient.")

    def __call__(self, x: Union[int, float]) -> Union[int, float, Tuple[int, int]]:
        """
        Evaluate the polynomial at x
        args:
            x: Union[int, float]
        returns:
            Union[int, float, Tuple[int, int]]
        """
        polysum = self.coeffs * np.power(x, np.arange(self.p, -1, -1))
        if self.as_fraction:
            return polysum.get_sum()
        return np.sum(polysum)

    def differentiate(self) -> "Polynomial":
        """
        Compute the derivative of the polynomial
        returns:
            Polynomial
        """
        if self.p == 0:
            return self.__class__(np.array([0]))  # The derivative of a constant is 0.
        return self.__class__(self.coeffs[:-1] * np.arange(self.p, 0, -1))

    def antidifferentiate(self) -> "Polynomial":
        """
        Compute the antiderivative of the polynomial
        returns:
            Polynomial
        """
        if self.as_int:
            return self.__class__(
                FractionArray(np.append(self.coeffs, 0)) // np.arange(self.p + 1, 0, -1)
            )
        if self.as_fraction:
            append_coeffs = FractionArray(
                np.append(self.coeffs.numerator, 0),
                np.append(self.coeffs.denominator, 1),
            )
            return self.__class__(append_coeffs // np.arange(self.p + 1, 0, -1))
        return self.__class__(np.append(self.coeffs, 0) / np.arange(self.p + 1, 0, -1))
