from dataclasses import dataclass
import numpy as np
from typing import Tuple, Union


@dataclass
class FractionArray:
    """
    A class for representing an array of fractions
    args:
        numerator: np.ndarray
        denominator: Union[int, np.ndarray]
    """

    numerator: np.ndarray
    denominator: Union[int, np.ndarray] = 1

    def __post_init__(self):
        if isinstance(self.denominator, int):
            self.denominator = np.full_like(self.numerator, self.denominator)
        if any(self.denominator == 0):
            raise ValueError("The denominator cannot be zero.")
        if self.denominator.dtype != int or self.numerator.dtype != int:
            raise ValueError("The numerator and denominator must be integers.")
        if self.denominator.ndim != 1 or self.numerator.ndim != 1:
            raise ValueError("The numerator and denominator must be 1D arrays.")
        if len(self.numerator) != len(self.denominator):
            raise ValueError(
                "The number of numerators and denominators must be the same."
            )

    def simplify(self) -> "FractionArray":
        """
        Simplify the fraction array
        returns:
            FractionArray
        """
        gcd = np.gcd(self.numerator, self.denominator)
        return self.__class__(self.numerator // gcd, self.denominator // gcd)

    def find_common_denominator(
        self, other: "FractionArray" = None
    ) -> Union["FractionArray", Tuple["FractionArray", "FractionArray"]]:
        """
        Find the common denominator of the fraction array or two fraction arrays
        args:
            other: FractionArray (optional)
        returns:
            Union[FractionArray, Tuple[FractionArray, FractionArray]]
        """
        include_other = other is not None
        # find common denominator
        combined_denominators = (
            np.concatenate((self.denominator, other.denominator))
            if include_other
            else self.denominator
        )
        common_denominator = np.lcm.reduce(combined_denominators)
        # compute revised fractions
        revised_self = self.__class__(
            self.numerator * common_denominator // self.denominator,
            common_denominator,
        )
        if include_other:
            revised_other = other.__class__(
                other.numerator * common_denominator // other.denominator,
                common_denominator,
            )
            return revised_self, revised_other
        return revised_self

    def to_float(self) -> np.ndarray:
        """
        Convert the fraction array to a float array
        returns:
            np.ndarray
        """
        return self.numerator / self.denominator

    def get_sum(self) -> Tuple[int, int]:
        """
        Compute the sum of the fraction array
        returns:
            Tuple[int, int]
        """
        out = self.find_common_denominator()
        out = self.__class__(np.sum(out.numerator), out.denominator[0])
        out = out.simplify()
        return out.numerator.item(), out.denominator.item()

    def __add__(self, other):
        if isinstance(other, self.__class__):
            revised_self, revised_other = self.find_common_denominator(other)
            return self.__class__(
                revised_self.numerator + revised_other.numerator,
                revised_self.denominator,
            )
        raise ValueError(f"Invalid type for addition: {type(other)}")

    def __mul__(self, other):
        if isinstance(other, int):
            return self.__class__(self.numerator * other, self.denominator)
        if isinstance(other, np.ndarray):
            if other.dtype == int:
                return self.__class__(self.numerator * other, self.denominator)
            elif other.dtype == float:
                return self.to_float() * other
            raise ValueError(
                f"Invalid type for multiplication: np.ndarray.dtype={other.dtype}"
            )
        if isinstance(other, self.__class__):
            return self.__class__(
                self.numerator * other.numerator, self.denominator * other.denominator
            )
        raise ValueError(f"Invalid type for multiplication: {type(other)}")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self.__class__(self.numerator, self.denominator * other)

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
        self.as_int = isinstance(self.coeffs, np.ndarray) and self.coeffs.dtype == int
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
