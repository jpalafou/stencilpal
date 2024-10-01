from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from typing import Iterable, Tuple, Union

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


def polynomial_evaluate(
    polynomial_coeffs: np.ndarray, x: Union[int, float], keep_dims: bool = False
) -> Union[int, float, np.ndarray]:
    """
    Evaluate a polynomial at a given value.

    Args:
        polynomial_coeffs (np.ndarray): An array of coefficients [an, ..., a0],
            representing the polynomial an*x^n + ... + a1*x + a0.
        x (Union[int, float]): The value at which to evaluate the polynomial.
        keep_dims (bool): Whether to keep the dimensions of the polynomial evaluation.
            If False, the output is an array of monomial evaluations ai*x^i
    Returns:
        out(keep_dims=False) Union[int, float]: scalar polynomial evaluation
        out(keep_dims=True) np.ndarray: array of monomial evaluations ai*x^i
    """
    p = len(polynomial_coeffs) - 1
    evaluation_of_each_power = polynomial_coeffs * np.power(x, np.arange(p, -1, -1))
    out = evaluation_of_each_power if keep_dims else np.sum(evaluation_of_each_power)
    return out


def polynomial_differentiate(polynomial_coeffs: np.ndarray) -> np.ndarray:
    """
    Compute the derivative of a polynomial.

    Args:
        polynomial_coeffs (np.ndarray): An array of coefficients [an, ..., a0],
            representing the polynomial an*x^n + ... + a1*x + a0.
    Returns:
        np.ndarray: An array of coefficients [an-1, ..., a0], representing the
            derivative of the polynomial.
    """
    p = len(polynomial_coeffs) - 1
    return polynomial_coeffs[:-1] * np.arange(p, 0, -1)


def _check_valid_rational_array(
    rarr: RationalArray,
    allow_floating_numerator: bool = False,
):
    """
    check validity of numpy array representation of rational numbers.
    args:
        rarr: RationalArray, array of rational numbers
        allow_floating_numerator: bool, whether to allow floating point numerators
    """
    numerator, denominator = rarr
    if not allow_floating_numerator:
        if not np.issubdtype(numerator.dtype, np.integer):
            raise ValueError(
                f"The numerator must have an integer type, not {numerator.dtype}."
            )
    if not np.issubdtype(denominator.dtype, np.integer):
        raise ValueError(
            f"The denominator must have an integer type, not {denominator.dtype}."
        )
    if numerator.shape != denominator.shape:
        raise ValueError("The numerator and denominator must have the same shape.")
    if np.any(denominator == 0):
        raise ValueError("The denominator cannot be zero.")


def simplify_fractions(rarr: RationalArray) -> RationalArray:
    """
    simplify numpy array representation of rational numbers.
    args:
        rarr: RationalArray, array of rational numbers
    returns:
        RationalArray, simplified array of rational numbers
    """
    _check_valid_rational_array(rarr)
    numerator, denominator = rarr
    gcd = np.gcd(numerator, denominator)
    return numerator // gcd, denominator // gcd


def form_common_denominator(
    rarr: RationalArray, allow_floating_numerator: bool = False
) -> RationalArray:
    """
    form a common denominator for a numpy array representation of rational numbers.
    args:
        rarr: RationalArray, array of rational numbers
        allow_floating_numerator: bool, whether to allow floating point numerators
    returns:
        RationalArray, array of rational numbers with a common denominator
    """
    _check_valid_rational_array(rarr, allow_floating_numerator=allow_floating_numerator)
    numerator, denominator = rarr
    common_denominator = np.lcm.reduce(denominator)
    return numerator * common_denominator // denominator, np.full_like(
        denominator, common_denominator
    )


def sum_fractions(rarr: RationalArray) -> RationalArray:
    """
    sum of numpy array representation of rational numbers.
    args:
        rational_array: RationalArray, array of rational numbers
    returns:
        RationalArray, sum of the fractions, simplified
    """
    _check_valid_rational_array(rarr)
    numerator, denominator = rarr
    n, d = form_common_denominator(numerator, denominator)
    return np.array([np.sum(n)]), np.array([d[0]])


def add_fractions(rarr1: RationalArray, rarr2: RationalArray) -> RationalArray:
    """
    sum of two numpy array representations of rational numbers.
    args:
        rarr1: RationalArray, array of rational numbers
        rarr2: RationalArray, array of rational numbers
    returns:
        RationalArray, sum of the fractions, simplified
    """
    _check_valid_rational_array(rarr1)
    _check_valid_rational_array(rarr2)
    numerator1, denominator1 = rarr1
    numerator2, denominator2 = rarr2
    unsimplified = (
        numerator1 * denominator2 + numerator2 * denominator1,
        denominator1 * denominator2,
    )
    return simplify_fractions(unsimplified)


def mul_fractions(rarr1: RationalArray, rarr2: RationalArray) -> RationalArray:
    """
    product of two numpy array representations of rational numbers.
    args:
        rarr1: RationalArray, array of rational numbers
        rarr2: RationalArray, array of rational numbers
    returns:
        RationalArray, product of the fractions, simplified
    """
    numerator1, denominator1 = rarr1
    numerator2, denominator2 = rarr2
    _check_valid_rational_array(numerator1, denominator1)
    _check_valid_rational_array(numerator2, denominator2)
    unsimplified = numerator1 * numerator2, denominator1 * denominator2
    return simplify_fractions(*unsimplified)


@dataclass
class FractionArray:
    """
    a class for representing an array of fractions
    args:
        numerator: np.ndarray
        denominator: Union[int, np.ndarray]
        np_type: type, used when converting the array to a numpy array
    """

    numerator: np.ndarray
    denominator: Union[int, np.ndarray] = 1
    np_type: type = np.float64

    def __post_init__(self):
        # validate input
        if isinstance(self.numerator, (int, np.integer)) and isinstance(
            self.denominator, (int, np.integer)
        ):
            self.numerator = np.array([self.numerator])
            self.denominator = np.array([self.denominator])
        elif isinstance(self.numerator, (int, np.integer)):
            self.numerator = np.full_like(self.denominator, self.numerator)
        elif isinstance(self.denominator, (int, np.integer)):
            self.denominator = np.full_like(self.numerator, self.denominator)
        if np.any(self.denominator == 0):
            raise ValueError("The denominator cannot be zero.")
        if not (
            np.issubdtype(self.numerator.dtype, np.integer)
            and np.issubdtype(self.denominator.dtype, np.integer)
        ):
            raise ValueError("The numerator and denominator must be integers.")
        if (
            self.denominator.ndim != 1
            or self.numerator.ndim != 1
            or self.numerator.size == 0
            or self.denominator.size == 0
        ):
            raise ValueError(
                "The numerator and denominator must be non-empty 1D arrays."
            )
        if len(self.numerator) != len(self.denominator):
            raise ValueError(
                "The number of numerators and denominators must be the same."
            )

        # compute the size of the array
        self.size = len(self.numerator)

    @classmethod
    def concatenate(cls, arrays: Union[Iterable["FractionArray"]]) -> "FractionArray":
        """
        concatenate multiple FractionArrays
        args:
            arrays: Union[Iterable["FractionArray"]]
        returns:
            FractionArray
        """
        return cls(
            np.concatenate([array.numerator for array in arrays]),
            np.concatenate([array.denominator for array in arrays]),
        )

    @classmethod
    def all_equal(cls, fraction_array1, fraction_array2) -> bool:
        """
        check if two FractionArrays are equal
        args:
            fraction_array1: FractionArray
            fraction_array2: FractionArray
        returns:
            bool
        """
        if isinstance(fraction_array1, np.ndarray) and isinstance(
            fraction_array2, np.ndarray
        ):
            return np.all(
                np.equal(fraction_array1.numerator, fraction_array2.numerator)
                & np.equal(fraction_array1.denominator, fraction_array2.denominator)
            )
        raise ValueError(
            f"Invalid types for comparison: {type(fraction_array1)} and {type(fraction_array2)}"
        )

    def simplify(self) -> "FractionArray":
        """
        independently simplify each fraction in the fraction array
        """
        gcd = np.gcd(self.numerator, self.denominator)
        return self.__class__(
            self.numerator // gcd,
            self.denominator // gcd,
        )

    def find_common_denominator(self) -> "FractionArray":
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
        return self.numerator.astype(self.np_type) / self.denominator.astype(
            self.np_type
        )

    def get_sum(self) -> "FractionArray":
        """
        compute the sum of the fraction array
        returns:
            FractionArray, single fraction representing the sum
        """
        farr = self.find_common_denominator()
        farr = self.__class__(np.sum(farr.numerator), farr.denominator[0]).simplify()
        return farr

    def _check_same_length(self, other: "FractionArray"):
        if len(self.numerator) != len(other.numerator):
            raise ValueError("The arrays must have the same length.")

    def reciprocal(self) -> "FractionArray":
        return self.__class__(self.denominator, self.numerator)

    def __add__(self, other: "FractionArray") -> "FractionArray":
        if isinstance(other, self.__class__):
            self._check_same_length(other)
            lcm = np.lcm(self.denominator, other.denominator)
            revised_numerator = self.numerator * (
                lcm // self.denominator
            ) + other.numerator * (lcm // other.denominator)
            return self.__class__(revised_numerator, lcm)
        raise ValueError(f"Invalid type for addition: {type(other)}")

    def mul_by_int(self, other: int) -> "FractionArray":
        return self.__class__(self.numerator * other, self.denominator)

    def mul_by_float(self, other: float) -> np.ndarray:
        return self.to_numpy() * other

    def mul_by_nparray(self, other: np.ndarray) -> np.ndarray:
        if np.issubdtype(other.dtype, np.integer):
            return self.__class__(self.numerator * other, self.denominator)
        elif np.issubdtype(other.dtype, np.floating):
            return self.to_numpy() * other
        raise ValueError(
            f"Invalid type for multiplication: np.ndarray.dtype={other.dtype}"
        )

    def __mul__(
        self, other: Union["FractionArray", np.ndarray, int, float]
    ) -> Union["FractionArray", np.ndarray]:
        if isinstance(other, self.__class__):
            self._check_same_length(other)
            return self.__class__(
                self.numerator * other.numerator, self.denominator * other.denominator
            )
        elif isinstance(other, np.ndarray):
            return self.mul_by_nparray(other)
        elif isinstance(other, int):
            return self.mul_by_int(other)
        elif isinstance(other, float):
            return self.mul_by_float(other)
        raise ValueError(f"Invalid type for multiplication: {type(other)}")

    def __rmul__(
        self, other: Union["FractionArray", np.ndarray, int, float]
    ) -> Union["FractionArray", np.ndarray]:
        print(other)
        self.__mul__(other)

    def __truediv__(self, other: "FractionArray") -> "FractionArray":
        if isinstance(other, self.__class__):
            self._check_same_length(other)
            if np.any(other.numerator == 0):
                raise ValueError("The numerator of the divisor cannot be zero.")
            return self.__class__(
                self.numerator * other.denominator, self.denominator * other.numerator
            )
        raise ValueError(f"Invalid type for true division: {type(other)}")

    def __div__(self, other: "FractionArray") -> "FractionArray":
        return self.__truediv__(other)

    def __neg__(self) -> "FractionArray":
        return self.__class__(-self.numerator, self.denominator)

    def __sub__(self, other: "FractionArray") -> "FractionArray":
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
        # validate input
        if not isinstance(self.coeffs, (np.ndarray, FractionArray)):
            raise ValueError("The coefficients must be a numpy array or FractionArray.")

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

        # case 2: FractionArray
        if isinstance(self.coeffs, FractionArray):
            # check non-empty
            if self.coeffs.size == 0:
                raise ValueError("The coefficients must be a non-empty 1D array.")
            # strip leading zeros
            non_zero_idx = np.nonzero(self.coeffs.numerator)[0]
            if non_zero_idx.size == 0:
                # in case all coefficients are zero, retain a single zero (for zero polynomial)
                self.coeffs = FractionArray(
                    self.coeffs.numerator[:1], self.coeffs.denominator[:1]
                )
            else:
                self.coeffs = FractionArray(
                    self.coeffs.numerator[non_zero_idx[0] :],
                    self.coeffs.denominator[non_zero_idx[0] :],
                )
            # assign attributes
            self.as_int = False
            self.as_fraction = True

        # assign remaining attributes
        self.p = self.coeffs.size - 1

    def __call__(self, x: Union[int, float]) -> Union[int, float, FractionArray]:
        """
        Evaluate the polynomial at x
        args:
            x: Union[int, float]
        returns:
            float if x is a float or coeffs are floats
            int if x is an int and coeffs are ints
            FractionArray if x is an int and coeffs are FractionArray
        """
        polysum = self.coeffs * np.power(x, np.arange(self.p, -1, -1))
        if isinstance(polysum, FractionArray):
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
            FractionArray(self.coeffs.numerator[:-1], self.coeffs.denominator[:-1])
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
            Polynomial, coeffs become FractionArray if the original coeffs has integer dtype or is a FractionArray
        """
        if self.as_fraction:
            if not isinstance(constant, FractionArray):
                constant = FractionArray(constant)
            appended_coeffs = FractionArray.concatenate((self.coeffs, constant))
        else:
            appended_coeffs = np.append(self.coeffs, constant)
        antiderivative_multiplier = FractionArray(1, np.arange(self.p + 1, 0, -1))
        print(antiderivative_multiplier)
        return self.__class__(appended_coeffs * antiderivative_multiplier)
