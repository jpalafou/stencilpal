from dataclasses import dataclass
import numpy as np
from typing import Any, Tuple, Union

# array function registry
HANDLED_FUNCTIONS = {}


# helper functions
def _int_like(x):
    return isinstance(x, int) or np.issubdtype(type(x), np.integer)


def _float_like(x):
    return isinstance(x, float) or np.issubdtype(type(x), np.floating)


def _raise_arg_not_supported(name: str, value: Any):
    if value is not None:
        raise ValueError(f"{name} argument is not supported.")


def implements(np_function):
    "Register an __array_function__ implementation for RationalArray objects."

    def decorator(func):
        HANDLED_FUNCTIONS[np_function] = func
        return func

    return decorator


@dataclass(eq=False, repr=False)
class RationalArray(np.lib.mixins.NDArrayOperatorsMixin):
    """
    a class for representing an array of rational numbers
    args:
        numerator: Union[int, np.ndarray]
        denominator: Union[int, np.ndarray]
    """

    numerator: Union[int, np.ndarray]
    denominator: Union[int, np.ndarray] = 1

    def __post_init__(self):
        n, d = self.numerator, self.denominator

        # convert integers to numpy arrays
        if isinstance(n, np.ndarray):
            if n.size == 1:
                n = n.flat[0]
        if isinstance(d, np.ndarray):
            if d.size == 1:
                d = d.flat[0]
        if _int_like(n) and _int_like(d):
            self.numerator = np.array([n])
            self.denominator = np.array([d])
        elif _int_like(n):
            self.numerator = np.full_like(d, n)
        elif _int_like(d):
            self.denominator = np.full_like(n, d)

        # convert array_likes to numpy arrays
        if not isinstance(self.numerator, np.ndarray):
            self.numerator = np.asarray(self.numerator)
        if not isinstance(self.denominator, np.ndarray):
            self.denominator = np.asarray(self.denominator)

        # store attributes
        self.auto_simplify = True
        self.dtype = self.numerator.dtype
        self.ndim = self.numerator.ndim
        self.size = self.numerator.size

        # validate input
        self._validate()

    def _validate(self):
        n, d = self.numerator, self.denominator

        if not np.all(d):
            raise ValueError("Denominator elements cannot be 0.")
        if np.isnan(n).any() or np.isnan(d).any():
            raise ValueError("Numerator or denominator contains NaN.")
        if np.isinf(n).any() or np.isinf(d).any():
            raise ValueError("Numerator or denominator contains infinity.")
        if not np.issubdtype(n.dtype, np.integer):
            raise ValueError("Numerator must be of integer type.")
        if not np.issubdtype(d.dtype, np.integer):
            raise ValueError("Denominator must be of integer type.")
        if n.size == 0:
            raise ValueError("Numerator must be non-empty.")
        if d.size == 0:
            raise ValueError("Denominator must be non-empty.")
        if n.shape != d.shape:
            raise ValueError("Numerator and denominator must have the same shape.")
        if n.dtype != d.dtype:
            raise ValueError("Numerator and denominator must have the same dtype.")

    def __repr__(self):
        if self.ndim > 1:
            raise NotImplementedError("RationalArray repr with ndim > 1 not supported.")
        elements = []
        for num, denom in zip(self.numerator, self.denominator):
            elements.append(f"{num}/{denom}")
        elements_str = ", ".join(elements)
        return f"[{elements_str}]"

    def simplify_negatives(self, in_place: bool = False) -> "RationalArray":
        """
        ensre that the denominator is positive
        """
        if np.any(self.denominator < 0):
            n, d = self.numerator, self.denominator
            positivity_factor = np.where(d < 0, -1, 1)
            if in_place:
                self.numerator = n * positivity_factor
                self.denominator = np.abs(d)
                return self
            return self.__class__(n * positivity_factor, np.abs(d))
        return self

    def simplify(self, in_place: bool = False) -> "RationalArray":
        """
        independently simplify each rational number in the array
        """
        arr = self
        arr.simplify_negatives(in_place=True)
        n, d = arr.numerator, arr.denominator
        gcd = np.gcd(n, d)
        if in_place:
            arr.numerator = n // gcd
            arr.denominator = d // gcd
            return arr
        return self.__class__(n // gcd, d // gcd)

    def form_common_denominator(self) -> "RationalArray":
        """
        find the common denominator of the rational array
        """
        rarr = self.simplify_negatives()
        n, d = rarr.numerator, rarr.denominator
        lcm = np.lcm.reduce(d)
        return self.__class__(n * (lcm // d), np.full_like(d, lcm))

    def decompose(self) -> Tuple["RationalArray", "RationalArray"]:
        """
        decompose the rational array into its numerator and denominator
        """
        return self.__class__(self.numerator, 1), self.__class__(1, self.denominator)

    def asnumpy(self) -> np.ndarray:
        """
        convert the rational array to a numpy array
        returns:
            np.ndarray
        """
        return self.numerator / self.denominator

    def __neg__(self) -> "RationalArray":
        return self.__class__(-self.numerator, self.denominator)

    def reciprocal(self) -> "RationalArray":
        return self.__class__(self.denominator, self.numerator)

    def __add__(self, other: "RationalArray") -> "RationalArray":
        if isinstance(other, self.__class__):
            n1, d1 = self.numerator, self.denominator
            n2, d2 = other.numerator, other.denominator
            lcm = np.lcm(d1, d2)
            rarr = self.__class__(n1 * (lcm // d1) + n2 * (lcm // d2), lcm)
            if self.auto_simplify or other.auto_simplify:
                return rarr.simplify()
            return rarr
        elif _int_like(other):
            return self.__add__(self.__class__(other, 1))
        return NotImplemented

    def __sub__(self, other: "RationalArray") -> "RationalArray":
        return self.__add__(-other)

    def _mul_by_float(self, other: float) -> np.ndarray:
        return self.asnumpy() * other

    def _mul_by_int(self, other: int) -> "RationalArray":
        rarr = self.__class__(self.numerator * other, self.denominator)
        if self.auto_simplify:
            return rarr.simplify()
        return rarr

    def _mul_by_nparray(self, other: np.ndarray) -> np.ndarray:
        if np.issubdtype(other.dtype, np.integer):
            rarr = self.__class__(self.numerator * other, self.denominator)
            if self.auto_simplify or other.auto_simplify:
                return rarr.simplify()
            return rarr
        elif np.issubdtype(other.dtype, np.floating):
            return self._mul_by_float(other)
        return NotImplemented

    def _mul_by_RationalArray(
        self,
        other: "RationalArray",
    ) -> "RationalArray":
        n1, d1 = self.numerator, self.denominator
        n2, d2 = other.numerator, other.denominator
        rarr = self.__class__(n1 * n2, d1 * d2)
        if self.auto_simplify or other.auto_simplify:
            return rarr.simplify()
        return rarr

    def __mul__(
        self,
        other: Union["RationalArray", np.ndarray, int, float],
    ) -> Union["RationalArray", np.ndarray]:
        if isinstance(other, self.__class__):
            return self._mul_by_RationalArray(other)
        elif isinstance(other, np.ndarray):
            return self._mul_by_nparray(other)
        elif _int_like(other):
            return self._mul_by_int(other)
        elif _float_like(other):
            return self._mul_by_float(other)
        return NotImplemented

    def __rmul__(
        self, other: Union["RationalArray", np.ndarray, int, float]
    ) -> Union["RationalArray", np.ndarray]:
        return self.__mul__(other)

    def __floordiv__(
        self, other: Union[int, float, np.ndarray, "RationalArray"]
    ) -> Union[np.ndarray, "RationalArray"]:
        if isinstance(other, self.__class__):
            return self * other.reciprocal()
        elif _int_like(other):
            return self * self.__class__(1, other)
        elif isinstance(other, np.ndarray) or _float_like(other):
            return self.asnumpy() / other
        return NotImplemented

    def __truediv__(
        self, other: Union[int, float, np.ndarray, "RationalArray"]
    ) -> Union[np.ndarray, "RationalArray"]:
        return self.__floordiv__(other)

    def __getitem__(self, idx):
        return self.__class__(self.numerator[idx], self.denominator[idx])

    def __setitem__(self, idx, value):
        if isinstance(value, RationalArray):
            self.numerator[idx] = value.numerator
            self.denominator[idx] = value.denominator
            self.__post_init__()
        ValueError("Value must be a RationalArray.")

    def __array__(self, dtype=None, copy=None):
        if dtype is not None:
            if not np.issubdtype(dtype, np.floating):
                raise ValueError(f"Invalid dtype for RationalArray conversion: {dtype}")
        if copy is not None:
            raise ValueError("copy argument is not supported.")
        return self.asnumpy()

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == "__call__":
            if ufunc not in HANDLED_FUNCTIONS:
                # fall back to NumPy ufunc
                np_inputs = [
                    x.asnumpy() if isinstance(x, self.__class__) else x for x in inputs
                ]
                return ufunc(*np_inputs, **kwargs)
            return HANDLED_FUNCTIONS[ufunc](*inputs, **kwargs)
        return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)


# register array function implementations
@implements(np.abs)
def abs(arr):
    return arr.__class__(np.abs(arr.numerator), np.abs(arr.denominator))


@implements(np.append)
def append(arr, values, axis=None):
    if not isinstance(values, RationalArray) or not isinstance(arr, RationalArray):
        raise ValueError("inputs must be RationalArrays.")
    appended_numerator = np.append(arr.numerator, values.numerator, axis=axis)
    appended_denominator = np.append(arr.denominator, values.denominator, axis=axis)
    return RationalArray(appended_numerator, appended_denominator)


@implements(np.concatenate)
def concatenate(arrs, **kwargs):
    _raise_arg_not_supported("out", kwargs.get("out", None))
    nums = [a.numerator for a in arrs]
    dens = [a.denominator for a in arrs]
    concatenated_numerator = np.concatenate(nums, **kwargs)
    concatenated_denominator = np.concatenate(dens, **kwargs)
    return RationalArray(concatenated_numerator, concatenated_denominator)


@implements(np.full_like)
def full_like(a, fill_value, *args, **kwargs):
    if isinstance(fill_value, RationalArray) or _int_like(fill_value):
        numerator = np.ones_like(a.numerator, *args, **kwargs)
        return RationalArray(numerator) * fill_value
    raise ValueError("fill_value must be a RationalArray or integer.")


@implements(np.insert)
def insert(arr, obj, values, axis=None):
    if not isinstance(values, RationalArray) or not isinstance(arr, RationalArray):
        raise ValueError("inputs must be RationalArrays.")
    inserted_numerator = np.insert(arr.numerator, obj, values.numerator)
    inserted_denominator = np.insert(arr.denominator, obj, values.denominator)
    return RationalArray(inserted_numerator, inserted_denominator)


@implements(np.mean)
def mean(arr):
    arr = np.sum(arr) * RationalArray(1, arr.size)
    if arr.auto_simplify:
        return arr.simplify()


@implements(np.multiply)
def multiply(arr1, arr2):
    if isinstance(arr1, np.ndarray):
        return arr2 * arr1
    return arr1 * arr2


@implements(np.nonzero)
def nonzero(arr):
    return np.nonzero(arr.numerator)


@implements(np.square)
def square(arr):
    arr = RationalArray(np.square(arr.numerator), np.square(arr.denominator))
    if arr.auto_simplify:
        return arr.simplify()


@implements(np.sum)
def sum(arr):
    arr = arr.form_common_denominator()
    arr = RationalArray(np.sum(arr.numerator), arr.denominator.flat[0])
    if arr.auto_simplify:
        return arr.simplify()
