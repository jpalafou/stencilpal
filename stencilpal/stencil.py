from dataclasses import dataclass
from numbers import Number
import numpy as np
from stencilpal.rational import _int_like, RationalArray
from typing import Union


@dataclass(eq=False, repr=False)
class Stencil:
    x: np.ndarray = None
    w: Union[np.ndarray, RationalArray] = None

    def __post_init__(self):
        # handle empty stencil
        if self.x is None or self.w is None:
            if self.x is None and self.w is None:
                self.size = 0
                return
            else:
                raise ValueError("both x and w must be provided")

        # handle rational weight type
        if isinstance(self.w, RationalArray):
            self.rational = True
        elif isinstance(self.w, np.ndarray):
            self.rational = False
        else:
            raise ValueError("w must be a RationalArray or np.ndarray")

        self._validate_and_format()
        self._expand_stencil()
        self.size = self.x.size

    def _validate_and_format(self):
        # validate input
        if self.x.ndim != 1 or self.w.ndim != 1:
            raise ValueError("x and w must be 1D arrays")
        if self.x.size != self.w.size:
            raise ValueError("x and w must have the same size")
        if len(set(self.x)) != len(self.x):
            raise ValueError("x must be unique")
        if not np.issubdtype(self.x.dtype, np.integer):
            raise ValueError("x must be integers")

        # sort x and w by x
        sorted_indices = np.argsort(self.x)
        self.x, self.w = self.x[sorted_indices], self.w[sorted_indices]

    def _expand_stencil(self, index1: int = None, index2: int = None):
        """
        Expands the stencil to include indices between index1 and index2, filling in
            zeros for missing indices.
        args:
            index1: int, start index
            index2: int, end index
        """
        xmin, xmax = self.x.min(), self.x.max()
        if index1 is None:
            index1 = xmin
        if index2 is None:
            index2 = xmax
        if index1 > xmin or index2 < xmax:
            raise ValueError("index out of bounds")
        _x = np.arange(index1, index2 + 1)
        if self.rational:
            _w = RationalArray(np.zeros_like(_x), np.ones_like(_x))
        else:
            _w = np.zeros_like(_x, dtype=self.w.dtype)
        _w[np.searchsorted(_x, self.x)] = self.w
        self.x, self.w = _x, _w

    def __repr__(self) -> str:
        if self.size == 0:
            return "Stencil(empty)"
        return f"Stencil({self.to_dict()})"

    def add_node(self, x: int, w: Union[int, float, RationalArray]):
        if self.size > 0:
            if (self.rational and not isinstance(w, RationalArray)) or (
                not self.rational and not isinstance(w, Number)
            ):
                raise ValueError("w must be of the same type as the existing weights")
            self.x = np.append(self.x, x)
            self.w = np.append(self.w, w)
        else:
            # add first position
            if _int_like(x):
                self.x = np.array([x])
            else:
                raise ValueError("x must be an integer")

            # add first weight
            if isinstance(w, RationalArray):
                self.w = w
            elif isinstance(w, Number):
                self.w = np.array([w])
            else:
                raise ValueError("w must be a RationalArray, np.ndarray, or Number")
        self.__post_init__()

    def rm_node(self, x):
        if x not in self.x:
            raise ValueError(f"{x} not in stencil")
        i = np.where(self.x == x)[0][0]
        self.x = np.delete(self.x, i)
        self.w = np.delete(self.w, i)
        self.__post_init__()

    def __neg__(self) -> "Stencil":
        return Stencil(self.x, -self.w)

    def __mul__(self, other: Union[Number, RationalArray]) -> "Stencil":
        if isinstance(other, (Number, RationalArray)):
            return Stencil(self.x, self.w * other)
        raise ValueError("unsupported operand type(s) for *")

    def __rmul__(self, other: Union[Number, RationalArray]) -> "Stencil":
        return self * other

    def __add__(self, other: "Stencil") -> "Stencil":
        if isinstance(other, Stencil):
            xmin = min(self.x.min(), other.x.min())
            xmax = max(self.x.max(), other.x.max())
            self._expand_stencil(xmin, xmax)
            other._expand_stencil(xmin, xmax)
            return Stencil(self.x, self.w + other.w)
        raise ValueError(f"unsupported operand type {type(other)} for +")

    def __radd__(self, other: "Stencil") -> "Stencil":
        return self + other

    def __sub__(self, other: "Stencil") -> "Stencil":
        return self + -other

    def to_dict(self) -> dict:
        """
        Returns a dictionary representation of the stencil.
        """
        return {x: w for x, w in zip(self.x, self.w)}

    def simplify(self):
        """
        Simplifies the rational weights of the stencil.
        """
        if self.rational:
            self.w = self.w.simplify()

    def form_common_denominator(self):
        """
        Forms a common denominator for the rational weights of the stencil.
        """
        if self.rational:
            self.w = self.w.form_common_denominator()

    def weights_as_ints(self):
        """
        Converts weights of stencil to numpy int array of numerators only
        """
        if self.rational:
            self.form_common_denominator()
            self.w = self.w.numerator
            self.rational = False

    def weights_as_rationals(self):
        """
        Converts weights of stencil to RationalArray
        """
        if not self.rational:
            denominator = np.sum(self.w)
            self.w = RationalArray(self.w, denominator).simplify()
            self.rational = True
