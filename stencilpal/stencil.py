from numbers import Number
from typing import Tuple, Union

import numpy as np
import rationalpy as rp


class StencilError(Exception):
    pass


class Stencil:
    """
    Stencil class for manipulating positions and weights.
    """

    def __init__(
        self,
        x: Union[list, tuple, np.ndarray],
        w: Union[list, tuple, np.ndarray, rp.RationalArray],
        h: int = 1,
    ):
        """
        Args:
            x: Union[list, tuple, np.ndarray] - stencil positions, must be integers
            w: Union[list, tuple, np.ndarray, rp.RationalArray] - stencil weights
            h: int - step size between stencil positions
        """
        self.x = np.array(x)
        self.w = np.array(w) if not isinstance(w, rp.RationalArray) else w
        self.h = h

        # validate and format
        self._validate_and_format()

    def _validate_and_format(self):
        if not np.issubdtype(self.x.dtype, np.integer):
            raise ValueError("x must be integers.")
        if self.x.ndim != 1 or self.w.ndim != 1:
            raise ValueError("x and w must be 1D arrays.")
        if self.x.size != self.w.size:
            raise ValueError("x and w must have the same length.")
        if len(np.unique(self.x)) != len(self.x):
            raise ValueError("x must be unique.")

        # sort x and w by x
        if not np.all(np.diff(self.x) > 0):
            sorted_indices = np.argsort(self.x)
            self.x, self.w = self.x[sorted_indices], self.w[sorted_indices]

        # store attributes
        self.rational = isinstance(self.w, rp.RationalArray)
        self.size = self.x.size

    def __repr__(self) -> str:
        return f"Stencil(x={self.x}, w={self.w}, rational={self.rational})"

    def __str__(self) -> str:
        return self.__repr__()

    def copy(self) -> "Stencil":
        """
        Returns a copy of the stencil.
        """
        return Stencil(self.x.copy(), self.w.copy(), self.h)

    def rescope(
        self, xl: int = None, xr: int = None, h: float = None, inplace: bool = True
    ) -> Union[None, "Stencil"]:
        """
            Rescopes the stencil to the interval [xl, xr] with step size h. Fills in zeros
                for weights at missing positions.

        Args:
            xl: int - left boundary of the stencil
            xr: int - right boundary of the stencil
            h: float - step size between stencil positions
            inplace: bool - if True, modify the stencil in place. If False, return a
                new stencil.

            Returns:
                None or Stencil - None if inplace is True, otherwise a new stencil
        """
        xl = self.x.min() if xl is None else xl
        xr = self.x.max() if xr is None else xr
        if xl > xr:
            raise StencilError("xl must be less than or equal to xr.")
        if xl > self.x.min() or xr < self.x.max():
            raise StencilError("Rescoping cannot clip existing stencil positions.")
        _h = self.h if h is None else h
        _x = np.arange(xl, xr + 1, _h)

        # check if there are any self.x not in _x
        if not np.all(np.isin(self.x, _x)):
            raise StencilError("Rescoping would exclude existing stencil positions.")

        _w = (
            np.zeros_like(_x, dtype=self.w.dtype)
            if not self.rational
            else rp.rational_array(np.zeros_like(_x))
        )
        _w[np.searchsorted(_x, self.x)] = self.w

        # ensure that the sum of the absolute values of the weights is the same
        if self.rational and np.sum(np.abs(_w)) != np.sum(np.abs(self.w)):
            raise StencilError("Weight abs sum mismatch.")

        if inplace:
            self.x, self.w, self.h = _x, _w, _h
            self._validate_and_format()
        else:
            return Stencil(_x, _w, _h)

    def trim_zeros(self, inplace: bool = True) -> Union[None, "Stencil"]:
        """
        Removes zeros from the stencil.

        Args:
            inplace: bool - if True, modify the stencil in place. If False, return a
                new stencil.

        Returns:
            None or Stencil - None if inplace is True, otherwise a new stencil
        """
        non_zero_idx = np.nonzero(self.w)[0]
        if non_zero_idx.size == 0:
            raise ValueError("Cannot trim all zeros.")
        _x = self.x[non_zero_idx]
        _w = self.w[non_zero_idx]
        _h = self.h
        if inplace:
            self.x, self.w, self.h = _x, _w, _h
            self._validate_and_format()
        else:
            return Stencil(_x, _w, _h)

    def trim_leading_and_trailing_zeros(
        self, inplace: bool = True
    ) -> Union[None, "Stencil"]:
        """
        Removes leading and trailing zeros from the stencil.

        Args:
            inplace: bool - if True, modify the stencil in place. If False, return a
                new stencil.

        Returns:
            None or Stencil - None if inplace is True, otherwise a new stencil
        """
        non_zero_indices = np.nonzero(self.w)[0]
        if non_zero_indices.size == 0:
            raise ValueError("Cannot trim all zeros.")
        start, end = non_zero_indices[0], non_zero_indices[-1] + 1
        _x = self.x[start:end]
        _w = self.w[start:end]
        _h = self.h
        if inplace:
            self.x, self.w, self.h = _x, _w, _h
            self._validate_and_format()
        else:
            return Stencil(_x, _w, _h)

    def simplify(self, inplace: bool = True) -> Union[None, "Stencil"]:
        """
        Simplifies the rational weights of the stencil.

        Args:
            inplace: bool - if True, modify the stencil in place. If False, return a
                new stencil.

        Returns:
            None or Stencil - None if inplace is True, otherwise a new stencil
        """
        stencil = self if inplace else self.copy()
        if self.rational:
            stencil.w.simplify()
        return None if inplace else stencil

    def form_common_denominator(self, inplace: bool = True) -> Union[None, "Stencil"]:
        """
        Forms a common denominator for the rational weights of the stencil.

        Args:
            inplace: bool - if True, modify the stencil in place. If False, return a
                new stencil.

        Returns:
            None or Stencil - None if inplace is True, otherwise a new stencil
        """
        stencil = self if inplace else self.copy()
        if self.rational:
            stencil.w.form_common_denominator()
        return None if inplace else stencil

    def asnumpy(self, mode: str = "numerator") -> np.ndarray:
        """
        Returns the weights of the stencil as a numpy array.

        Args:
            mode: str - 'numerator' or 'float'
                'numerator' - returns the numerator of the rational weights after
                forming a common denominator
                'float' - returns the weights as a float array

        Returns:
            np.ndarray - the weights of the stencil
        """
        if mode == "numerator":
            if not self.rational:
                raise ValueError(
                    "'numerator' mode can only be used with rational stencils."
                )
            stencil = self.simplify(inplace=False)
            stencil.rescope(inplace=True)
            stencil.form_common_denominator(inplace=True)
            return stencil.w.numerator
        elif mode == "float":
            stencil = self.rescope(inplace=False)
            return np.array(stencil.w)
        raise ValueError(f"Unsupported mode {mode}.")

    def __neg__(self) -> "Stencil":
        return Stencil(self.x, -self.w, self.h)

    def _align_with(self, other: "Stencil") -> Tuple["Stencil", "Stencil"]:
        """
        Return two stencils aligned to the same positions.

        Returns:
            Tuple[Stencil, Stencil] - the aligned stencils
        """
        xl, xr = min(self.x.min(), other.x.min()), max(self.x.max(), other.x.max())
        _h = min(self.h, other.h)
        rescope1 = self.rescope(xl, xr, _h, inplace=False)
        rescope2 = other.rescope(xl, xr, _h, inplace=False)
        if not np.all(rescope1.x == rescope2.x):
            raise StencilError("Failed to align stencils.")
        return rescope1, rescope2

    def __mul__(
        self, other: Union["Stencil", rp.RationalArray, np.ndarray, Number]
    ) -> "Stencil":
        if isinstance(other, Stencil):
            rescope1, rescope2 = self._align_with(other)
            return Stencil(rescope1.x, rescope1.w * rescope2.w, rescope1.h)
        elif (
            isinstance(other, rp.RationalArray)
            or isinstance(other, np.ndarray)
            or isinstance(other, Number)
        ):
            return Stencil(self.x, self.w * other, self.h)
        raise ValueError(f"Unsupported operand type {type(other)} for *.")

    def __rmul__(self, other: Union[Number, rp.RationalArray]) -> "Stencil":
        return self * other

    def __add__(self, other: "Stencil") -> "Stencil":
        if isinstance(other, Stencil):
            rescope1, rescope2 = self._align_with(other)
            return Stencil(rescope1.x, rescope1.w + rescope2.w, rescope1.h)
        raise ValueError(f"Unsupported operand type {type(other)} for +.")

    def __radd__(self, other: "Stencil") -> "Stencil":
        return self + other

    def __sub__(self, other: "Stencil") -> "Stencil":
        return self + -other
