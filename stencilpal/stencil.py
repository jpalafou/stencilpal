import warnings
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
        x: Union[list, tuple, np.ndarray] = (),
        w: Union[list, tuple, np.ndarray, rp.RationalArray] = (),
    ):
        """
        Args:
            x: Union[list, tuple, np.ndarray] - stencil positions, must be non-empty,
                1D, non-repeating, and integers
            w: Union[list, tuple, np.ndarray, rp.RationalArray] - stencil weights, must
                be non-empty and 1D
        """
        # assign x attribute
        if isinstance(x, list) or isinstance(x, tuple):
            self.x = np.array([], int) if len(x) == 0 else np.array(x)
        elif isinstance(x, np.ndarray):
            self.x = x
        else:
            raise ValueError(f"Unsupported type {type(x)} for x.")

        # assign w attribute
        if isinstance(w, list) or isinstance(w, tuple):
            self.w = np.array([], float) if len(w) == 0 else np.array(w)
        elif isinstance(w, np.ndarray) or isinstance(w, rp.RationalArray):
            self.w = w
        else:
            raise ValueError(f"Unsupported type {type(w)} for w.")

        # validate and format
        self._validate_and_format()

    def _validate_and_format(self):
        if not np.issubdtype(self.x.dtype, np.integer):
            raise ValueError(f"x must be integers, but got {self.x.dtype}.")
        if self.x.ndim != 1 or self.w.ndim != 1:
            raise ValueError("x and w must be 1D arrays.")
        if self.x.size != self.w.size:
            raise ValueError("x and w must have the same length.")
        if len(np.unique(self.x)) != len(self.x):
            raise ValueError("x must be unique.")

        # store attributes
        self.rational = isinstance(self.w, rp.RationalArray)
        self.size = self.x.size

        # sort x and w by x
        if not np.all(np.diff(self.x) > 0):
            sorted_indices = np.argsort(self.x)
            self.x, self.w = self.x[sorted_indices], self.w[sorted_indices]

    def _check_uniform_spacing(self) -> bool:
        """
        Checks if the stencil positions are uniformly spaced.
        """
        if self.size < 2:
            return True
        return np.all(np.diff(self.x) == np.diff(self.x)[0])

    def __repr__(self) -> str:
        return f"Stencil(x={self.x}, w={self.w}, rational={self.rational})"

    def __str__(self) -> str:
        return self.__repr__()

    def add_node(
        self, x: int, w: Union[float, rp.RationalArray] = 1, inplace: bool = True
    ) -> Union[None, "Stencil"]:
        """
        Adds a node to the stencil.

        Args:
            x: int - position of the new node
            w: Union[float, rp.RationalArray] - weight of the new node
            inplace: bool - if True, modify the stencil in place. If False, return a
                new stencil.

        Returns:
            None or Stencil - None if inplace is True, otherwise a new stencil
        """
        if not np.issubdtype(type(x), np.integer):
            raise ValueError(f"x must be an integer, but got {type(x)}.")
        if not (isinstance(w, float) or isinstance(w, rp.RationalArray)):
            raise ValueError(
                f"w must be a float or rp.RationalArray, but got {type(w)}."
            )
        _x, _w = np.append(self.x, x), np.append(self.w, w)
        if inplace:
            self.x = _x
            self.w = _w
            self._validate_and_format()
        else:
            return Stencil(_x, _w)

    def remove_node(self, x: int, inplace: bool = True) -> Union[None, "Stencil"]:
        """
        Removes a node from the stencil.

        Args:
            x: int - position of the node to remove
            inplace: bool - if True, modify the stencil in place. If False, return a
                new stencil.

        Returns:
            None or Stencil - None if inplace is True, otherwise a new stencil
        """
        if self.rational:
            raise NotImplementedError("Cannot remove node from rational stencil.")
        if not np.issubdtype(type(x), np.integer):
            raise ValueError(f"x must be an integer, but got {type(x)}.")
        if x not in self.x:
            raise ValueError(f"Node at position {x} not found in stencil.")
        _x, _w = (
            np.delete(self.x, np.where(self.x == x)),
            np.delete(self.w, np.where(self.x == x)),
        )
        if inplace:
            self.x = _x
            self.w = _w
            self._validate_and_format()
        else:
            return Stencil(_x, _w)

    def copy(self) -> "Stencil":
        """
        Returns a copy of the stencil.
        """
        return Stencil(self.x.copy(), self.w.copy())

    def rescope(
        self, x: np.ndarray = None, h: int = 1, inplace: bool = True
    ) -> Union[None, "Stencil"]:
        """
        Rescopes the stencil to new positions defined by x. Fills in zeros for weights
            at missing positions.

        Args:
            x: np.ndarray - new stencil positions. If None, the stencil is rescoped to
                the range of existing positions uniformly spaced by h.
            h: int - step size between stencil positions used when x is None.
            inplace: bool - if True, modify the stencil in place. If False, return a
                new stencil.

        Returns:
            None or Stencil - None if inplace is True, otherwise a new stencil
        """
        if self.size < 2 and x is None:
            raise ValueError(
                "Cannot rescope stencil with less than 2 positions without specifying x."
            )

        _x = np.arange(self.x[0], self.x[-1] + 1, h) if x is None else x

        if not np.all(np.isin(self.x[np.nonzero(self.w)[0]], _x)):
            raise StencilError(
                "Rescoping would exclude existing stencil positions with non-zero weights."
            )

        _w = (
            rp.rational_array(np.zeros_like(_x))
            if self.rational
            else np.zeros_like(_x, dtype=self.w.dtype)
        )
        if not np.all(self.w == 0):
            _w[np.searchsorted(_x, self.x)] = self.w

        if inplace:
            self.x, self.w = _x, _w
            self._validate_and_format()
        else:
            return Stencil(_x, _w)

    def _trim_helper(self, f: callable, inplace: bool = True) -> Union[None, "Stencil"]:
        """
        Helper function for trimming zeros from the stencil.

        Args:
            f: callable - function to apply to non-zero indices. Should take an array
                of non-zero indices and return a tuple of new x and w arrays.
                f(non_zero_indices) -> (new_x, new_w)
            inplace: bool - if True, modify the stencil in place. If False, return a
                new stencil.
        """
        non_zero_indices = np.nonzero(self.w)[0]
        _x, _w = (
            f(non_zero_indices)
            if non_zero_indices.size != 0
            else (np.array([], dtype=self.x.dtype), np.array([], dtype=self.w.dtype))
        )
        if inplace:
            self.x, self.w = _x, _w
            self._validate_and_format()
        else:
            return Stencil(_x, _w)

    def trim_all_zeros(self, inplace: bool = True) -> Union[None, "Stencil"]:
        """
        Removes all stencil nodes at which the weight is zero.

        Args:
            inplace: bool - if True, modify the stencil in place. If False, return a
                new stencil.

        Returns:
            None or Stencil - None if inplace is True, otherwise a new stencil
        """
        return self._trim_helper(
            lambda non_zero_indices: (
                self.x[non_zero_indices],
                self.w[non_zero_indices],
            ),
            inplace,
        )

    def trim_leading_and_trailing_zeros(
        self, inplace: bool = True
    ) -> Union[None, "Stencil"]:
        """
        Removes leading and trailing zero-weight nodes from the stencil.

        Args:
            inplace: bool - if True, modify the stencil in place. If False, return a
                new stencil.

        Returns:
            None or Stencil - None if inplace is True, otherwise a new stencil
        """
        return self._trim_helper(
            lambda non_zero_indices: (
                self.x[non_zero_indices[0] : non_zero_indices[-1] + 1],
                self.w[non_zero_indices[0] : non_zero_indices[-1] + 1],
            ),
            inplace,
        )

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
        else:
            warnings.warn("Cannot simplify non-rational stencil.")
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
        else:
            warnings.warn("Cannot form common denominator for non-rational stencil.")
        return None if inplace else stencil

    def asnumpy(
        self, mode: str = "numerator", trim_leading_and_trailing_zeros: bool = True
    ) -> np.ndarray:
        """
        Returns the weights of the stencil as a numpy array.

        Args:
            mode: str - 'numerator' or 'float'
                'numerator' - returns the numerator of the rational weights after
                forming a common denominator. Only valid for rational stencils.
                'float' - returns the weights as a float array
            trim_leading_and_trailing_zeros: bool - if True, trim leading and trailing
                zeros from the stencil before returning the weights

        Returns:
            np.ndarray - the weights of the stencil
        """
        if not self._check_uniform_spacing():
            raise ValueError("Cannot convert non-uniform stencil to numpy array.")
        if mode == "numerator":
            if not self.rational:
                raise ValueError(
                    "'numerator' mode can only be used with rational stencils."
                )
            stencil = self.simplify(inplace=False)
            stencil.form_common_denominator(inplace=True)
            return stencil.w.numerator
        elif mode == "float":
            stencil = self.rescope(inplace=False)
            return np.array(stencil.w)
        raise ValueError(f"Unsupported mode {mode}.")

    def _align_with(self, other: "Stencil") -> Tuple["Stencil", "Stencil"]:
        """
        Return two stencils aligned to the same positions.

        Returns:
            Tuple[Stencil, Stencil] - the aligned stencils
        """
        _x = np.union1d(self.x, other.x)
        rescope1 = self.rescope(_x, inplace=False)
        rescope2 = other.rescope(_x, inplace=False)
        return rescope1, rescope2

    def __neg__(self) -> "Stencil":
        return Stencil(self.x, -self.w)

    def __add__(self, other: "Stencil") -> "Stencil":
        if isinstance(other, Stencil):
            rescope1, rescope2 = self._align_with(other)
            return Stencil(rescope1.x, rescope1.w + rescope2.w)
        raise ValueError(f"Unsupported operand type {type(other)} for +.")

    def __radd__(self, other: "Stencil") -> "Stencil":
        return self + other

    def __sub__(self, other: "Stencil") -> "Stencil":
        return self + -other

    def __mul__(
        self, other: Union["Stencil", rp.RationalArray, np.ndarray, Number]
    ) -> "Stencil":
        if isinstance(other, Stencil):
            rescope1, rescope2 = self._align_with(other)
            return Stencil(rescope1.x, rescope1.w * rescope2.w)
        elif (
            isinstance(other, rp.RationalArray)
            or isinstance(other, np.ndarray)
            or isinstance(other, Number)
        ):
            return Stencil(self.x, self.w * other)
        raise ValueError(f"Unsupported operand type {type(other)} for *.")

    def __rmul__(self, other: Union[Number, rp.RationalArray]) -> "Stencil":
        return self * other
