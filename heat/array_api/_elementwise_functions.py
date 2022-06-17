from __future__ import annotations

from ._dtypes import _numeric_dtypes, _result_type
from ._array_object import Array

import heat as ht


def equal(x1: Array, x2: Array, /) -> Array:
    """
    Computes the truth value of ``x1_i == x2_i`` for each element ``x1_i`` of
    the input array ``x1`` with the respective element ``x2_i`` of the input
    array ``x2``.

    Parameters
    ----------
    x1 : Array
        First input array.
    x2 : Array
        Second input array. Must be compatible with ``x1``.
    """
    return Array._new(ht.eq(x1._array, x2._array))


def isfinite(x: Array, /) -> Array:
    """
    Tests each element ``x_i`` of the input array ``x`` to determine if finite
    (i.e., not ``NaN`` and not equal to positive or negative infinity).

    Parameters
    ----------
    x : Array
        Input array. Must have a numeric data type.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in isfinite")
    return Array._new(ht.isfinite(x._array))


def isinf(x: Array, /) -> Array:
    """
    Tests each element ``x_i`` of the input array ``x`` to determine if equal
    to positive or negative infinity.

    Parameters
    ----------
    x : Array
        Input array. Must have a numeric data type.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in isinf")
    return Array._new(ht.isinf(x._array))


def isnan(x: Array, /) -> Array:
    """
    Tests each element ``x_i`` of the input array ``x`` to determine whether
    the element is ``NaN``.

    Parameters
    ----------
    x : Array
        Input array. Must have a numeric data type.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in isnan")
    return Array._new(ht.isnan(x._array))
