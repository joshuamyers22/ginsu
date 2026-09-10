"""Validation helpers owned by Ginsu's public input contract."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl


def to_engine_array(frame: pl.DataFrame) -> npt.NDArray:
    """Convert a validated canonical frame once for the sparse search engine."""
    array = frame.to_numpy()
    if array.ndim != 2:
        raise ValueError(
            f"X must be two-dimensional; received shape {array.shape!r}."
        )
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("X must contain at least one row and one feature.")
    if np.iscomplexobj(array):
        raise ValueError("X must not contain complex values.")
    return array


def normalize_errors(
    errors: Any, *, expected_length: int, require_positive: bool = True
) -> npt.NDArray:
    """Return a contiguous float64 vector of nonnegative finite losses."""
    if isinstance(errors, pl.DataFrame):
        if errors.width != 1:
            raise ValueError(
                "errors DataFrame must contain exactly one column."
            )
        array = errors.to_series(0).to_numpy()
    elif isinstance(errors, pl.Series):
        array = errors.to_numpy()
    else:
        array = np.asarray(errors)

    if array.ndim != 1:
        raise ValueError(
            f"errors must be one-dimensional; received shape {array.shape!r}."
        )
    if array.shape[0] != expected_length:
        raise ValueError(
            "X and errors have inconsistent row counts: "
            f"{expected_length} and {array.shape[0]}."
        )
    if array.dtype.kind not in "biuf":
        raise TypeError("errors must contain numeric or Boolean loss values.")

    try:
        result = np.ascontiguousarray(array, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError("errors must be convertible to float64.") from error

    if not np.all(np.isfinite(result)):
        raise ValueError("errors must contain only finite values.")
    if np.any(result < 0):
        raise ValueError("errors must contain only nonnegative values.")
    if require_positive and not np.any(result > 0):
        raise ValueError("errors must contain at least one positive value.")
    if require_positive and (
        not np.isfinite(result.mean()) or result.mean() <= 0
    ):
        raise ValueError("errors must have a positive finite mean.")
    return result
