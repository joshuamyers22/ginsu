"""Canonical Polars input boundary for Ginsu."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import polars as pl

FrameKind = Literal["polars", "numpy", "interchange", "arrow"]


@dataclass(frozen=True, slots=True)
class FrameInput:
    """A validated Polars frame and the kind of its producer."""

    frame: pl.DataFrame
    kind: FrameKind


def normalize_frame(
    data: Any, *, allow_copy: bool = True, allow_nan: bool = False
) -> FrameInput:
    """Normalize a supported two-dimensional input to a Polars DataFrame.

    pandas support is provided through the public dataframe interchange
    protocol. Arrow objects are accepted by Polars directly; Ginsu never
    imports either producer library.
    """
    if isinstance(data, pl.LazyFrame):
        raise TypeError(
            "LazyFrame input is not supported; collect it explicitly before fit."
        )

    if not allow_copy and not isinstance(data, pl.DataFrame):
        raise RuntimeError(
            "Copy-free conversion can only be guaranteed for an existing "
            "Polars DataFrame."
        )

    if isinstance(data, pl.DataFrame):
        result = FrameInput(data, "polars")
    elif isinstance(data, np.ndarray) or isinstance(data, (list, tuple)):
        array = np.asarray(data)
        if array.ndim != 2:
            raise ValueError(
                f"X must be two-dimensional; received {array.ndim} dimensions."
            )
        columns = [f"column_{index}" for index in range(array.shape[1])]
        result = FrameInput(
            pl.DataFrame(array, schema=columns, orient="row"), "numpy"
        )
    elif hasattr(data, "__arrow_c_stream__") or hasattr(
        data, "__arrow_c_array__"
    ):
        try:
            frame = pl.DataFrame(data)
        except TypeError as error:
            raise TypeError(
                "Arrow X input must be a table or record batch."
            ) from error
        result = FrameInput(frame, "arrow")
    elif hasattr(data, "__dataframe__"):
        result = FrameInput(pl.from_dataframe(data), "interchange")
    else:
        raise TypeError(
            "X must be a Polars DataFrame, a two-dimensional NumPy-like "
            "array, an Arrow table, or a dataframe-interchange producer."
        )

    _validate_frame(
        result.frame,
        allow_object=result.kind == "numpy",
        allow_nan=allow_nan,
    )
    return result


def validate_schema(
    frame: pl.DataFrame,
    *,
    expected: tuple[tuple[str, str], ...],
) -> None:
    """Require the same ordered names and dtypes captured during fitting."""
    actual = schema_signature(frame)
    if actual != expected:
        raise ValueError(
            "X schema does not match the fitted schema. "
            f"Expected {expected!r}, received {actual!r}."
        )


def schema_signature(frame: pl.DataFrame) -> tuple[tuple[str, str], ...]:
    """Return a stable ordered signature for a Polars schema."""
    return tuple((name, str(dtype)) for name, dtype in frame.schema.items())


def _validate_frame(
    frame: pl.DataFrame, *, allow_object: bool, allow_nan: bool
) -> None:
    if frame.height == 0:
        raise ValueError("X must contain at least one row.")
    if frame.width == 0:
        raise ValueError("X must contain at least one feature.")
    if len(frame.columns) != len(set(frame.columns)):
        raise ValueError("X column names must be unique.")
    reserved = [name for name in frame.columns if name.startswith("__ginsu_")]
    if reserved:
        raise ValueError(
            "X column names beginning with '__ginsu_' are reserved: "
            f"{reserved!r}."
        )
    if frame.null_count().row(0) != (0,) * frame.width:
        raise ValueError("X must not contain null values.")

    unsupported = []
    for name, dtype in frame.schema.items():
        if dtype.is_nested() or dtype in (pl.Null, pl.Unknown):
            unsupported.append((name, str(dtype)))
        elif dtype == pl.Object and not allow_object:
            unsupported.append((name, str(dtype)))
        elif isinstance(dtype, pl.Decimal):
            unsupported.append((name, str(dtype)))
        elif isinstance(dtype, pl.Datetime) and dtype.time_zone is not None:
            unsupported.append((name, str(dtype)))
    if unsupported:
        raise TypeError(
            f"X contains unsupported Polars dtypes: {unsupported!r}."
        )

    numeric_columns = [
        name for name, dtype in frame.schema.items() if dtype.is_float()
    ]
    if numeric_columns and not allow_nan:
        has_nan = frame.select(
            pl.any_horizontal(
                [pl.col(name).is_nan() for name in numeric_columns]
            )
            .any()
            .alias("has_nan")
        ).item()
        if has_nan:
            raise ValueError("X must not contain NaN values.")
