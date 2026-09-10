"""Smoke-test an installed Ginsu release candidate by dependency profile."""

from __future__ import annotations

import argparse
import importlib.util
from importlib import metadata

import polars as pl

from ginsu import Slicefinder, is_numba_available


def _finder() -> tuple[Slicefinder, pl.DataFrame]:
    frame = pl.DataFrame(
        {
            "region": ["east", "east", "west", "west"],
            "tier": [1, 2, 1, 2],
        }
    )
    finder = Slicefinder(
        alpha=0.95,
        k=3,
        max_l=2,
        min_sup=1,
        verbose=False,
    ).fit(frame, [4.0, 3.0, 1.0, 1.0])
    assert finder.slices_.height
    assert finder.membership_frame(frame).height == frame.height
    return finder, frame


def smoke(profile: str, expected_version: str) -> None:
    """Exercise the installed artifact and the selected optional boundary."""
    assert metadata.version("ginsu") == expected_version
    assert importlib.util.find_spec("sliceline") is None
    finder, frame = _finder()

    if profile == "core":
        assert importlib.util.find_spec("pandas") is None
        assert importlib.util.find_spec("pyarrow") is None
        assert importlib.util.find_spec("plotly") is None
        assert importlib.util.find_spec("numba") is None
    elif profile == "compat":
        import pandas as pd
        import pyarrow as pa

        arrow_frame = pa.table(frame.to_dict(as_series=False))
        pandas_frame = arrow_frame.to_pandas(types_mapper=pd.ArrowDtype)
        arrow_finder = Slicefinder(min_sup=1, verbose=False).fit(
            arrow_frame, [4.0, 3.0, 1.0, 1.0]
        )
        pandas_finder = Slicefinder(min_sup=1, verbose=False).fit(
            pandas_frame, [4.0, 3.0, 1.0, 1.0]
        )
        assert arrow_finder.slices_.equals(pandas_finder.slices_)
        assert importlib.util.find_spec("plotly") is None
        assert importlib.util.find_spec("numba") is None
    elif profile == "plot":
        from ginsu.plotting import plot_impact

        figure = plot_impact(finder)
        assert figure.data
        assert importlib.util.find_spec("pandas") is None
        assert importlib.util.find_spec("pyarrow") is None
        assert importlib.util.find_spec("numba") is None
    elif profile == "optimized":
        assert is_numba_available()
        assert importlib.util.find_spec("pandas") is None
        assert importlib.util.find_spec("pyarrow") is None
        assert importlib.util.find_spec("plotly") is None
    else:
        raise ValueError(f"Unknown smoke profile {profile!r}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("core", "compat", "plot", "optimized"),
        required=True,
    )
    parser.add_argument("--expected-version", required=True)
    arguments = parser.parse_args()
    smoke(arguments.profile, arguments.expected_version)


if __name__ == "__main__":
    main()
