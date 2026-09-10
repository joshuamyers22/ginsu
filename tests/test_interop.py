"""Cross-producer parity tests for the canonical Polars boundary."""

from datetime import date, datetime, timezone

import numpy as np
import polars as pl
import pytest

from ginsu import Slicefinder
from ginsu._frame import normalize_frame


def test_polars_arrow_and_arrow_backed_pandas_have_result_parity():
    pa = pytest.importorskip("pyarrow")
    pd = pytest.importorskip("pandas")
    frame = pl.DataFrame(
        {
            "city": ["München", "München", "東京", "東京", "Lima", "Lima"],
            "day": [date(2026, 1, 1), date(2026, 1, 2)] * 3,
            "flag": [True, False] * 3,
        }
    )
    errors = np.array([5.0, 4.0, 1.0, 1.0, 1.0, 1.0])
    arrow = pa.table(frame.to_dict(as_series=False))
    pandas_arrow = arrow.to_pandas(types_mapper=pd.ArrowDtype)

    fitted = [
        Slicefinder(alpha=0.95, k=2, max_l=2, min_sup=1, verbose=False).fit(
            producer, errors
        )
        for producer in (frame, arrow, pandas_arrow)
    ]
    expected = fitted[0]

    for actual in fitted[1:]:
        assert actual.slices_.to_dicts() == expected.slices_.to_dicts()
        assert actual.predicates_.to_dicts() == expected.predicates_.to_dicts()
        assert actual.slice_statistics_.to_dicts() == pytest.approx(
            expected.slice_statistics_.to_dicts()
        )
        assert actual.transform(frame).equals(expected.transform(frame))


def test_arrow_dictionary_and_polars_categorical_have_parity():
    pytest.importorskip("pyarrow")
    frame = pl.DataFrame(
        {"city": ["east", "east", "west", "west"], "tier": [1, 2, 1, 2]}
    ).with_columns(pl.col("city").cast(pl.Categorical))
    errors = [4.0, 3.0, 1.0, 1.0]

    polars_fit = Slicefinder(alpha=0.95, min_sup=1, verbose=False).fit(
        frame, errors
    )
    arrow_fit = Slicefinder(alpha=0.95, min_sup=1, verbose=False).fit(
        frame.to_arrow(), errors
    )

    assert arrow_fit.slices_.to_dicts() == polars_fit.slices_.to_dicts()
    assert arrow_fit.slices_.schema == polars_fit.slices_.schema


def test_chunked_arrow_table_is_supported():
    pa = pytest.importorskip("pyarrow")
    table = pa.table(
        {
            "region": pa.chunked_array([["east", "east"], ["west", "west"]]),
            "tier": pa.chunked_array([[1, 2], [1, 2]]),
        }
    )

    finder = Slicefinder(alpha=0.95, min_sup=1, verbose=False).fit(
        table, [4.0, 3.0, 1.0, 1.0]
    )

    assert finder.slices_.height > 0


def test_unknown_categories_are_not_silently_remapped():
    frame = pl.DataFrame(
        {"region": ["east", "east", "west", "west"], "tier": [1, 2, 1, 2]}
    )
    finder = Slicefinder(alpha=0.95, min_sup=1, verbose=False).fit(
        frame, [4.0, 3.0, 1.0, 1.0]
    )
    unseen = pl.DataFrame({"region": ["north"], "tier": [3]})

    membership = finder.transform(unseen)

    assert (
        membership.select(pl.all().any()).row(0) == (False,) * membership.width
    )


@pytest.mark.parametrize(
    "frame, message",
    [
        (pl.DataFrame({"value": [1.0, float("nan")]}), "NaN"),
        (pl.DataFrame({"value": [1, None]}), "null"),
        (pl.DataFrame({"value": [[1], [2]]}), "unsupported"),
        (
            pl.DataFrame(
                {
                    "value": [
                        datetime(2026, 1, 1, tzinfo=timezone.utc),
                        datetime(2026, 1, 2, tzinfo=timezone.utc),
                    ]
                }
            ),
            "unsupported",
        ),
    ],
)
def test_unsupported_or_missing_values_fail_at_the_boundary(frame, message):
    with pytest.raises((TypeError, ValueError), match=message):
        normalize_frame(frame)


def test_transform_rejects_dtype_changes():
    frame = pl.DataFrame({"value": [1, 2, 1, 2]})
    finder = Slicefinder(alpha=0.95, min_sup=1, verbose=False).fit(
        frame, [4.0, 1.0, 1.0, 1.0]
    )

    with pytest.raises(ValueError, match="schema does not match"):
        finder.transform(frame.cast({"value": pl.Float64}))
