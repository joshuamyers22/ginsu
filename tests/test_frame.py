"""Tests for the producer-neutral Polars input boundary."""

import numpy as np
import polars as pl
import pytest
from sklearn.exceptions import NotFittedError

from ginsu import Slicefinder
from ginsu._frame import normalize_frame


def _example() -> tuple[pl.DataFrame, np.ndarray]:
    frame = pl.DataFrame(
        {
            "region": ["east", "east", "west", "west"],
            "tier": [1, 2, 1, 2],
        }
    )
    return frame, np.array([4.0, 1.0, 1.0, 1.0])


def test_polars_is_the_native_table_boundary():
    frame, errors = _example()

    finder = Slicefinder(
        alpha=0.95, k=1, max_l=2, min_sup=1, verbose=False
    ).fit(frame, errors)

    membership = finder.transform(frame)
    selected = finder.get_slice(frame, 0)

    assert isinstance(membership, pl.DataFrame)
    assert membership.schema == {"slice_0": pl.Boolean}
    assert isinstance(selected, pl.DataFrame)
    assert selected.schema == frame.schema


def test_numpy_input_and_outputs_remain_arrays():
    frame, errors = _example()
    array = frame.to_numpy()
    finder = Slicefinder(
        alpha=0.95, k=1, max_l=2, min_sup=1, verbose=False
    ).fit(array, errors)

    assert isinstance(finder.transform(array), np.ndarray)
    assert isinstance(finder.get_slice(array, 0), np.ndarray)


def test_polars_schema_order_is_exact():
    frame, errors = _example()
    finder = Slicefinder(
        alpha=0.95, k=1, max_l=2, min_sup=1, verbose=False
    ).fit(frame, errors)

    with pytest.raises(ValueError, match="schema does not match"):
        finder.transform(frame.select("tier", "region"))


def test_pandas_uses_dataframe_interchange_without_production_import():
    pd = pytest.importorskip("pandas")
    frame, errors = _example()
    pandas_frame = pd.DataFrame(frame.to_dict(as_series=False))

    normalized = normalize_frame(pandas_frame)
    finder = Slicefinder(
        alpha=0.95, k=1, max_l=2, min_sup=1, verbose=False
    ).fit(pandas_frame, errors)

    assert normalized.kind in ("arrow", "interchange")
    assert isinstance(finder.transform(pandas_frame), pl.DataFrame)
    assert isinstance(finder.get_slice(pandas_frame, 0), pl.DataFrame)


def test_arrow_table_uses_arrow_protocol():
    pa = pytest.importorskip("pyarrow")
    frame, errors = _example()
    table = pa.table(frame.to_dict(as_series=False))

    normalized = normalize_frame(table)
    finder = Slicefinder(
        alpha=0.95, k=1, max_l=2, min_sup=1, verbose=False
    ).fit(table, errors)

    assert normalized.kind == "arrow"
    assert isinstance(finder.transform(table), pl.DataFrame)


def test_lazy_input_requires_explicit_collection():
    frame, _ = _example()

    with pytest.raises(TypeError, match="collect it explicitly"):
        normalize_frame(frame.lazy())


def test_copy_free_policy_is_conservative_for_external_producers():
    frame, _ = _example()

    assert normalize_frame(frame, allow_copy=False).frame is frame
    with pytest.raises(RuntimeError, match="only be guaranteed"):
        normalize_frame(frame.to_numpy(), allow_copy=False)


def test_transform_before_fit_has_standard_error():
    frame, _ = _example()

    with pytest.raises(NotFittedError):
        Slicefinder(verbose=False).transform(frame)


@pytest.mark.parametrize(
    "errors",
    [
        pl.Series("loss", [4.0, 1.0, 1.0, 1.0]),
        pl.DataFrame({"loss": [4.0, 1.0, 1.0, 1.0]}),
    ],
)
def test_polars_error_vectors_are_native(errors):
    frame, _ = _example()

    model = Slicefinder(
        alpha=0.95, k=1, max_l=2, min_sup=1, verbose=False
    ).fit(frame, errors)

    assert model.average_error_ == pytest.approx(1.75)


def test_boolean_errors_are_supported_as_indicator_loss():
    frame, _ = _example()

    model = Slicefinder(alpha=0.95, min_sup=1, verbose=False).fit(
        frame, [True, False, False, False]
    )

    assert model.average_error_ == pytest.approx(0.25)


def test_membership_frame_uses_stable_ids_and_row_identity():
    frame, errors = _example()
    finder = Slicefinder(
        alpha=0.95, k=1, max_l=2, min_sup=1, verbose=False
    ).fit(frame, errors)

    membership = finder.membership_frame(
        frame, row_id=pl.Series("record_id", [10, 11, 12, 13])
    )

    assert membership.columns == [
        "__ginsu_row",
        *finder.slices_.get_column("__ginsu_id").to_list(),
    ]
    assert membership.schema["__ginsu_row"] == pl.Int64
    assert all(
        membership.schema[name] == pl.Boolean
        for name in membership.columns[1:]
    )


def test_membership_frame_rejects_non_unique_row_identity():
    frame, errors = _example()
    finder = Slicefinder(
        alpha=0.95, k=1, max_l=2, min_sup=1, verbose=False
    ).fit(frame, errors)

    with pytest.raises(ValueError, match="must be unique"):
        finder.membership_frame(frame, row_id="region")


@pytest.mark.parametrize("bad_name", ["__ginsu_id", "__ginsu_private"])
def test_reserved_names_are_rejected(bad_name):
    with pytest.raises(ValueError, match="reserved"):
        normalize_frame(pl.DataFrame({bad_name: [1]}))
