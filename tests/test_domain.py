"""Tests for canonical slice values and result frames."""

from datetime import date, datetime

import numpy as np
import polars as pl
import pytest

from ginsu import Predicate, Slice, Slicefinder
from ginsu._domain import canonical_json, value_from_canonical_json


def test_slice_id_is_order_independent_and_type_sensitive():
    first = Slice((Predicate("tier", 1), Predicate("region", "east")))
    reordered = Slice((Predicate("region", "east"), Predicate("tier", 1)))
    string_tier = Slice((Predicate("region", "east"), Predicate("tier", "1")))

    assert first == reordered
    assert first.id == reordered.id
    assert first.id.startswith("ginsu:v1:")
    assert first.id != string_tier.id


def test_parent_relationship_is_a_proper_predicate_subset():
    parent = Slice((Predicate("region", "east"),))
    child = Slice((Predicate("region", "east"), Predicate("tier", 1)))

    assert parent.is_parent_of(child)
    assert not child.is_parent_of(parent)
    assert not parent.is_parent_of(parent)


def test_duplicate_features_are_rejected():
    with pytest.raises(ValueError, match="at most one"):
        Slice((Predicate("region", "east"), Predicate("region", "west")))


@pytest.mark.parametrize(
    "value",
    [True, 3, 2.5, "東京", date(2026, 9, 9), datetime(2026, 9, 9, 12, 30)],
)
def test_canonical_values_decode_exactly(value):
    assert value_from_canonical_json(canonical_json(value)) == value


@pytest.mark.parametrize(
    "payload",
    [
        '{"type":"int","value":"01"}',
        '{"type":"int","value":"1","value":"2"}',
        '{"value":"1","type":"int"}',
        '{"type":"unknown","value":"1"}',
    ],
)
def test_noncanonical_predicate_payloads_are_rejected(payload):
    with pytest.raises(ValueError):
        value_from_canonical_json(payload)


def test_fit_builds_canonical_polars_results():
    frame = pl.DataFrame(
        {
            "region": ["east", "east", "west", "west"],
            "tier": [1, 2, 1, 2],
        }
    )
    errors = np.array([4.0, 3.0, 1.0, 1.0])

    finder = Slicefinder(
        alpha=0.95, k=1, max_l=2, min_sup=1, verbose=False
    ).fit(frame, errors)

    assert finder.slices_.columns == [
        "__ginsu_id",
        "__ginsu_rank",
        "__ginsu_rule",
        "region",
        "tier",
    ]
    assert finder.slice_statistics_.height == finder.slices_.height
    assert finder.predicates_.height >= finder.slices_.height
    stats = finder.slice_statistics_.row(0, named=True)
    assert stats["support_fraction"] == pytest.approx(
        stats["support_count"] / frame.height
    )
    assert stats["error_lift"] == pytest.approx(
        stats["error_mean"] / stats["baseline_error_mean"]
    )
    assert stats["excess_error"] == pytest.approx(
        stats["error_sum"]
        - stats["support_count"] * stats["baseline_error_mean"]
    )


def test_empty_results_keep_canonical_schemas():
    frame = pl.DataFrame({"region": ["east", "west"], "tier": [1, 2]})
    finder = Slicefinder(
        alpha=0.95, k=1, max_l=2, min_sup=3, verbose=False
    ).fit(frame, [1.0, 1.0])

    assert finder.slices_.schema == {
        "__ginsu_id": pl.String,
        "__ginsu_rank": pl.UInt32,
        "__ginsu_rule": pl.String,
        "region": pl.String,
        "tier": pl.Int64,
    }
    assert finder.slice_statistics_.schema == {
        "__ginsu_id": pl.String,
        "rank": pl.UInt32,
        "slice_score": pl.Float64,
        "support_count": pl.UInt64,
        "support_fraction": pl.Float64,
        "error_sum": pl.Float64,
        "error_max": pl.Float64,
        "error_mean": pl.Float64,
        "baseline_error_mean": pl.Float64,
        "error_lift": pl.Float64,
        "excess_error": pl.Float64,
        "predicate_count": pl.UInt32,
    }
    assert finder.predicates_.schema == {
        "__ginsu_id": pl.String,
        "rank": pl.UInt32,
        "predicate_position": pl.UInt32,
        "feature": pl.String,
        "operator": pl.String,
        "value_json": pl.String,
        "display_value": pl.String,
        "source_dtype": pl.String,
    }
    assert finder.membership_frame(frame).schema == {"__ginsu_row": pl.UInt64}
