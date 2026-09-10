"""Tests for auditable score, equivalence, and diversity post-selection."""

import polars as pl
import pytest

from ginsu import (
    AnalysisLimitError,
    SelectionLimits,
    Slicefinder,
    select_slices,
)
from ginsu.selection import SELECTION_SCHEMA


@pytest.fixture
def selection_example(monkeypatch):
    frame = pl.DataFrame(
        {
            "a": [0, 0, 0, 0, 1, 1, 1, 1],
            "b": [0, 0, 1, 1, 0, 0, 1, 1],
            "c": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    finder = Slicefinder(
        alpha=0.95,
        k=20,
        max_l=2,
        min_sup=1,
        verbose=False,
    ).fit(frame, [0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0])
    identifiers = finder.slices_.get_column("__ginsu_id").to_list()
    assert len(identifiers) >= 4
    patterns = [
        [True, True, True, False, False, False, False, False],
        [True, True, True, False, False, False, False, False],
        [True, False, False, True, True, False, False, False],
        [True, True, True, True, False, False, False, False],
    ]
    patterns.extend(
        [[False] * frame.height for _ in range(len(identifiers) - 4)]
    )
    membership = pl.DataFrame(
        {
            "__ginsu_row": pl.Series(range(frame.height), dtype=pl.UInt64),
            **{
                identifier: pl.Series(pattern, dtype=pl.Boolean)
                for identifier, pattern in zip(
                    identifiers, patterns, strict=True
                )
            },
        }
    )
    monkeypatch.setattr(
        finder,
        "membership_frame",
        lambda *_args, **_kwargs: membership.clone(),
    )
    return finder, frame, identifiers


def test_score_selection_preserves_raw_rank(selection_example):
    finder, frame, identifiers = selection_example
    slices_before = finder.slices_.clone()
    statistics_before = finder.slice_statistics_.clone()

    result = finder.select_slices(frame, method="score", k=2)

    assert result.decisions.schema == SELECTION_SCHEMA
    assert result.decisions.get_column("__ginsu_id").to_list() == identifiers
    assert (
        result.selected_slices.get_column("__ginsu_id").to_list()
        == (identifiers[:2])
    )
    assert result.selected_count == 2
    assert set(result.selected_slices["selection_status"]) == {
        "selected_score"
    }
    assert finder.slices_.equals(slices_before)
    assert finder.slice_statistics_.equals(statistics_before)


def test_unique_membership_keeps_first_ranked_representative(
    selection_example,
):
    finder, frame, identifiers = selection_example

    result = select_slices(
        finder,
        frame,
        method="unique_membership",
        k=10,
    )
    duplicate = result.decisions.row(1, named=True)

    assert duplicate["selected"] is False
    assert duplicate["selection_status"] == "excluded_equivalent"
    assert duplicate["representative_id"] == identifiers[0]
    assert duplicate["max_selected_jaccard"] == 1.0
    assert result.selected_slices["discovery_rank"].to_list() == [1, 3, 4]
    assert "GINSU_SELECTION_NO_REFERENCE_MEMBERS" in result.warning_codes


def test_diverse_selection_audits_overlap_and_incremental_coverage(
    selection_example,
):
    finder, frame, identifiers = selection_example

    result = finder.select_slices(
        frame,
        method="diverse",
        k=10,
        max_jaccard=0.5,
    )
    first, duplicate, third, fourth = result.decisions.head(4).iter_rows(
        named=True
    )

    assert result.selected_slices["discovery_rank"].to_list() == [1, 3]
    assert first["candidate_incremental_support_count"] == 3
    assert first["cumulative_selected_support_count"] == 3
    assert duplicate["selection_status"] == "excluded_overlap"
    assert duplicate["representative_id"] == identifiers[0]
    assert third["candidate_incremental_support_count"] == 2
    assert third["cumulative_selected_support_count"] == 5
    assert fourth["selection_status"] == "excluded_overlap"
    assert fourth["representative_id"] == identifiers[0]
    assert fourth["max_selected_jaccard"] == pytest.approx(0.75)


def test_diverse_threshold_is_inclusive(selection_example):
    finder, frame, _ = selection_example

    result = finder.select_slices(
        frame,
        method="diverse",
        k=10,
        max_jaccard=0.75,
    )

    assert result.selected_slices["discovery_rank"].to_list() == [1, 3, 4]


def test_capacity_exclusions_remain_visible(selection_example):
    finder, frame, _ = selection_example

    result = finder.select_slices(
        frame,
        method="diverse",
        k=1,
        max_jaccard=0.5,
    )

    assert result.selected_count == 1
    assert "excluded_capacity" in set(result.decisions["selection_status"])
    assert "GINSU_SELECTION_CAPACITY_REACHED" in result.warning_codes


def test_selection_limits_fail_before_membership_materialization(
    monkeypatch, selection_example
):
    finder, frame, _ = selection_example
    monkeypatch.setattr(
        finder,
        "membership_frame",
        lambda *_args, **_kwargs: pytest.fail("membership was materialized"),
    )

    with pytest.raises(AnalysisLimitError) as raised:
        finder.select_slices(
            frame,
            method="diverse",
            limits=SelectionLimits(max_slices=1),
        )
    assert raised.value.code == "GINSU_MAX_SELECTION_SLICES"

    with pytest.raises(AnalysisLimitError) as raised:
        finder.select_slices(
            frame,
            method="diverse",
            limits=SelectionLimits(max_membership_cells=1),
        )
    assert raised.value.code == "GINSU_MAX_SELECTION_MEMBERSHIP_CELLS"

    with pytest.raises(AnalysisLimitError) as raised:
        finder.select_slices(
            frame,
            method="diverse",
            limits=SelectionLimits(max_pair_comparisons=1),
        )
    assert raised.value.code == "GINSU_MAX_SELECTION_PAIR_COMPARISONS"


def test_score_selection_does_not_require_pairwise_budget(selection_example):
    finder, frame, _ = selection_example

    result = finder.select_slices(
        frame,
        method="score",
        k=2,
        limits=SelectionLimits(max_pair_comparisons=1),
    )

    assert result.selected_count == 2


def test_empty_discoveries_keep_selection_schema():
    frame = pl.DataFrame({"region": ["east", "west"]})
    finder = Slicefinder(min_sup=3, verbose=False).fit(frame, [1.0, 1.0])

    result = finder.select_slices(frame, method="diverse")

    assert result.selected_count == 0
    assert result.decisions.schema == SELECTION_SCHEMA


@pytest.mark.parametrize(
    "kwargs, error_type, message",
    [
        ({"method": "unknown"}, ValueError, "method"),
        ({"k": 0}, ValueError, "positive integer"),
        ({"k": True}, ValueError, "positive integer"),
        (
            {"method": "diverse", "max_jaccard": -0.1},
            ValueError,
            "max_jaccard",
        ),
        (
            {"method": "diverse", "max_jaccard": float("nan")},
            ValueError,
            "max_jaccard",
        ),
        ({"limits": object()}, TypeError, "SelectionLimits"),
    ],
)
def test_invalid_selection_configuration_is_rejected(
    selection_example, kwargs, error_type, message
):
    finder, frame, _ = selection_example

    with pytest.raises(error_type, match=message):
        finder.select_slices(frame, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_slices": 0},
        {"max_membership_cells": True},
        {"max_pair_comparisons": 1.5},
    ],
)
def test_invalid_selection_limits_are_rejected(kwargs):
    with pytest.raises(ValueError, match="positive integer"):
        SelectionLimits(**kwargs)


def test_selection_rejects_schema_drift(selection_example):
    finder, frame, _ = selection_example

    with pytest.raises(ValueError, match="schema does not match"):
        finder.select_slices(frame.select("c", "b", "a"))


def test_selection_is_deterministic(selection_example):
    finder, frame, _ = selection_example

    first = finder.select_slices(frame, method="diverse", max_jaccard=0.5)
    second = finder.select_slices(frame, method="diverse", max_jaccard=0.5)

    assert first.decisions.equals(second.decisions)


def test_selection_supports_arrow_reference_data(selection_example):
    pytest.importorskip("pyarrow")
    finder, frame, _ = selection_example

    result = finder.select_slices(frame.to_arrow(), method="diverse")

    assert result.input_kind == "arrow"
