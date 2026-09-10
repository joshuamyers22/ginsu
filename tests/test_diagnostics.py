from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import polars as pl
import pytest
from sklearn.exceptions import NotFittedError

from ginsu import SearchLimitError, SearchLimits, Slicefinder


def test_successful_fit_exposes_immutable_search_report() -> None:
    features = pl.DataFrame(
        {"segment": ["a", "a", "b", "b"], "region": [1, 2, 1, 2]}
    )
    finder = Slicefinder(alpha=1.0, k=2, max_l=2, min_sup=1, verbose=False)

    finder.fit(features, [1.0, 1.0, 0.1, 0.1])

    report = finder.search_report_
    assert report.status == "complete"
    assert report.exhaustive
    assert report.row_count == 4
    assert report.feature_count == 2
    assert report.encoded_feature_count == 4
    assert report.feature_cardinalities == (("segment", 2), ("region", 2))
    assert report.copy_boundaries == ("polars->numpy", "numpy->scipy-csr")
    assert report.levels[0].level == 1
    assert report.levels[1].potential_pairs == 6
    assert report.elapsed_seconds >= 0
    with pytest.raises(FrozenInstanceError):
        report.status = "failed"  # type: ignore[misc]


def test_no_valid_slices_is_an_exhaustive_outcome() -> None:
    finder = Slicefinder(alpha=1.0, min_sup=1, verbose=False)
    finder.fit(np.ones((4, 1)), np.ones(4))

    assert finder.search_report_.status == "no_valid_slices"
    assert finder.search_report_.exhaustive


def test_cardinality_limit_fails_before_engine_conversion() -> None:
    finder = Slicefinder(
        limits=SearchLimits(max_feature_cardinality=2), verbose=False
    )

    with pytest.raises(SearchLimitError) as raised:
        finder.fit(pl.DataFrame({"feature": [1, 2, 3]}), [1.0, 1.0, 1.0])

    assert raised.value.code == "GINSU_MAX_FEATURE_CARDINALITY"
    assert finder.search_report_.status == "limit_reached"
    assert not finder.search_report_.exhaustive
    assert finder.search_report_.encoded_feature_count is None
    assert finder.search_report_.warning_codes == (
        "GINSU_MAX_FEATURE_CARDINALITY",
    )
    with pytest.raises(NotFittedError):
        finder.get_feature_names_out()


def test_encoded_feature_limit_is_reported() -> None:
    finder = Slicefinder(
        limits=SearchLimits(
            max_feature_cardinality=None,
            max_encoded_features=3,
        ),
        verbose=False,
    )

    with pytest.raises(SearchLimitError) as raised:
        finder.fit(
            pl.DataFrame({"left": [1, 2], "right": [1, 2]}),
            [1.0, 2.0],
        )

    assert raised.value.code == "GINSU_MAX_ENCODED_FEATURES"
    assert finder.search_report_.encoded_feature_count == 4


def test_dense_pair_estimate_fails_before_compatibility_allocation() -> None:
    features = pl.DataFrame(
        {
            "a": [0, 0, 0, 0, 1, 1, 1, 1],
            "b": [0, 0, 1, 1, 0, 0, 1, 1],
            "c": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    finder = Slicefinder(
        max_l=2,
        min_sup=1,
        limits=SearchLimits(max_pair_matrix_bytes=100),
        verbose=False,
    )

    with pytest.raises(SearchLimitError) as raised:
        finder.fit(features, np.ones(8))

    assert raised.value.code == "GINSU_MAX_PAIR_MATRIX_BYTES"
    assert raised.value.observed == 360
    assert finder.search_report_.levels[0].level == 1


def test_candidate_and_tie_limits_have_stable_codes() -> None:
    features = pl.DataFrame(
        {
            "a": [0, 0, 0, 0, 1, 1, 1, 1],
            "b": [0, 0, 1, 1, 0, 0, 1, 1],
            "c": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    candidate_limited = Slicefinder(
        max_l=2,
        min_sup=1,
        limits=SearchLimits(max_candidates_per_level=1),
        verbose=False,
    )
    with pytest.raises(SearchLimitError) as candidate_error:
        candidate_limited.fit(features, np.ones(8))
    assert candidate_error.value.code == "GINSU_MAX_CANDIDATES_PER_LEVEL"

    tied = pl.DataFrame(
        {"a": [0, 0, 1, 1], "b": [0, 0, 1, 1], "c": [0, 0, 1, 1]}
    )
    tie_limited = Slicefinder(
        alpha=1.0,
        k=1,
        min_sup=1,
        limits=SearchLimits(max_tied_slices=2),
        verbose=False,
    )
    with pytest.raises(SearchLimitError) as tie_error:
        tie_limited.fit(tied, [0.0, 0.0, 1.0, 1.0])
    assert tie_error.value.code == "GINSU_MAX_TIED_SLICES"


def test_elapsed_limit_uses_injected_monotonic_clock() -> None:
    ticks = iter((0.0, 2.0, 3.0))
    finder = Slicefinder(
        limits=SearchLimits(max_search_seconds=1.0),
        clock=lambda: next(ticks),
        verbose=False,
    )

    with pytest.raises(SearchLimitError) as raised:
        finder.fit(pl.DataFrame({"a": [0, 1]}), [1.0, 2.0])

    assert raised.value.code == "GINSU_MAX_SEARCH_SECONDS"
    assert finder.search_report_.elapsed_seconds == 3.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_feature_cardinality": 0},
        {"max_encoded_features": -1},
        {"max_search_seconds": True},
    ],
)
def test_search_limits_require_positive_values(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="must be positive or None"):
        SearchLimits(**kwargs)  # type: ignore[arg-type]
