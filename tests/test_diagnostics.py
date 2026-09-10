from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import polars as pl
import pytest
from sklearn.exceptions import NotFittedError

from ginsu import (
    SearchLimitError,
    SearchLimits,
    SearchStageReport,
    Slicefinder,
)


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
    assert [stage.stage for stage in report.stages] == [
        "input_normalization",
        "engine_conversion",
        "one_hot_encoding",
        "level_1_evaluation",
        "level_2_join",
        "level_2_evaluation",
        "result_materialization",
    ]
    assert {stage.status for stage in report.stages} == {"complete"}
    assert report.memory_measurement is None
    assert report.observed_peak_memory_bytes is None
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
    assert candidate_limited.search_report_.stages[-1].stage == "level_2_join"
    assert candidate_limited.search_report_.stages[-1].status == "terminated"

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
    ticks = iter((0.0, 2.0))
    finder = Slicefinder(
        limits=SearchLimits(max_search_seconds=1.0),
        clock=lambda: next(ticks),
        verbose=False,
    )

    with pytest.raises(SearchLimitError) as raised:
        finder.fit(pl.DataFrame({"a": [0, 1]}), [1.0, 2.0])

    assert raised.value.code == "GINSU_MAX_SEARCH_SECONDS"
    assert finder.search_report_.elapsed_seconds == 2.0
    assert finder.search_report_.stages == (
        SearchStageReport(
            stage="input_normalization",
            status="complete",
            elapsed_seconds=2.0,
        ),
    )


def test_stage_timing_and_memory_sampling_are_deterministic() -> None:
    ticks = iter(float(value) for value in range(9))
    memory = iter((100, 110, 120, 130, 140, 150, 170, 160))
    finder = Slicefinder(
        alpha=1.0,
        k=2,
        max_l=2,
        min_sup=1,
        clock=lambda: next(ticks),
        memory_sampler=lambda: next(memory),
        memory_measurement="test_boundary_bytes",
        verbose=False,
    )
    frame = pl.DataFrame(
        {"segment": ["a", "a", "b", "b"], "region": [1, 2, 1, 2]}
    )

    finder.fit(frame, [1.0, 1.0, 0.1, 0.1])

    report = finder.search_report_
    assert report.elapsed_seconds == 8.0
    assert [stage.elapsed_seconds for stage in report.stages] == [
        1.0,
        1.0,
        1.0,
        1.0,
        2.0,
        1.0,
        1.0,
    ]
    assert [stage.memory_start_bytes for stage in report.stages] == [
        100,
        110,
        120,
        130,
        140,
        150,
        170,
    ]
    assert [stage.memory_end_bytes for stage in report.stages] == [
        110,
        120,
        130,
        140,
        150,
        170,
        160,
    ]
    assert report.memory_measurement == "test_boundary_bytes"
    assert report.observed_peak_memory_bytes == 170


@pytest.mark.parametrize(
    "kwargs,error_type,message",
    [
        ({"memory_sampler": 1}, TypeError, "callable"),
        ({"memory_sampler": lambda: 1}, ValueError, "provided together"),
        (
            {"memory_measurement": "process_rss_bytes"},
            ValueError,
            "provided together",
        ),
        (
            {"memory_sampler": lambda: 1, "memory_measurement": ""},
            ValueError,
            "non-empty string",
        ),
        (
            {"memory_sampler": lambda: 1, "memory_measurement": "   "},
            ValueError,
            "non-empty string",
        ),
    ],
)
def test_memory_diagnostic_configuration_is_validated(
    kwargs, error_type, message
) -> None:
    finder = Slicefinder(verbose=False, **kwargs)
    with pytest.raises(error_type, match=message):
        finder.fit(pl.DataFrame({"a": [0, 1]}), [1.0, 2.0])


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_memory_sampler_requires_nonnegative_integer_bytes(value) -> None:
    finder = Slicefinder(
        memory_sampler=lambda: value,
        memory_measurement="test_bytes",
        verbose=False,
    )
    with pytest.raises(ValueError, match="nonnegative integer"):
        finder.fit(pl.DataFrame({"a": [0, 1]}), [1.0, 2.0])


def test_memory_sampler_failure_preserves_terminated_stage_evidence() -> None:
    ticks = iter((0.0, 1.0, 2.0))
    calls = iter((100, RuntimeError("sampler unavailable")))

    def sample_memory() -> int:
        value = next(calls)
        if isinstance(value, Exception):
            raise value
        return value

    finder = Slicefinder(
        clock=lambda: next(ticks),
        memory_sampler=sample_memory,
        memory_measurement="test_boundary_bytes",
        verbose=False,
    )

    with pytest.raises(RuntimeError, match="sampler unavailable"):
        finder.fit(pl.DataFrame({"a": [0, 1]}), [1.0, 2.0])

    assert finder.search_report_.status == "failed"
    assert finder.search_report_.elapsed_seconds == 2.0
    assert finder.search_report_.stages == (
        SearchStageReport(
            stage="input_normalization",
            status="terminated",
            elapsed_seconds=2.0,
            memory_start_bytes=100,
            observed_peak_memory_bytes=100,
        ),
    )
    assert finder.search_report_.observed_peak_memory_bytes == 100


def test_stage_report_rejects_inconsistent_boundary_peak() -> None:
    with pytest.raises(ValueError, match="largest boundary"):
        SearchStageReport(
            stage="input_normalization",
            status="complete",
            elapsed_seconds=1.0,
            memory_start_bytes=10,
            memory_end_bytes=20,
            observed_peak_memory_bytes=10,
        )


def test_search_report_rejects_inconsistent_stage_sequence() -> None:
    finder = Slicefinder(min_sup=1, verbose=False).fit(
        pl.DataFrame({"a": [0, 1]}), [1.0, 2.0]
    )
    report = finder.search_report_

    with pytest.raises(ValueError, match="elapsed time"):
        replace(report, elapsed_seconds=report.elapsed_seconds + 1.0)
    with pytest.raises(ValueError, match="unique"):
        replace(report, stages=(report.stages[0], report.stages[0]))
    terminated = replace(report.stages[0], status="terminated")
    with pytest.raises(ValueError, match="final search stage"):
        replace(report, stages=(terminated, *report.stages[1:]))


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
