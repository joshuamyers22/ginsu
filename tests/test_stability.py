"""Tests for exact-rule stability aggregation across supplied runs."""

import json

import polars as pl
import pytest

from ginsu import (
    AnalysisLimitError,
    Slicefinder,
    StabilityLimits,
    StabilityRun,
    evaluate_stability,
)
from ginsu.stability import (
    SLICE_RUN_SCHEMA,
    STABILITY_RUN_SCHEMA,
    STABILITY_SUMMARY_SCHEMA,
)


def _finder(region: str, *, scale: float = 1.0) -> Slicefinder:
    frame = pl.DataFrame({"region": ["east"] * 10 + ["west"] * 10})
    if region == "east":
        errors = [5.0 * scale] * 10 + [1.0] * 10
    else:
        errors = [1.0] * 10 + [5.0 * scale] * 10
    return Slicefinder(
        alpha=0.95,
        k=1,
        max_l=1,
        min_sup=2,
        verbose=False,
    ).fit(frame, errors)


@pytest.fixture
def stability_runs():
    first = StabilityRun.from_finder(
        _finder("east"),
        run_id="fold-1",
        partition_id="fold-1-train",
        resampling_unit="customer",
        seed=1,
    )
    second = StabilityRun.from_finder(
        _finder("east", scale=1.1),
        run_id="fold-2",
        partition_id="fold-2-train",
        resampling_unit="customer",
        seed=2,
    )
    third = StabilityRun.from_finder(
        _finder("west"),
        run_id="fold-3",
        partition_id="fold-3-train",
        resampling_unit="customer",
        seed=3,
    )
    failed = StabilityRun.unavailable(
        run_id="fold-4",
        status="limit_reached",
        partition_id="fold-4-train",
        resampling_unit="customer",
        seed=4,
        failure_reason="candidate limit reached",
        parameters={"max_l": 2},
    )
    return [first, second, third, failed]


def test_stability_aggregates_frequency_and_metrics(stability_runs):
    report = evaluate_stability(
        stability_runs,
        stable_frequency=0.6,
    )

    assert report.runs.schema == STABILITY_RUN_SCHEMA
    assert report.slice_runs.schema == SLICE_RUN_SCHEMA
    assert report.summary.schema == STABILITY_SUMMARY_SCHEMA
    assert report.runs.height == 4
    assert report.summary.height == 2
    east_id = stability_runs[0].observations["__ginsu_id"][0]
    east = report.summary.filter(pl.col("__ginsu_id") == east_id).row(
        0, named=True
    )
    assert east["attempted_run_count"] == 4
    assert east["successful_run_count"] == 3
    assert east["unavailable_run_count"] == 1
    assert east["selected_run_count"] == 2
    assert east["selection_frequency_successful"] == pytest.approx(2 / 3)
    assert east["selection_frequency_attempted"] == pytest.approx(0.5)
    assert east["rank_mean"] == 1.0
    assert east["rank_std"] == 0.0
    assert east["slice_score_std"] > 0
    assert east["stability_status"] == "stable"
    assert "GINSU_STABILITY_UNAVAILABLE_RUNS" in report.warning_codes


def test_failed_runs_are_unknown_not_false_absences(stability_runs):
    report = evaluate_stability(stability_runs)
    unavailable = report.slice_runs.filter(pl.col("run_id") == "fold-4")

    assert unavailable.height == report.summary.height
    assert unavailable["selected"].null_count() == unavailable.height
    assert unavailable["rank"].null_count() == unavailable.height
    assert set(unavailable["run_status"]) == {"limit_reached"}


def test_fragile_status_does_not_remove_the_rule(stability_runs):
    report = evaluate_stability(stability_runs, stable_frequency=0.8)

    assert set(report.summary["stability_status"]) == {"fragile"}
    assert set(report.summary["selected_run_count"]) == {1, 2}


def test_insufficient_successful_runs_are_explicit():
    run = StabilityRun.from_finder(
        _finder("east"),
        run_id="only-run",
        partition_id="train",
        resampling_unit="row",
    )

    report = evaluate_stability([run], minimum_successful_runs=2)

    assert set(report.summary["stability_status"]) == {
        "insufficient_successful_runs"
    }
    assert "GINSU_STABILITY_INSUFFICIENT_RUNS" in report.warning_codes


def test_no_valid_slices_count_as_a_successful_observation():
    frame = pl.DataFrame({"region": ["east", "west"]})
    empty_finder = Slicefinder(min_sup=3, verbose=False).fit(frame, [1.0, 1.0])
    empty = StabilityRun.from_finder(
        empty_finder,
        run_id="empty",
        partition_id="empty-train",
        resampling_unit="row",
    )
    selected = StabilityRun.from_finder(
        _finder("east"),
        run_id="selected",
        partition_id="selected-train",
        resampling_unit="row",
    )

    report = evaluate_stability([empty, selected])

    assert empty.status == "no_valid_slices"
    assert report.summary["successful_run_count"][0] == 2
    assert report.summary["selection_frequency_successful"][0] == 0.5


def test_all_unavailable_runs_remain_visible_with_empty_summary():
    failed = StabilityRun.unavailable(
        run_id="failed",
        status="failed",
        partition_id="train",
        resampling_unit="account",
        failure_reason="worker failed",
    )

    report = evaluate_stability([failed])

    assert report.runs.height == 1
    assert report.slice_runs.is_empty()
    assert report.summary.is_empty()
    assert report.summary.schema == STABILITY_SUMMARY_SCHEMA
    assert set(report.warning_codes) == {
        "GINSU_STABILITY_UNAVAILABLE_RUNS",
        "GINSU_STABILITY_INSUFFICIENT_RUNS",
    }


def test_run_records_seed_partition_unit_and_canonical_parameters():
    run = StabilityRun.from_finder(
        _finder("east"),
        run_id="run",
        partition_id="train-v1",
        resampling_unit="customer",
        seed=42,
        parameters={"z": True, "alpha": 0.95, "label": "candidate"},
    )

    assert run.parameters_json == (
        '{"alpha":0.95,"label":"candidate","z":true}'
    )
    report = evaluate_stability([run])
    row = report.runs.row(0, named=True)
    assert row["partition_id"] == "train-v1"
    assert row["resampling_unit"] == "customer"
    assert row["seed"] == 42
    assert json.loads(row["parameters_json"])["alpha"] == 0.95


def test_stability_limits_fail_before_cross_run_materialization(
    stability_runs,
):
    with pytest.raises(AnalysisLimitError) as raised:
        evaluate_stability(stability_runs, limits=StabilityLimits(max_runs=1))
    assert raised.value.code == "GINSU_MAX_STABILITY_RUNS"

    with pytest.raises(AnalysisLimitError) as raised:
        evaluate_stability(
            stability_runs,
            limits=StabilityLimits(max_union_slices=1),
        )
    assert raised.value.code == "GINSU_MAX_STABILITY_SLICES"

    with pytest.raises(AnalysisLimitError) as raised:
        evaluate_stability(
            stability_runs,
            limits=StabilityLimits(max_slice_run_cells=1),
        )
    assert raised.value.code == "GINSU_MAX_STABILITY_SLICE_RUN_CELLS"


@pytest.mark.parametrize(
    "kwargs, error_type, message",
    [
        ({"runs": []}, ValueError, "at least one"),
        ({"runs": [object()]}, TypeError, "StabilityRun"),
        (
            {"runs": None},
            TypeError,
            "sequence",
        ),
        ({"minimum_successful_runs": 0}, ValueError, "positive integer"),
        ({"stable_frequency": 0}, ValueError, "stable_frequency"),
        ({"stable_frequency": float("nan")}, ValueError, "stable_frequency"),
        ({"limits": object()}, TypeError, "StabilityLimits"),
    ],
)
def test_invalid_stability_configuration_is_rejected(
    stability_runs, kwargs, error_type, message
):
    arguments = {"runs": stability_runs, **kwargs}
    with pytest.raises(error_type, match=message):
        evaluate_stability(**arguments)


def test_duplicate_run_ids_are_rejected(stability_runs):
    with pytest.raises(ValueError, match="unique"):
        evaluate_stability([stability_runs[0], stability_runs[0]])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_runs": 0},
        {"max_union_slices": True},
        {"max_slice_run_cells": 1.5},
    ],
)
def test_invalid_stability_limits_are_rejected(kwargs):
    with pytest.raises(ValueError, match="positive integer"):
        StabilityLimits(**kwargs)


def test_invalid_run_metadata_and_parameters_are_rejected():
    finder = _finder("east")
    common = {
        "finder": finder,
        "run_id": "run",
        "partition_id": "train",
        "resampling_unit": "row",
    }
    with pytest.raises(ValueError, match="run_id"):
        StabilityRun.from_finder(**{**common, "run_id": ""})
    with pytest.raises(ValueError, match="seed"):
        StabilityRun.from_finder(**common, seed=-1)
    with pytest.raises(TypeError, match="JSON scalar"):
        StabilityRun.from_finder(**common, parameters={"bad": []})
    with pytest.raises(ValueError, match="finite"):
        StabilityRun.from_finder(**common, parameters={"bad": float("inf")})


def test_unavailable_run_requires_failure_status_and_reason():
    common = {
        "run_id": "run",
        "partition_id": "train",
        "resampling_unit": "row",
    }
    with pytest.raises(ValueError, match="Unavailable status"):
        StabilityRun.unavailable(
            **common,
            status="complete",
            failure_reason="not allowed",
        )
    with pytest.raises(ValueError, match="failure reason"):
        StabilityRun.unavailable(
            **common,
            status="failed",
            failure_reason="",
        )


def test_inconsistent_rule_for_a_canonical_id_is_rejected(stability_runs):
    original = stability_runs[0]
    changed = original.observations.with_columns(
        pl.lit("different display rule").alias("__ginsu_rule")
    )
    inconsistent = StabilityRun(
        run_id="inconsistent",
        status="complete",
        partition_id="train",
        resampling_unit="row",
        seed=None,
        parameters_json="{}",
        observations=changed,
    )

    with pytest.raises(ValueError, match="inconsistent display rules"):
        evaluate_stability([original, inconsistent])
