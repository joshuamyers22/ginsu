"""Tests for exact-rule stability aggregation across supplied runs."""

import json

import polars as pl
import pytest

from ginsu import (
    AnalysisLimitError,
    Slicefinder,
    StabilityLimits,
    StabilityReferenceLimits,
    StabilityRun,
    evaluate_similarity_stability,
    evaluate_stability,
)
from ginsu.stability import (
    RUN_PREDICATE_SCHEMA,
    RUN_REFERENCE_MEMBERSHIP_SCHEMA,
    SIMILARITY_MATCH_SCHEMA,
    SIMILARITY_SUMMARY_SCHEMA,
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


def _two_feature_finder(target: str) -> Slicefinder:
    frame = pl.DataFrame(
        {
            "a": [0, 0, 0, 0, 1, 1, 1, 1],
            "b": [0, 0, 1, 1, 0, 0, 1, 1],
        }
    )
    if target == "a":
        errors = [5.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0]
    elif target == "b":
        errors = [5.0, 5.0, 1.0, 1.0, 5.0, 5.0, 1.0, 1.0]
    else:
        errors = [1.0, 1.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0]
    return Slicefinder(
        alpha=0.95,
        k=1,
        max_l=2,
        min_sup=1,
        verbose=False,
    ).fit(frame, errors)


def _similarity_runs(reference: pl.DataFrame | None = None):
    kwargs = {}
    if reference is not None:
        kwargs = {
            "reference_frame": reference,
            "reference_id": "reference-v1",
            "reference_row_id": pl.Series("row", range(reference.height)),
        }
    return [
        StabilityRun.from_finder(
            _two_feature_finder("a"),
            run_id="run-a",
            partition_id="train-a",
            resampling_unit="account",
            **kwargs,
        ),
        StabilityRun.from_finder(
            _two_feature_finder("b"),
            run_id="run-b",
            partition_id="train-b",
            resampling_unit="account",
            **kwargs,
        ),
    ]


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


def test_from_finder_captures_predicate_sets_for_similarity():
    run = _similarity_runs()[0]

    assert run.predicates.schema == RUN_PREDICATE_SCHEMA
    assert run.predicates.height == run.observations.height == 1
    assert len(run.predicates["predicate_tokens"][0]) == 1


def test_predicate_similarity_matches_related_rules_without_merging_ids():
    related = StabilityRun.from_finder(
        _two_feature_finder("intersection"),
        run_id="intersection",
        partition_id="train-intersection",
        resampling_unit="account",
    )
    broad = StabilityRun.from_finder(
        _two_feature_finder("a"),
        run_id="broad",
        partition_id="train-broad",
        resampling_unit="account",
    )

    report = evaluate_similarity_stability(
        [related, broad],
        method="predicate",
        similarity_threshold=0.5,
    )

    assert report.matches.schema == SIMILARITY_MATCH_SCHEMA
    assert report.summary.schema == SIMILARITY_SUMMARY_SCHEMA
    anchor_id = related.observations["__ginsu_id"][0]
    summary = report.summary.filter(pl.col("anchor_id") == anchor_id).row(
        0, named=True
    )
    assert summary["matched_run_count"] == 2
    assert summary["exact_run_count"] == 1
    assert summary["match_frequency_successful"] == 1.0
    assert summary["similarity_min"] == 0.5
    assert (
        report.exact.summary.filter(pl.col("__ginsu_id") == anchor_id)[
            "selection_frequency_successful"
        ][0]
        == 0.5
    )


def test_membership_similarity_uses_verified_common_reference():
    reference = pl.DataFrame({"a": [0, 0, 1, 1], "b": [0, 0, 1, 1]})
    runs = _similarity_runs(reference)

    predicate = evaluate_similarity_stability(
        runs, method="predicate", similarity_threshold=0.8
    )
    membership = evaluate_similarity_stability(
        runs, method="membership", similarity_threshold=0.8
    )

    assert set(predicate.summary["match_frequency_successful"]) == {0.5}
    assert set(membership.summary["match_frequency_successful"]) == {1.0}
    assert set(membership.summary["similarity_min"]) == {1.0}
    assert membership.reference_id == "reference-v1"
    assert membership.reference_row_count == reference.height
    assert membership.reference_row_fingerprint.startswith(
        "ginsu:reference-rows:v1:"
    )
    assert runs[0].reference_memberships.schema == (
        RUN_REFERENCE_MEMBERSHIP_SCHEMA
    )


def test_membership_similarity_rejects_misaligned_reference_rows():
    reference = pl.DataFrame({"a": [0, 0, 1, 1], "b": [0, 0, 1, 1]})
    first, _ = _similarity_runs(reference)
    second = StabilityRun.from_finder(
        _two_feature_finder("b"),
        run_id="run-b",
        partition_id="train-b",
        resampling_unit="account",
        reference_frame=reference,
        reference_id="reference-v1",
        reference_row_id=pl.Series("row", [1, 0, 2, 3]),
    )

    with pytest.raises(ValueError, match="ordered reference-row"):
        evaluate_similarity_stability([first, second], method="membership")


def test_membership_similarity_requires_captured_reference():
    with pytest.raises(ValueError, match="no common-reference"):
        evaluate_similarity_stability(_similarity_runs(), method="membership")


def test_similarity_retains_unavailable_runs_as_unknown():
    run = _similarity_runs()[0]
    failed = StabilityRun.unavailable(
        run_id="failed",
        status="failed",
        partition_id="failed-train",
        resampling_unit="account",
        failure_reason="worker failed",
    )

    report = evaluate_similarity_stability(
        [run, failed], method="predicate", minimum_successful_runs=1
    )
    unavailable = report.matches.filter(pl.col("run_id") == "failed")

    assert unavailable["evidence_available"].to_list() == [False]
    assert unavailable["matched"].to_list() == [None]
    assert unavailable["similarity"].to_list() == [None]
    assert report.summary["match_frequency_successful"][0] == 1.0


def test_empty_reference_union_is_unknown_not_perfect_overlap():
    reference = pl.DataFrame({"a": [1, 1], "b": [1, 1]})
    report = evaluate_similarity_stability(
        _similarity_runs(reference),
        method="membership",
        similarity_threshold=0.5,
    )

    assert report.matches["similarity"].null_count() == report.matches.height
    assert report.summary["matched_run_count"].sum() == 0
    assert "GINSU_STABILITY_EMPTY_REFERENCE_MEMBERSHIP" in (
        report.warning_codes
    )


def test_similarity_comparison_limit_fails_before_pairwise_work():
    runs = _similarity_runs()

    with pytest.raises(AnalysisLimitError) as raised:
        evaluate_similarity_stability(
            runs,
            limits=StabilityLimits(max_similarity_comparisons=1),
        )

    assert raised.value.code == ("GINSU_MAX_STABILITY_SIMILARITY_COMPARISONS")


def test_reference_capture_limits_fail_before_membership(monkeypatch):
    finder = _two_feature_finder("a")
    reference = pl.DataFrame({"a": [0, 1], "b": [0, 1]})
    monkeypatch.setattr(
        finder,
        "membership_frame",
        lambda *_args, **_kwargs: pytest.fail("membership was materialized"),
    )

    with pytest.raises(AnalysisLimitError) as raised:
        StabilityRun.from_finder(
            finder,
            run_id="run",
            partition_id="train",
            resampling_unit="row",
            reference_frame=reference,
            reference_id="reference-v1",
            reference_row_id=pl.Series("row", [0, 1]),
            reference_limits=StabilityReferenceLimits(max_rows=1),
        )

    assert raised.value.code == "GINSU_MAX_STABILITY_REFERENCE_ROWS"


@pytest.mark.parametrize(
    "kwargs, error_type, message",
    [
        ({"method": "cosine"}, ValueError, "method"),
        ({"similarity_threshold": 0}, ValueError, "similarity_threshold"),
        (
            {"similarity_threshold": float("nan")},
            ValueError,
            "similarity_threshold",
        ),
    ],
)
def test_invalid_similarity_configuration_is_rejected(
    kwargs, error_type, message
):
    with pytest.raises(error_type, match=message):
        evaluate_similarity_stability(_similarity_runs(), **kwargs)
