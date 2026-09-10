"""Tests for auditable one-to-one analysis comparison."""

import polars as pl
import pytest

from ginsu import (
    AnalysisLimitError,
    ComparisonLimits,
    DiscretizationPlan,
    FixedBins,
    SliceAnalysis,
    Slicefinder,
    compare_analyses,
)
from ginsu.comparison import COMPARISON_SCHEMA, COMPARISON_SUMMARY_SCHEMA


def _analysis(target: str, *, scale: float = 1.0, k: int = 1) -> SliceAnalysis:
    frame = pl.DataFrame(
        {
            "a": [0, 0, 0, 0, 1, 1, 1, 1],
            "b": [0, 0, 1, 1, 0, 0, 1, 1],
        }
    )
    if target == "a":
        errors = [5.0 * scale] * 4 + [1.0] * 4
    elif target == "b":
        errors = [5.0 * scale, 5.0 * scale, 1.0, 1.0] * 2
    else:
        errors = [1.0, 1.0, 5.0 * scale, 5.0 * scale, 1.0, 1.0, 1.0, 1.0]
    finder = Slicefinder(
        alpha=0.95,
        k=k,
        max_l=2,
        min_sup=1,
        verbose=False,
    ).fit(frame, errors)
    return SliceAnalysis.from_finder(
        finder, dataset_fingerprint=f"sha256:{target}-{scale}"
    )


def test_exact_comparison_reports_metric_deltas_and_direction():
    baseline = _analysis("a")
    candidate = _analysis("a", scale=1.5)

    comparison = compare_analyses(baseline, candidate, method="exact")

    assert comparison.comparable
    assert comparison.changes.schema == COMPARISON_SCHEMA
    assert comparison.summary.schema == COMPARISON_SUMMARY_SCHEMA
    assert comparison.changes.height == 1
    row = comparison.changes.row(0, named=True)
    assert row["match_type"] == "exact"
    assert row["similarity"] == 1.0
    assert row["comparison_status"] == "regressed"
    assert row["error_lift_delta"] > 0
    assert row["support_fraction_delta"] == 0
    assert comparison.summary["exact_match_count"][0] == 1
    assert comparison.summary["regressed_count"][0] == 1
    assert comparison.warning_codes == (
        "GINSU_COMPARISON_NO_REFERENCE_MEMBERSHIP",
    )


def test_predicate_matching_is_one_to_one_and_keeps_drift():
    baseline = _analysis("intersection")
    candidate = _analysis("a")

    comparison = compare_analyses(
        baseline,
        candidate,
        method="predicate",
        similarity_threshold=0.5,
    )

    row = comparison.changes.row(0, named=True)
    assert row["match_type"] == "predicate"
    assert row["similarity"] == 0.5
    assert row["shared_predicate_count"] == 1
    assert len(row["removed_predicates"]) == 1
    assert row["added_predicates"] == []
    assert comparison.summary["related_match_count"][0] == 1


def test_unmatched_rules_are_explicitly_emerged_and_resolved():
    comparison = compare_analyses(
        _analysis("a"),
        _analysis("b"),
        method="predicate",
        similarity_threshold=0.8,
    )

    assert set(comparison.changes["comparison_status"]) == {
        "emerged",
        "resolved",
    }
    assert set(comparison.changes["match_type"]) == {"unmatched"}
    assert comparison.summary["emerged_count"][0] == 1
    assert comparison.summary["resolved_count"][0] == 1
    assert comparison.changes["error_lift_delta"].null_count() == 2


def test_membership_matching_uses_one_common_reference_frame():
    reference = pl.DataFrame({"a": [0, 0, 1, 1], "b": [0, 0, 1, 1]})
    comparison = compare_analyses(
        _analysis("a"),
        _analysis("b"),
        method="membership",
        reference_data=reference,
        reference_id="sha256:reference-v1",
    )

    row = comparison.changes.row(0, named=True)
    assert row["match_type"] == "membership"
    assert row["similarity"] == 1.0
    assert row["reference_jaccard"] == 1.0
    assert row["reference_both_count"] == 2
    assert row["reference_neither_count"] == 2
    assert comparison.reference_row_count == 4
    assert comparison.summary["reference_id"][0] == "sha256:reference-v1"


def test_reference_migration_counts_are_retained_for_predicate_match():
    reference = pl.DataFrame(
        {
            "a": [0, 0, 0, 0, 1, 1, 1, 1],
            "b": [0, 0, 1, 1, 0, 0, 1, 1],
        }
    )
    comparison = compare_analyses(
        _analysis("intersection"),
        _analysis("a"),
        method="predicate",
        similarity_threshold=0.5,
        reference_data=reference,
        reference_id="reference-v1",
    )

    row = comparison.changes.row(0, named=True)
    assert row["reference_baseline_only_count"] == 0
    assert row["reference_candidate_only_count"] == 2
    assert row["reference_both_count"] == 2
    assert row["reference_union_count"] == 4
    assert row["reference_jaccard"] == 0.5


def test_direction_tolerance_can_label_small_change_unchanged():
    comparison = compare_analyses(
        _analysis("a"),
        _analysis("a", scale=1.01),
        method="exact",
        direction_tolerance=0.1,
    )

    assert comparison.changes["comparison_status"].to_list() == ["unchanged"]


def test_incompatible_schema_returns_non_comparable_report():
    left = _analysis("a")
    frame = pl.DataFrame({"different": [0] * 10 + [1] * 10})
    finder = Slicefinder(
        alpha=0.95, k=1, max_l=1, min_sup=2, verbose=False
    ).fit(frame, [5.0] * 10 + [1.0] * 10)
    right = SliceAnalysis.from_finder(
        finder, dataset_fingerprint="sha256:different"
    )

    comparison = compare_analyses(left, right)

    assert not comparison.comparable
    assert comparison.changes.is_empty()
    assert comparison.summary["compatibility_status"][0] == (
        "incompatible_feature_schema"
    )
    assert comparison.warning_codes == ("GINSU_COMPARISON_NOT_COMPARABLE",)


def test_recorded_discretization_is_applied_to_raw_reference():
    raw = pl.DataFrame({"age": [10, 20, 30, 40, 50, 60, 70, 80]})
    plan = DiscretizationPlan(numeric={"age": FixedBins((45.0,))}).fit(raw)
    features = plan.transform(raw)
    finder = Slicefinder(
        alpha=0.95, k=1, max_l=1, min_sup=1, verbose=False
    ).fit(features, [5.0] * 4 + [1.0] * 4)
    baseline = SliceAnalysis.from_finder(
        finder,
        dataset_fingerprint="sha256:baseline",
        discretization=plan,
    )
    candidate = SliceAnalysis.from_finder(
        finder,
        dataset_fingerprint="sha256:candidate",
        discretization=plan,
    )

    comparison = compare_analyses(
        baseline,
        candidate,
        method="exact",
        reference_data=raw,
        reference_id="raw-reference",
    )

    assert comparison.comparable
    assert comparison.changes["reference_both_count"][0] == 4
    assert comparison.changes["reference_jaccard"][0] == 1.0


def test_different_discretization_specs_are_non_comparable():
    raw = pl.DataFrame({"age": [10, 20, 30, 40, 50, 60, 70, 80]})

    def build(boundary: float) -> SliceAnalysis:
        plan = DiscretizationPlan(numeric={"age": FixedBins((boundary,))}).fit(
            raw
        )
        finder = Slicefinder(
            alpha=0.95, k=1, max_l=1, min_sup=1, verbose=False
        ).fit(plan.transform(raw), [5.0] * 4 + [1.0] * 4)
        return SliceAnalysis.from_finder(
            finder,
            dataset_fingerprint=f"sha256:{boundary}",
            discretization=plan,
        )

    comparison = compare_analyses(build(35.0), build(45.0))

    assert not comparison.comparable
    assert comparison.summary["compatibility_status"][0] == (
        "incompatible_discretization"
    )


def test_comparison_limits_fail_fail_before_pair_work():
    baseline = _analysis("a", k=10)
    candidate = _analysis("b", k=10)

    with pytest.raises(AnalysisLimitError) as raised:
        compare_analyses(
            baseline,
            candidate,
            limits=ComparisonLimits(max_pair_comparisons=1),
        )
    assert raised.value.code == "GINSU_MAX_COMPARISON_PAIRS"

    with pytest.raises(AnalysisLimitError) as raised:
        compare_analyses(
            baseline,
            candidate,
            reference_data=pl.DataFrame({"a": [0, 1], "b": [0, 1]}),
            reference_id="reference",
            limits=ComparisonLimits(max_reference_rows=1),
        )
    assert raised.value.code == "GINSU_MAX_COMPARISON_REFERENCE_ROWS"


@pytest.mark.parametrize(
    "kwargs, error_type, message",
    [
        ({"method": "cosine"}, ValueError, "method"),
        ({"similarity_threshold": 0}, ValueError, "similarity_threshold"),
        ({"direction_tolerance": -1}, ValueError, "direction_tolerance"),
        ({"method": "membership"}, ValueError, "reference_data"),
        (
            {"reference_data": pl.DataFrame({"a": [0], "b": [0]})},
            ValueError,
            "reference_id",
        ),
    ],
)
def test_invalid_comparison_configuration_is_rejected(
    kwargs, error_type, message
):
    with pytest.raises(error_type, match=message):
        compare_analyses(_analysis("a"), _analysis("b"), **kwargs)
