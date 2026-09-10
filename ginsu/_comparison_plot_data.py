"""Bounded Polars plot-data contracts for analysis comparisons."""

from __future__ import annotations

from typing import Literal

import polars as pl

from ginsu.comparison import AnalysisComparison
from ginsu.diagnostics import AnalysisLimitError

ComparisonPlotMetric = Literal[
    "rank",
    "slice_score",
    "support_count",
    "support_fraction",
    "error_lift",
    "excess_error",
]

COMPARISON_DUMBBELL_SCHEMA = pl.Schema(
    {
        "comparison_id": pl.String,
        "label": pl.String,
        "match_type": pl.String,
        "comparison_status": pl.String,
        "similarity": pl.Float64,
        "baseline_id": pl.String,
        "candidate_id": pl.String,
        "metric": pl.String,
        "baseline_value": pl.Float64,
        "candidate_value": pl.Float64,
        "delta": pl.Float64,
    }
)

COMPARISON_MIGRATION_SCHEMA = pl.Schema(
    {
        "comparison_id": pl.String,
        "label": pl.String,
        "match_type": pl.String,
        "comparison_status": pl.String,
        "similarity": pl.Float64,
        "baseline_id": pl.String,
        "candidate_id": pl.String,
        "membership_segment": pl.String,
        "membership_count": pl.UInt64,
        "reference_id": pl.String,
        "reference_row_count": pl.UInt64,
        "reference_neither_count": pl.UInt64,
        "reference_union_count": pl.UInt64,
        "reference_jaccard": pl.Float64,
    }
)

_METRIC_COLUMNS: dict[ComparisonPlotMetric, tuple[str, str, str]] = {
    "rank": ("baseline_rank", "candidate_rank", "rank_delta"),
    "slice_score": (
        "baseline_slice_score",
        "candidate_slice_score",
        "slice_score_delta",
    ),
    "support_count": (
        "baseline_support_count",
        "candidate_support_count",
        "support_count_delta",
    ),
    "support_fraction": (
        "baseline_support_fraction",
        "candidate_support_fraction",
        "support_fraction_delta",
    ),
    "error_lift": (
        "baseline_error_lift",
        "candidate_error_lift",
        "error_lift_delta",
    ),
    "excess_error": (
        "baseline_excess_error",
        "candidate_excess_error",
        "excess_error_delta",
    ),
}


def comparison_dumbbell_data(
    comparison: AnalysisComparison,
    *,
    metric: ComparisonPlotMetric = "error_lift",
    max_changes: int = 100,
) -> pl.DataFrame:
    """Return one bounded row per matched, emerged, or resolved rule."""
    _validate_comparison(comparison)
    _validate_max_changes(max_changes)
    if metric not in _METRIC_COLUMNS:
        choices = ", ".join(_METRIC_COLUMNS)
        raise ValueError(f"metric must be one of: {choices}.")
    _check_change_limit(comparison, max_changes=max_changes)
    if not comparison.comparable:
        return pl.DataFrame(schema=COMPARISON_DUMBBELL_SCHEMA)

    baseline_column, candidate_column, delta_column = _METRIC_COLUMNS[metric]
    return comparison.changes.select(
        "comparison_id",
        _comparison_label(),
        "match_type",
        "comparison_status",
        "similarity",
        "baseline_id",
        "candidate_id",
        pl.lit(metric).alias("metric"),
        pl.col(baseline_column).cast(pl.Float64).alias("baseline_value"),
        pl.col(candidate_column).cast(pl.Float64).alias("candidate_value"),
        pl.col(delta_column).cast(pl.Float64).alias("delta"),
    ).cast(COMPARISON_DUMBBELL_SCHEMA)


def comparison_migration_data(
    comparison: AnalysisComparison,
    *,
    max_changes: int = 100,
) -> pl.DataFrame:
    """Return bounded long-form common-reference membership migration."""
    _validate_comparison(comparison)
    _validate_max_changes(max_changes)
    _check_change_limit(comparison, max_changes=max_changes)
    if not comparison.comparable:
        return pl.DataFrame(schema=COMPARISON_MIGRATION_SCHEMA)
    if (
        comparison.reference_id is None
        or comparison.reference_row_count is None
    ):
        raise ValueError(
            "Membership migration requires comparison reference evidence."
        )

    common = [
        pl.col("comparison_id"),
        _comparison_label(),
        pl.col("match_type"),
        pl.col("comparison_status"),
        pl.col("similarity"),
        pl.col("baseline_id"),
        pl.col("candidate_id"),
    ]
    details = [
        pl.lit(comparison.reference_id).alias("reference_id"),
        pl.lit(comparison.reference_row_count, dtype=pl.UInt64).alias(
            "reference_row_count"
        ),
        pl.col("reference_neither_count"),
        pl.col("reference_union_count"),
        pl.col("reference_jaccard"),
    ]
    segments = (
        ("baseline_only", "reference_baseline_only_count"),
        ("both", "reference_both_count"),
        ("candidate_only", "reference_candidate_only_count"),
    )
    frames = [
        comparison.changes.select(
            *common,
            pl.lit(segment).alias("membership_segment"),
            pl.col(column).alias("membership_count"),
            *details,
        )
        for segment, column in segments
    ]
    return pl.concat(frames).cast(COMPARISON_MIGRATION_SCHEMA)


def _comparison_label() -> pl.Expr:
    baseline = pl.col("baseline_rule")
    candidate = pl.col("candidate_rule")
    return (
        pl.when(baseline.is_null())
        .then(pl.concat_str(candidate, pl.lit(" (emerged)")))
        .when(candidate.is_null())
        .then(pl.concat_str(baseline, pl.lit(" (resolved)")))
        .when(baseline == candidate)
        .then(baseline)
        .otherwise(pl.concat_str(baseline, pl.lit(" → "), candidate))
        .alias("label")
    )


def _validate_comparison(comparison: AnalysisComparison) -> None:
    if not isinstance(comparison, AnalysisComparison):
        raise TypeError("comparison must be an AnalysisComparison.")


def _validate_max_changes(max_changes: int) -> None:
    if isinstance(max_changes, bool) or not isinstance(max_changes, int):
        raise ValueError("max_changes must be a positive integer.")
    if max_changes <= 0:
        raise ValueError("max_changes must be a positive integer.")


def _check_change_limit(
    comparison: AnalysisComparison, *, max_changes: int
) -> None:
    if comparison.changes.height > max_changes:
        raise AnalysisLimitError(
            "GINSU_MAX_COMPARISON_PLOT_CHANGES",
            observed=comparison.changes.height,
            limit=max_changes,
            stage="comparison plot data",
        )
