"""Bounded Polars plot-data contracts for search diagnostics."""

from __future__ import annotations

import math
from typing import Any

import polars as pl

from ginsu.diagnostics import (
    AnalysisLimitError,
    SearchLevelReport,
    SearchLimits,
    SearchReport,
)

SEARCH_FUNNEL_SCHEMA = pl.Schema(
    {
        "report_status": pl.String,
        "exhaustive": pl.Boolean,
        "level": pl.UInt32,
        "source_slices": pl.UInt64,
        "stage": pl.String,
        "stage_order": pl.UInt8,
        "count": pl.UInt64,
        "is_last_completed_level": pl.Boolean,
        "termination_reason": pl.String,
        "per_level_candidate_limit": pl.UInt64,
    }
)

SEARCH_CARDINALITY_SCHEMA = pl.Schema(
    {
        "feature": pl.String,
        "dtype": pl.String,
        "cardinality": pl.UInt64,
        "max_feature_cardinality": pl.UInt64,
        "over_limit": pl.Boolean,
    }
)

_SEARCH_SUMMARY_DTYPES: dict[str, Any] = {
    "status": pl.String,
    "exhaustive": pl.Boolean,
    "backend": pl.String,
    "numba_used": pl.Boolean,
    "input_kind": pl.String,
    "row_count": pl.UInt64,
    "feature_count": pl.UInt64,
    "encoded_feature_count": pl.UInt64,
    "completed_level_count": pl.UInt32,
    "last_completed_level": pl.UInt32,
    "elapsed_seconds": pl.Float64,
    "copy_boundaries": pl.List(pl.String),
    "warning_codes": pl.List(pl.String),
    "termination_reason": pl.String,
    "max_feature_cardinality": pl.UInt64,
    "max_encoded_features": pl.UInt64,
    "max_pair_matrix_bytes": pl.UInt64,
    "max_candidates_per_level": pl.UInt64,
    "max_total_candidates": pl.UInt64,
    "max_search_seconds": pl.Float64,
    "max_tied_slices": pl.UInt64,
}
SEARCH_SUMMARY_SCHEMA = pl.Schema(_SEARCH_SUMMARY_DTYPES)

_FUNNEL_STAGES = (
    ("source_slices", "Source slices"),
    ("potential_pairs", "Potential pairs"),
    ("compatible_pairs", "Compatible pairs"),
    ("candidates_after_pruning", "After pruning"),
    ("evaluated_candidates", "Evaluated candidates"),
    ("valid_candidates", "Valid candidates"),
)


def search_funnel_data(
    report: SearchReport,
    *,
    max_levels: int = 100,
    max_cells: int = 10_000,
) -> pl.DataFrame:
    """Return one bounded row per completed lattice level and funnel stage."""
    _validate_report(report)
    _validate_positive_integer("max_levels", max_levels)
    _validate_positive_integer("max_cells", max_cells)
    level_count = len(report.levels)
    if level_count > max_levels:
        raise AnalysisLimitError(
            "GINSU_MAX_SEARCH_PLOT_LEVELS",
            observed=level_count,
            limit=max_levels,
            stage="search profile plot data",
        )
    cell_count = level_count * len(_FUNNEL_STAGES)
    if cell_count > max_cells:
        raise AnalysisLimitError(
            "GINSU_MAX_SEARCH_PLOT_CELLS",
            observed=cell_count,
            limit=max_cells,
            stage="search profile plot data",
        )

    rows = []
    last_level = report.levels[-1].level if report.levels else None
    candidate_limit = report.limits.max_candidates_per_level
    for level in report.levels:
        for order, (attribute, label) in enumerate(_FUNNEL_STAGES):
            count = getattr(level, attribute)
            if level.level == 1 and attribute in (
                "potential_pairs",
                "compatible_pairs",
            ):
                count = None
            rows.append(
                {
                    "report_status": report.status,
                    "exhaustive": report.exhaustive,
                    "level": level.level,
                    "source_slices": level.source_slices,
                    "stage": label,
                    "stage_order": order,
                    "count": count,
                    "is_last_completed_level": level.level == last_level,
                    "termination_reason": report.termination_reason,
                    "per_level_candidate_limit": candidate_limit,
                }
            )
    return pl.DataFrame(rows, schema=SEARCH_FUNNEL_SCHEMA)


def search_cardinality_data(
    report: SearchReport,
    *,
    max_features: int = 1_000,
) -> pl.DataFrame:
    """Return bounded ordered source-feature cardinality evidence."""
    _validate_report(report)
    _validate_positive_integer("max_features", max_features)
    feature_count = len(report.feature_cardinalities)
    if feature_count > max_features:
        raise AnalysisLimitError(
            "GINSU_MAX_SEARCH_PLOT_FEATURES",
            observed=feature_count,
            limit=max_features,
            stage="search profile plot data",
        )
    dtype_by_feature = dict(report.input_schema)
    limit = report.limits.max_feature_cardinality
    rows = [
        {
            "feature": feature,
            "dtype": dtype_by_feature[feature],
            "cardinality": cardinality,
            "max_feature_cardinality": limit,
            "over_limit": limit is not None and cardinality > limit,
        }
        for feature, cardinality in report.feature_cardinalities
    ]
    return pl.DataFrame(rows, schema=SEARCH_CARDINALITY_SCHEMA)


def search_summary_data(report: SearchReport) -> pl.DataFrame:
    """Return a one-row Polars summary of recorded execution evidence."""
    _validate_report(report)
    limits = report.limits
    last_level = report.levels[-1].level if report.levels else None
    row = {
        "status": report.status,
        "exhaustive": report.exhaustive,
        "backend": report.backend,
        "numba_used": report.numba_used,
        "input_kind": report.input_kind,
        "row_count": report.row_count,
        "feature_count": report.feature_count,
        "encoded_feature_count": report.encoded_feature_count,
        "completed_level_count": len(report.levels),
        "last_completed_level": last_level,
        "elapsed_seconds": report.elapsed_seconds,
        "copy_boundaries": list(report.copy_boundaries),
        "warning_codes": list(report.warning_codes),
        "termination_reason": report.termination_reason,
        "max_feature_cardinality": limits.max_feature_cardinality,
        "max_encoded_features": limits.max_encoded_features,
        "max_pair_matrix_bytes": limits.max_pair_matrix_bytes,
        "max_candidates_per_level": limits.max_candidates_per_level,
        "max_total_candidates": limits.max_total_candidates,
        "max_search_seconds": limits.max_search_seconds,
        "max_tied_slices": limits.max_tied_slices,
    }
    return pl.DataFrame([row], schema=SEARCH_SUMMARY_SCHEMA)


def _validate_report(report: SearchReport) -> None:
    if not isinstance(report, SearchReport):
        raise TypeError("report must be a SearchReport.")
    if not isinstance(report.limits, SearchLimits):
        raise TypeError("report.limits must be SearchLimits.")
    if report.status not in (
        "complete",
        "no_valid_slices",
        "limit_reached",
        "cancelled",
        "failed",
    ):
        raise ValueError("report has an unsupported status.")
    if report.backend not in ("numba", "numpy"):
        raise ValueError("report has an unsupported backend.")
    if (
        report.row_count < 0
        or report.feature_count < 0
        or report.encoded_feature_count is not None
        and report.encoded_feature_count < 0
    ):
        raise ValueError("report counts must be nonnegative.")
    if not math.isfinite(report.elapsed_seconds) or report.elapsed_seconds < 0:
        raise ValueError(
            "report elapsed_seconds must be finite and nonnegative."
        )
    schema_names = tuple(name for name, _dtype in report.input_schema)
    cardinality_names = tuple(
        name for name, _count in report.feature_cardinalities
    )
    if len(schema_names) != report.feature_count:
        raise ValueError("report feature_count does not match input_schema.")
    if cardinality_names != schema_names:
        raise ValueError(
            "report feature_cardinalities do not match input_schema order."
        )
    if len(set(schema_names)) != len(schema_names):
        raise ValueError("report feature names must be unique.")
    if any(
        cardinality < 0 for _name, cardinality in report.feature_cardinalities
    ):
        raise ValueError("report cardinalities must be nonnegative.")
    previous_level = 0
    for level in report.levels:
        _validate_level(level, previous_level=previous_level)
        previous_level = level.level


def _validate_level(level: SearchLevelReport, *, previous_level: int) -> None:
    if not isinstance(level, SearchLevelReport):
        raise TypeError("report levels must contain SearchLevelReport values.")
    values = (
        level.level,
        level.source_slices,
        level.potential_pairs,
        level.compatible_pairs,
        level.candidates_after_pruning,
        level.evaluated_candidates,
        level.valid_candidates,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in values
    ):
        raise ValueError("search level counts must be integers.")
    if level.level <= previous_level or any(value < 0 for value in values[1:]):
        raise ValueError(
            "search levels must increase and contain nonnegative counts."
        )
    if level.level > 1 and level.compatible_pairs > level.potential_pairs:
        raise ValueError("compatible pair count exceeds potential pairs.")
    if level.level > 1 and (
        level.candidates_after_pruning > level.compatible_pairs
    ):
        raise ValueError("post-pruning count exceeds compatible pairs.")
    if level.evaluated_candidates > level.candidates_after_pruning:
        raise ValueError(
            "evaluated candidate count exceeds post-pruning count."
        )
    if level.valid_candidates > level.evaluated_candidates:
        raise ValueError("valid candidate count exceeds evaluated candidates.")


def _validate_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
