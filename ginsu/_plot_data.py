"""Pure-Polars data contracts used by Ginsu visualizations."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import polars as pl
from sklearn.utils.validation import check_is_fitted

from ginsu._frame import normalize_frame, validate_schema
from ginsu._validation import normalize_errors
from ginsu.diagnostics import AnalysisLimitError


def _require_positive(name: str, value: int) -> None:
    if isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be positive; received {value!r}.")


def _check_membership_bounds(
    *,
    row_count: int,
    slice_count: int,
    max_slices: int,
    max_membership_cells: int,
) -> None:
    _require_positive("max_slices", max_slices)
    _require_positive("max_membership_cells", max_membership_cells)
    if slice_count > max_slices:
        raise AnalysisLimitError(
            "GINSU_MAX_ANALYSIS_SLICES",
            observed=slice_count,
            limit=max_slices,
            stage="membership analysis",
        )
    membership_cells = row_count * slice_count
    if membership_cells > max_membership_cells:
        raise AnalysisLimitError(
            "GINSU_MAX_MEMBERSHIP_CELLS",
            observed=membership_cells,
            limit=max_membership_cells,
            stage="membership analysis",
        )


def impact_plot_data(finder: Any) -> pl.DataFrame:
    """Return ranked slice impact values without importing a plot backend."""
    check_is_fitted(finder, ("slices_", "slice_statistics_"))
    return finder.slice_statistics_.join(
        finder.slices_.select("__ginsu_id", "__ginsu_rule"),
        on="__ginsu_id",
        how="left",
        validate="1:1",
    ).select(
        "__ginsu_id",
        "rank",
        "__ginsu_rule",
        "slice_score",
        "support_count",
        "support_fraction",
        "error_mean",
        "baseline_error_mean",
        "error_lift",
        "excess_error",
        "predicate_count",
    )


def predicate_matrix_data(
    finder: Any,
    *,
    max_slices: int = 100,
    max_cells: int = 10_000,
) -> pl.DataFrame:
    """Return one bounded row for every returned-slice/feature cell."""
    _require_positive("max_slices", max_slices)
    _require_positive("max_cells", max_cells)
    check_is_fitted(finder, ("slices_", "predicates_", "slice_statistics_"))
    slice_count = finder.slices_.height
    feature_names = [name for name, _dtype in finder._feature_schema]
    if slice_count > max_slices:
        raise AnalysisLimitError(
            "GINSU_MAX_ANALYSIS_SLICES",
            observed=slice_count,
            limit=max_slices,
            stage="predicate matrix",
        )
    cell_count = slice_count * len(feature_names)
    if cell_count > max_cells:
        raise AnalysisLimitError(
            "GINSU_MAX_PREDICATE_MATRIX_CELLS",
            observed=cell_count,
            limit=max_cells,
            stage="predicate matrix",
        )

    slices = finder.slices_.select(
        "__ginsu_id", "__ginsu_rank", "__ginsu_rule"
    )
    features = pl.DataFrame(
        {
            "feature": pl.Series(feature_names, dtype=pl.String),
            "feature_order": pl.Series(
                range(len(feature_names)), dtype=pl.UInt32
            ),
        }
    )
    cells = slices.join(features, how="cross")
    predicate_values = finder.predicates_.select(
        "__ginsu_id", "feature", "display_value", "value_json"
    )
    statistics = finder.slice_statistics_.select(
        "__ginsu_id", "support_count", "error_lift", "slice_score"
    )
    return (
        cells.join(
            predicate_values,
            on=("__ginsu_id", "feature"),
            how="left",
            validate="1:1",
        )
        .join(statistics, on="__ginsu_id", how="left", validate="m:1")
        .with_columns(pl.col("display_value").is_not_null().alias("is_used"))
        .sort("__ginsu_rank", "feature_order")
        .select(
            "__ginsu_id",
            "__ginsu_rank",
            "__ginsu_rule",
            "feature",
            "feature_order",
            "display_value",
            "value_json",
            "is_used",
            "support_count",
            "error_lift",
            "slice_score",
        )
    )


def equivalence_groups(
    finder: Any,
    X: Any,
    *,
    max_slices: int = 100,
    max_membership_cells: int = 10_000_000,
) -> pl.DataFrame:
    """Return exact groups of rules selecting the same observations."""
    check_is_fitted(finder, ("slices_", "_feature_schema"))
    normalized = normalize_frame(X)
    validate_schema(normalized.frame, expected=finder._feature_schema)
    identifiers = finder.slices_.get_column("__ginsu_id").to_list()
    _check_membership_bounds(
        row_count=normalized.frame.height,
        slice_count=len(identifiers),
        max_slices=max_slices,
        max_membership_cells=max_membership_cells,
    )

    schema: dict[str, Any] = {
        "__ginsu_equivalence_id": pl.String,
        "representative_id": pl.String,
        "represented_rule_count": pl.UInt32,
        "slice_ids": pl.List(pl.String),
        "support_count": pl.UInt64,
    }
    if not identifiers:
        return pl.DataFrame(schema=schema)

    membership = finder.membership_frame(normalized.frame)
    masks = membership.select(identifiers).to_numpy().T.astype(bool)
    groups: list[tuple[list[str], np.ndarray]] = []
    for identifier, mask in zip(identifiers, masks, strict=True):
        for members, representative_mask in groups:
            if np.array_equal(mask, representative_mask):
                members.append(identifier)
                break
        else:
            groups.append(([identifier], mask))

    rows = []
    for members, mask in groups:
        payload = json.dumps(
            sorted(members), ensure_ascii=False, separators=(",", ":")
        ).encode()
        rows.append(
            {
                "__ginsu_equivalence_id": (
                    f"ginsu:eq:v1:{hashlib.sha256(payload).hexdigest()}"
                ),
                "representative_id": members[0],
                "represented_rule_count": len(members),
                "slice_ids": members,
                "support_count": int(mask.sum()),
            }
        )
    return pl.DataFrame(rows, schema=schema)


def overlap_data(
    finder: Any,
    X: Any,
    *,
    metric: str = "jaccard",
    max_slices: int = 100,
    max_cells: int = 10_000,
    max_membership_cells: int = 10_000_000,
) -> pl.DataFrame:
    """Return a full symmetric, bounded pairwise slice-overlap table."""
    if metric != "jaccard":
        raise ValueError("metric must be 'jaccard'.")
    _require_positive("max_cells", max_cells)
    check_is_fitted(finder, ("slices_", "_feature_schema"))
    normalized = normalize_frame(X)
    validate_schema(normalized.frame, expected=finder._feature_schema)
    identifiers = finder.slices_.get_column("__ginsu_id").to_list()
    _check_membership_bounds(
        row_count=normalized.frame.height,
        slice_count=len(identifiers),
        max_slices=max_slices,
        max_membership_cells=max_membership_cells,
    )
    pair_cells = len(identifiers) ** 2
    if pair_cells > max_cells:
        raise AnalysisLimitError(
            "GINSU_MAX_OVERLAP_CELLS",
            observed=pair_cells,
            limit=max_cells,
            stage="pairwise Jaccard overlap",
        )

    schema = {
        "left_id": pl.String,
        "right_id": pl.String,
        "left_rank": pl.UInt32,
        "right_rank": pl.UInt32,
        "intersection_count": pl.UInt64,
        "union_count": pl.UInt64,
        "jaccard": pl.Float64,
        "equivalent": pl.Boolean,
    }
    if not identifiers:
        return pl.DataFrame(schema=schema)

    membership = finder.membership_frame(normalized.frame)
    masks = membership.select(identifiers).to_numpy().astype(bool)
    rows = []
    for left_index, left_id in enumerate(identifiers):
        left = masks[:, left_index]
        for right_index, right_id in enumerate(identifiers):
            right = masks[:, right_index]
            intersection = int(np.count_nonzero(left & right))
            union = int(np.count_nonzero(left | right))
            rows.append(
                {
                    "left_id": left_id,
                    "right_id": right_id,
                    "left_rank": left_index + 1,
                    "right_rank": right_index + 1,
                    "intersection_count": intersection,
                    "union_count": union,
                    "jaccard": intersection / union if union else None,
                    "equivalent": bool(np.array_equal(left, right)),
                }
            )
    return pl.DataFrame(rows, schema=schema)


def lattice_edges_data(finder: Any, *, max_nodes: int = 100) -> pl.DataFrame:
    """Return deterministic exact one-predicate parent-child rule edges."""
    _require_positive("max_nodes", max_nodes)
    check_is_fitted(finder, ("slices_", "_slice_objects"))
    nodes = finder._slice_objects
    if len(nodes) > max_nodes:
        raise AnalysisLimitError(
            "GINSU_MAX_LATTICE_NODES",
            observed=len(nodes),
            limit=max_nodes,
            stage="slice lattice",
        )

    schema = {
        "parent_id": pl.String,
        "child_id": pl.String,
        "parent_rank": pl.UInt32,
        "child_rank": pl.UInt32,
        "parent_predicate_count": pl.UInt32,
        "child_predicate_count": pl.UInt32,
        "added_feature": pl.String,
        "added_operator": pl.String,
        "added_value_json": pl.String,
    }
    rows = []
    for child_index, child in enumerate(nodes):
        for parent_index, parent in enumerate(nodes):
            if len(child.predicates) != len(parent.predicates) + 1:
                continue
            if not parent.is_parent_of(child):
                continue
            added = next(
                predicate
                for predicate in child.predicates
                if predicate not in parent.predicates
            )
            rows.append(
                {
                    "parent_id": parent.id,
                    "child_id": child.id,
                    "parent_rank": parent_index + 1,
                    "child_rank": child_index + 1,
                    "parent_predicate_count": len(parent.predicates),
                    "child_predicate_count": len(child.predicates),
                    "added_feature": added.feature,
                    "added_operator": added.operator,
                    "added_value_json": added.value_json,
                }
            )
    return pl.DataFrame(rows, schema=schema)


def error_dependence_data(
    finder: Any,
    X: Any,
    errors: Any,
    *,
    feature: str,
    slice_id: str | None = None,
) -> pl.DataFrame:
    """Return row-level observed feature/loss/membership data for one slice."""
    check_is_fitted(finder, ("slices_", "slice_statistics_"))
    normalized = normalize_frame(X)
    validate_schema(normalized.frame, expected=finder._feature_schema)
    if feature not in normalized.frame.columns:
        raise ValueError(f"Unknown feature {feature!r}.")
    losses = normalize_errors(
        errors,
        expected_length=normalized.frame.height,
        require_positive=False,
    )
    slice_index, selected_id = _resolve_slice(finder, slice_id)
    membership = finder.transform(normalized.frame).get_column(
        f"slice_{slice_index}"
    )

    return (
        normalized.frame.select(pl.col(feature).alias("feature_value"))
        .with_columns(
            pl.Series("error", losses),
            membership.alias("in_slice"),
            pl.lit(selected_id).alias("__ginsu_id"),
        )
        .with_row_index("__ginsu_row")
    )


def error_dependence_summary(
    data: pl.DataFrame, *, n_bins: int = 10
) -> pl.DataFrame:
    """Summarize observed losses using every eligible row."""
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")
    dtype = data.schema["feature_value"]
    if dtype.is_numeric() and data["feature_value"].n_unique() > n_bins:
        grouped = data.with_columns(
            pl.col("feature_value")
            .qcut(n_bins, allow_duplicates=True)
            .alias("feature_group")
        )
    else:
        grouped = data.with_columns(
            pl.col("feature_value").cast(pl.String).alias("feature_group")
        )

    return (
        grouped.group_by("feature_group", "in_slice")
        .agg(
            pl.len().alias("count"),
            pl.col("feature_value").mean().alias("feature_mean")
            if dtype.is_numeric()
            else pl.lit(None, dtype=pl.Float64).alias("feature_mean"),
            pl.col("error").mean().alias("error_mean"),
            pl.col("error").median().alias("error_median"),
            pl.col("error").quantile(0.25).alias("error_q25"),
            pl.col("error").quantile(0.75).alias("error_q75"),
        )
        .sort("feature_group", "in_slice")
    )


def _resolve_slice(finder: Any, slice_id: str | None) -> tuple[int, str]:
    identifiers = finder.slices_.get_column("__ginsu_id").to_list()
    if not identifiers:
        raise ValueError("Ginsu did not find any slices to visualize.")
    if slice_id is None:
        return 0, identifiers[0]
    try:
        return identifiers.index(slice_id), slice_id
    except ValueError as error:
        raise ValueError(f"Unknown slice_id {slice_id!r}.") from error
