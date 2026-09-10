"""Auditable post-selection views over immutable discovery rankings."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import polars as pl
from sklearn.utils.validation import check_is_fitted

from ginsu._frame import normalize_frame, validate_schema
from ginsu.diagnostics import AnalysisLimitError

SelectionMethod = Literal["score", "unique_membership", "diverse"]

SELECTION_SCHEMA = {
    "__ginsu_id": pl.String,
    "discovery_rank": pl.UInt32,
    "__ginsu_rule": pl.String,
    "slice_score": pl.Float64,
    "reference_support_count": pl.UInt64,
    "reference_support_fraction": pl.Float64,
    "selected": pl.Boolean,
    "selection_rank": pl.UInt32,
    "selection_status": pl.String,
    "representative_id": pl.String,
    "max_selected_jaccard": pl.Float64,
    "candidate_incremental_support_count": pl.UInt64,
    "candidate_incremental_support_fraction": pl.Float64,
    "cumulative_selected_support_count": pl.UInt64,
    "cumulative_selected_support_fraction": pl.Float64,
}


@dataclass(frozen=True, slots=True)
class SelectionLimits:
    """Bounds checked before selection membership and pairwise work."""

    max_slices: int = 1_000
    max_membership_cells: int = 10_000_000
    max_pair_comparisons: int = 500_000

    def __post_init__(self) -> None:
        for name in (
            "max_slices",
            "max_membership_cells",
            "max_pair_comparisons",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer.")


@dataclass(frozen=True, slots=True)
class SliceSelection:
    """Complete audit table for one deterministic post-selection view."""

    decisions: pl.DataFrame
    method: SelectionMethod
    requested_count: int
    row_count: int
    input_kind: str
    max_jaccard: float | None
    warning_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.decisions.schema != SELECTION_SCHEMA:
            raise ValueError("decisions does not match the selection schema.")
        if self.method not in ("score", "unique_membership", "diverse"):
            raise ValueError("Unsupported selection method.")
        if self.requested_count <= 0:
            raise ValueError("requested_count must be positive.")
        if self.row_count <= 0:
            raise ValueError("row_count must be positive.")
        if self.method == "diverse":
            _validate_max_jaccard(self.max_jaccard)
        elif self.max_jaccard is not None:
            raise ValueError(
                "max_jaccard is only valid for diverse selection."
            )
        object.__setattr__(self, "decisions", self.decisions.clone())

    @property
    def selected_count(self) -> int:
        """Number of rules selected by this view."""
        return self.decisions.filter(pl.col("selected")).height

    @property
    def selected_slices(self) -> pl.DataFrame:
        """Return selected decisions in deterministic selection order."""
        return self.decisions.filter(pl.col("selected")).sort("selection_rank")


def select_slices(
    finder: Any,
    X: Any,
    *,
    method: SelectionMethod = "score",
    k: int = 20,
    max_jaccard: float = 0.8,
    limits: SelectionLimits | None = None,
) -> SliceSelection:
    """Create a bounded post-selection view without changing raw discoveries.

    ``score`` selects the first ``k`` discovery-ranked rules.
    ``unique_membership`` greedily keeps the first rule for each exact
    reference-membership mask. ``diverse`` greedily follows discovery rank and
    keeps a rule only when its Jaccard overlap with every already-selected rule
    is at most ``max_jaccard``. Every candidate remains in the decision table.
    """
    check_is_fitted(
        finder,
        ("slices_", "slice_statistics_", "_feature_schema"),
    )
    if method not in ("score", "unique_membership", "diverse"):
        raise ValueError(
            "method must be 'score', 'unique_membership', or 'diverse'."
        )
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")
    if method == "diverse":
        _validate_max_jaccard(max_jaccard)
    active_limits = limits or SelectionLimits()
    if not isinstance(active_limits, SelectionLimits):
        raise TypeError("limits must be a SelectionLimits instance or None.")

    normalized = normalize_frame(X)
    validate_schema(normalized.frame, expected=finder._feature_schema)
    identifiers = finder.slices_.get_column("__ginsu_id").to_list()
    _check_selection_bounds(
        row_count=normalized.frame.height,
        slice_count=len(identifiers),
        method=method,
        limits=active_limits,
    )

    if identifiers:
        membership = finder.membership_frame(normalized.frame)
        masks = membership.select(identifiers).to_numpy().astype(bool)
        decisions = _build_decisions(
            finder,
            masks=masks,
            method=method,
            k=k,
            max_jaccard=max_jaccard,
        )
    else:
        decisions = pl.DataFrame(schema=SELECTION_SCHEMA)

    statuses = set(decisions.get_column("selection_status").to_list())
    warning_codes = []
    if "excluded_no_reference_members" in statuses:
        warning_codes.append("GINSU_SELECTION_NO_REFERENCE_MEMBERS")
    if "excluded_capacity" in statuses:
        warning_codes.append("GINSU_SELECTION_CAPACITY_REACHED")

    return SliceSelection(
        decisions=decisions,
        method=method,
        requested_count=k,
        row_count=normalized.frame.height,
        input_kind=normalized.kind,
        max_jaccard=max_jaccard if method == "diverse" else None,
        warning_codes=tuple(warning_codes),
    )


def _validate_max_jaccard(value: float | None) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError("max_jaccard must be finite and in [0, 1].")


def _check_selection_bounds(
    *,
    row_count: int,
    slice_count: int,
    method: SelectionMethod,
    limits: SelectionLimits,
) -> None:
    if slice_count > limits.max_slices:
        raise AnalysisLimitError(
            "GINSU_MAX_SELECTION_SLICES",
            observed=slice_count,
            limit=limits.max_slices,
            stage="slice post-selection",
        )
    membership_cells = row_count * slice_count
    if membership_cells > limits.max_membership_cells:
        raise AnalysisLimitError(
            "GINSU_MAX_SELECTION_MEMBERSHIP_CELLS",
            observed=membership_cells,
            limit=limits.max_membership_cells,
            stage="slice post-selection",
        )
    if method != "score":
        pair_comparisons = slice_count * (slice_count - 1) // 2
        if pair_comparisons > limits.max_pair_comparisons:
            raise AnalysisLimitError(
                "GINSU_MAX_SELECTION_PAIR_COMPARISONS",
                observed=pair_comparisons,
                limit=limits.max_pair_comparisons,
                stage="slice post-selection",
            )


def _build_decisions(
    finder: Any,
    *,
    masks: np.ndarray,
    method: SelectionMethod,
    k: int,
    max_jaccard: float,
) -> pl.DataFrame:
    discovery = finder.slice_statistics_.join(
        finder.slices_.select("__ginsu_id", "__ginsu_rule"),
        on="__ginsu_id",
        how="left",
        validate="1:1",
    ).sort("rank")
    row_count = masks.shape[0]
    selected_indices: list[int] = []
    covered = np.zeros(row_count, dtype=bool)
    rows: list[dict[str, Any]] = []

    for index, item in enumerate(discovery.iter_rows(named=True)):
        candidate = masks[:, index]
        support = int(np.count_nonzero(candidate))
        incremental = int(np.count_nonzero(candidate & ~covered))
        selected = False
        selection_rank = None
        representative_id = None
        maximum_overlap = None

        if method == "score":
            selected = len(selected_indices) < k
            status = "selected_score" if selected else "excluded_capacity"
        elif support == 0:
            status = "excluded_no_reference_members"
        else:
            blocker_index, maximum_overlap = _closest_selected(
                candidate,
                masks=masks,
                selected_indices=selected_indices,
            )
            if method == "unique_membership" and maximum_overlap == 1.0:
                status = "excluded_equivalent"
                representative_id = discovery.row(blocker_index, named=True)[
                    "__ginsu_id"
                ]
            elif method == "diverse" and (
                maximum_overlap is not None and maximum_overlap > max_jaccard
            ):
                status = "excluded_overlap"
                representative_id = discovery.row(blocker_index, named=True)[
                    "__ginsu_id"
                ]
            elif len(selected_indices) >= k:
                status = "excluded_capacity"
            else:
                selected = True
                status = (
                    "selected_unique_membership"
                    if method == "unique_membership"
                    else "selected_diverse"
                )

        if selected:
            selected_indices.append(index)
            selection_rank = len(selected_indices)
            representative_id = item["__ginsu_id"]
            covered |= candidate

        rows.append(
            {
                "__ginsu_id": item["__ginsu_id"],
                "discovery_rank": item["rank"],
                "__ginsu_rule": item["__ginsu_rule"],
                "slice_score": item["slice_score"],
                "reference_support_count": support,
                "reference_support_fraction": support / row_count,
                "selected": selected,
                "selection_rank": selection_rank,
                "selection_status": status,
                "representative_id": representative_id,
                "max_selected_jaccard": maximum_overlap,
                "candidate_incremental_support_count": incremental,
                "candidate_incremental_support_fraction": (
                    incremental / row_count
                ),
                "cumulative_selected_support_count": int(
                    np.count_nonzero(covered)
                ),
                "cumulative_selected_support_fraction": (
                    np.count_nonzero(covered) / row_count
                ),
            }
        )
    return pl.DataFrame(rows, schema=SELECTION_SCHEMA)


def _closest_selected(
    candidate: np.ndarray,
    *,
    masks: np.ndarray,
    selected_indices: list[int],
) -> tuple[int | None, float | None]:
    blocker_index = None
    maximum_overlap = None
    for selected_index in selected_indices:
        selected = masks[:, selected_index]
        intersection = int(np.count_nonzero(candidate & selected))
        union = int(np.count_nonzero(candidate | selected))
        overlap = intersection / union if union else 0.0
        if maximum_overlap is None or overlap > maximum_overlap:
            blocker_index = selected_index
            maximum_overlap = overlap
    return blocker_index, maximum_overlap
