"""Auditable one-to-one comparison of canonical Ginsu analyses."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import polars as pl

from ginsu._domain import value_from_canonical_json
from ginsu._frame import normalize_frame, validate_schema
from ginsu.artifacts import SliceAnalysis, _validate_analysis
from ginsu.diagnostics import AnalysisLimitError

ComparisonMethod = Literal["exact", "predicate", "membership"]

COMPARISON_SCHEMA: dict[str, Any] = {
    "comparison_id": pl.String,
    "match_type": pl.String,
    "similarity": pl.Float64,
    "comparison_status": pl.String,
    "baseline_id": pl.String,
    "baseline_rule": pl.String,
    "baseline_rank": pl.UInt32,
    "candidate_id": pl.String,
    "candidate_rule": pl.String,
    "candidate_rank": pl.UInt32,
    "baseline_slice_score": pl.Float64,
    "candidate_slice_score": pl.Float64,
    "slice_score_delta": pl.Float64,
    "baseline_support_count": pl.UInt64,
    "candidate_support_count": pl.UInt64,
    "support_count_delta": pl.Int64,
    "baseline_support_fraction": pl.Float64,
    "candidate_support_fraction": pl.Float64,
    "support_fraction_delta": pl.Float64,
    "baseline_error_lift": pl.Float64,
    "candidate_error_lift": pl.Float64,
    "error_lift_delta": pl.Float64,
    "baseline_excess_error": pl.Float64,
    "candidate_excess_error": pl.Float64,
    "excess_error_delta": pl.Float64,
    "rank_delta": pl.Int64,
    "baseline_predicate_count": pl.UInt32,
    "candidate_predicate_count": pl.UInt32,
    "shared_predicate_count": pl.UInt32,
    "removed_predicates": pl.List(pl.String),
    "added_predicates": pl.List(pl.String),
    "reference_baseline_only_count": pl.UInt64,
    "reference_candidate_only_count": pl.UInt64,
    "reference_both_count": pl.UInt64,
    "reference_neither_count": pl.UInt64,
    "reference_union_count": pl.UInt64,
    "reference_jaccard": pl.Float64,
}

COMPARISON_SUMMARY_SCHEMA = {
    "comparable": pl.Boolean,
    "compatibility_status": pl.String,
    "compatibility_reason": pl.String,
    "matching_method": pl.String,
    "similarity_threshold": pl.Float64,
    "direction_metric": pl.String,
    "direction_tolerance": pl.Float64,
    "baseline_slice_count": pl.UInt64,
    "candidate_slice_count": pl.UInt64,
    "exact_match_count": pl.UInt64,
    "related_match_count": pl.UInt64,
    "emerged_count": pl.UInt64,
    "resolved_count": pl.UInt64,
    "regressed_count": pl.UInt64,
    "improved_count": pl.UInt64,
    "unchanged_count": pl.UInt64,
    "reference_id": pl.String,
    "reference_row_count": pl.UInt64,
}


@dataclass(frozen=True, slots=True)
class ComparisonLimits:
    """Bounds checked before comparison and reference membership work."""

    max_slices_per_analysis: int = 10_000
    max_pair_comparisons: int = 5_000_000
    max_reference_rows: int = 1_000_000
    max_reference_membership_cells: int = 10_000_000

    def __post_init__(self) -> None:
        for name in (
            "max_slices_per_analysis",
            "max_pair_comparisons",
            "max_reference_rows",
            "max_reference_membership_cells",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer.")


@dataclass(frozen=True, slots=True)
class AnalysisComparison:
    """Comparison compatibility, complete change rows, and count summary."""

    changes: pl.DataFrame
    summary: pl.DataFrame
    matching_method: ComparisonMethod
    similarity_threshold: float
    direction_tolerance: float
    reference_id: str | None = None
    reference_row_count: int | None = None
    warning_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.matching_method not in ("exact", "predicate", "membership"):
            raise ValueError("Unsupported comparison matching method.")
        _validate_similarity_threshold(self.similarity_threshold)
        _validate_direction_tolerance(self.direction_tolerance)
        if self.changes.schema != COMPARISON_SCHEMA:
            raise ValueError("changes does not match the comparison schema.")
        if self.summary.schema != COMPARISON_SUMMARY_SCHEMA:
            raise ValueError("summary does not match the comparison schema.")
        if self.summary.height != 1:
            raise ValueError(
                "comparison summary must contain exactly one row."
            )
        summary = self.summary.row(0, named=True)
        if summary["matching_method"] != self.matching_method:
            raise ValueError("Summary matching method does not match report.")
        if summary["similarity_threshold"] != self.similarity_threshold:
            raise ValueError(
                "Summary similarity threshold does not match report."
            )
        if summary["direction_tolerance"] != self.direction_tolerance:
            raise ValueError(
                "Summary direction tolerance does not match report."
            )
        object.__setattr__(self, "changes", self.changes.clone())
        object.__setattr__(self, "summary", self.summary.clone())

    @property
    def comparable(self) -> bool:
        """Whether the analysis semantics passed compatibility checks."""
        return bool(self.summary["comparable"][0])


def compare_analyses(
    baseline: SliceAnalysis,
    candidate: SliceAnalysis,
    *,
    method: ComparisonMethod = "predicate",
    similarity_threshold: float = 0.8,
    direction_tolerance: float = 0.0,
    reference_data: Any | None = None,
    reference_id: str | None = None,
    limits: ComparisonLimits | None = None,
) -> AnalysisComparison:
    """Compare two exhaustive analyses with deterministic one-to-one matches."""
    if not isinstance(baseline, SliceAnalysis) or not isinstance(
        candidate, SliceAnalysis
    ):
        raise TypeError(
            "baseline and candidate must be SliceAnalysis objects."
        )
    _validate_analysis(baseline)
    _validate_analysis(candidate)
    if method not in ("exact", "predicate", "membership"):
        raise ValueError(
            "method must be 'exact', 'predicate', or 'membership'."
        )
    _validate_similarity_threshold(similarity_threshold)
    _validate_direction_tolerance(direction_tolerance)
    active_limits = limits or ComparisonLimits()
    if not isinstance(active_limits, ComparisonLimits):
        raise TypeError("limits must be a ComparisonLimits instance or None.")
    if reference_data is None and reference_id is not None:
        raise ValueError("reference_id requires reference_data.")
    if reference_data is not None and (
        not isinstance(reference_id, str) or not reference_id
    ):
        raise ValueError("reference_id is required with reference_data.")
    if method == "membership" and reference_data is None:
        raise ValueError("Membership matching requires reference_data.")

    compatibility_status, compatibility_reason = _compatibility(
        baseline, candidate
    )
    if compatibility_status != "compatible":
        return _non_comparable(
            baseline,
            candidate,
            method=method,
            similarity_threshold=float(similarity_threshold),
            direction_tolerance=float(direction_tolerance),
            status=compatibility_status,
            reason=compatibility_reason,
            reference_id=reference_id,
        )

    baseline_count = baseline.slices.height
    candidate_count = candidate.slices.height
    _check_comparison_limits(
        baseline_count=baseline_count,
        candidate_count=candidate_count,
        method=method,
        limits=active_limits,
    )
    baseline_masks: dict[str, np.ndarray] | None = None
    candidate_masks: dict[str, np.ndarray] | None = None
    reference_row_count: int | None = None
    if reference_data is not None:
        normalized = normalize_frame(reference_data)
        if normalized.frame.height > active_limits.max_reference_rows:
            raise AnalysisLimitError(
                "GINSU_MAX_COMPARISON_REFERENCE_ROWS",
                observed=normalized.frame.height,
                limit=active_limits.max_reference_rows,
                stage="analysis comparison",
            )
        membership_cells = normalized.frame.height * (
            baseline_count + candidate_count
        )
        if membership_cells > active_limits.max_reference_membership_cells:
            raise AnalysisLimitError(
                "GINSU_MAX_COMPARISON_REFERENCE_MEMBERSHIP_CELLS",
                observed=membership_cells,
                limit=active_limits.max_reference_membership_cells,
                stage="analysis comparison",
            )
        baseline_frame = _reference_frame(baseline, normalized.frame)
        candidate_frame = _reference_frame(candidate, normalized.frame)
        baseline_masks = _analysis_masks(baseline, baseline_frame)
        candidate_masks = _analysis_masks(candidate, candidate_frame)
        reference_row_count = normalized.frame.height

    pairs = _match_pairs(
        baseline,
        candidate,
        method=method,
        threshold=float(similarity_threshold),
        baseline_masks=baseline_masks,
        candidate_masks=candidate_masks,
    )
    changes = _change_frame(
        baseline,
        candidate,
        pairs=pairs,
        direction_tolerance=float(direction_tolerance),
        baseline_masks=baseline_masks,
        candidate_masks=candidate_masks,
        reference_row_count=reference_row_count,
    )
    summary = _summary_frame(
        changes,
        comparable=True,
        compatibility_status="compatible",
        compatibility_reason=None,
        method=method,
        similarity_threshold=float(similarity_threshold),
        direction_tolerance=float(direction_tolerance),
        baseline_count=baseline_count,
        candidate_count=candidate_count,
        reference_id=reference_id,
        reference_row_count=reference_row_count,
    )
    warning_codes = []
    if reference_data is None:
        warning_codes.append("GINSU_COMPARISON_NO_REFERENCE_MEMBERSHIP")
    elif changes["reference_jaccard"].null_count():
        warning_codes.append("GINSU_COMPARISON_EMPTY_REFERENCE_UNION")
    return AnalysisComparison(
        changes=changes,
        summary=summary,
        matching_method=method,
        similarity_threshold=float(similarity_threshold),
        direction_tolerance=float(direction_tolerance),
        reference_id=reference_id,
        reference_row_count=reference_row_count,
        warning_codes=tuple(warning_codes),
    )


def _compatibility(
    baseline: SliceAnalysis, candidate: SliceAnalysis
) -> tuple[str, str | None]:
    if baseline.feature_schema != candidate.feature_schema:
        return (
            "incompatible_feature_schema",
            "Ordered feature names or dtypes differ.",
        )
    if (baseline.discretization is None) != (candidate.discretization is None):
        return (
            "incompatible_discretization",
            "Only one analysis records a discretization plan.",
        )
    if (
        baseline.discretization is not None
        and candidate.discretization is not None
        and baseline.discretization.to_spec()
        != candidate.discretization.to_spec()
    ):
        return (
            "incompatible_discretization",
            "Discretization specifications differ.",
        )
    return "compatible", None


def _non_comparable(
    baseline: SliceAnalysis,
    candidate: SliceAnalysis,
    *,
    method: ComparisonMethod,
    similarity_threshold: float,
    direction_tolerance: float,
    status: str,
    reason: str | None,
    reference_id: str | None,
) -> AnalysisComparison:
    changes = pl.DataFrame(schema=COMPARISON_SCHEMA)
    summary = _summary_frame(
        changes,
        comparable=False,
        compatibility_status=status,
        compatibility_reason=reason,
        method=method,
        similarity_threshold=similarity_threshold,
        direction_tolerance=direction_tolerance,
        baseline_count=baseline.slices.height,
        candidate_count=candidate.slices.height,
        reference_id=reference_id,
        reference_row_count=None,
    )
    return AnalysisComparison(
        changes=changes,
        summary=summary,
        matching_method=method,
        similarity_threshold=similarity_threshold,
        direction_tolerance=direction_tolerance,
        reference_id=reference_id,
        warning_codes=("GINSU_COMPARISON_NOT_COMPARABLE",),
    )


def _check_comparison_limits(
    *,
    baseline_count: int,
    candidate_count: int,
    method: ComparisonMethod,
    limits: ComparisonLimits,
) -> None:
    observed_max = max(baseline_count, candidate_count)
    if observed_max > limits.max_slices_per_analysis:
        raise AnalysisLimitError(
            "GINSU_MAX_COMPARISON_SLICES",
            observed=observed_max,
            limit=limits.max_slices_per_analysis,
            stage="analysis comparison",
        )
    comparisons = baseline_count * candidate_count if method != "exact" else 0
    if comparisons > limits.max_pair_comparisons:
        raise AnalysisLimitError(
            "GINSU_MAX_COMPARISON_PAIRS",
            observed=comparisons,
            limit=limits.max_pair_comparisons,
            stage="analysis comparison",
        )


def _reference_frame(
    analysis: SliceAnalysis, raw_reference: pl.DataFrame
) -> pl.DataFrame:
    if analysis.discretization is not None:
        return analysis.discretization.transform(raw_reference)
    validate_schema(raw_reference, expected=analysis.feature_schema)
    return raw_reference


def _analysis_masks(
    analysis: SliceAnalysis, frame: pl.DataFrame
) -> dict[str, np.ndarray]:
    predicates: dict[str, list[dict[str, Any]]] = {
        identifier: []
        for identifier in analysis.slices["__ginsu_id"].to_list()
    }
    for row in analysis.predicates.iter_rows(named=True):
        predicates[row["__ginsu_id"]].append(row)
    masks: dict[str, np.ndarray] = {}
    for identifier, rows in predicates.items():
        expressions = [
            pl.col(row["feature"])
            == value_from_canonical_json(row["value_json"])
            for row in rows
        ]
        mask = frame.select(pl.all_horizontal(expressions).alias("member"))[
            "member"
        ]
        masks[identifier] = mask.to_numpy().astype(bool)
    return masks


def _predicate_sets(analysis: SliceAnalysis) -> dict[str, frozenset[str]]:
    tokens: dict[str, set[str]] = {
        identifier: set()
        for identifier in analysis.slices["__ginsu_id"].to_list()
    }
    for row in analysis.predicates.iter_rows(named=True):
        token = json.dumps(
            {
                "feature": row["feature"],
                "operator": row["operator"],
                "value_json": row["value_json"],
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        tokens[row["__ginsu_id"]].add(token)
    return {
        identifier: frozenset(items) for identifier, items in tokens.items()
    }


def _match_pairs(
    baseline: SliceAnalysis,
    candidate: SliceAnalysis,
    *,
    method: ComparisonMethod,
    threshold: float,
    baseline_masks: dict[str, np.ndarray] | None,
    candidate_masks: dict[str, np.ndarray] | None,
) -> list[tuple[str | None, str | None, str, float | None]]:
    baseline_ids = baseline.slices["__ginsu_id"].to_list()
    candidate_ids = candidate.slices["__ginsu_id"].to_list()
    candidate_set = set(candidate_ids)
    exact_ids = [
        identifier
        for identifier in baseline_ids
        if identifier in candidate_set
    ]
    pairs: list[tuple[str | None, str | None, str, float | None]] = [
        (identifier, identifier, "exact", 1.0) for identifier in exact_ids
    ]
    used_baseline = set(exact_ids)
    used_candidate = set(exact_ids)
    if method != "exact":
        baseline_evidence: dict[str, Any]
        candidate_evidence: dict[str, Any]
        if method == "predicate":
            baseline_evidence = _predicate_sets(baseline)
            candidate_evidence = _predicate_sets(candidate)
        else:
            if baseline_masks is None or candidate_masks is None:
                raise RuntimeError("Membership evidence is unavailable.")
            baseline_evidence = baseline_masks
            candidate_evidence = candidate_masks
        ranks_b = _rank_lookup(baseline)
        ranks_c = _rank_lookup(candidate)
        candidates = []
        for baseline_id in baseline_ids:
            if baseline_id in used_baseline:
                continue
            for candidate_id in candidate_ids:
                if candidate_id in used_candidate:
                    continue
                similarity = _jaccard(
                    baseline_evidence[baseline_id],
                    candidate_evidence[candidate_id],
                )
                if similarity is not None and similarity >= threshold:
                    candidates.append(
                        (
                            -similarity,
                            ranks_b[baseline_id],
                            ranks_c[candidate_id],
                            baseline_id,
                            candidate_id,
                            similarity,
                        )
                    )
        for _, _, _, baseline_id, candidate_id, similarity in sorted(
            candidates
        ):
            if baseline_id in used_baseline or candidate_id in used_candidate:
                continue
            pairs.append((baseline_id, candidate_id, method, similarity))
            used_baseline.add(baseline_id)
            used_candidate.add(candidate_id)
    pairs.extend(
        (identifier, None, "unmatched", None)
        for identifier in baseline_ids
        if identifier not in used_baseline
    )
    pairs.extend(
        (None, identifier, "unmatched", None)
        for identifier in candidate_ids
        if identifier not in used_candidate
    )
    return pairs


def _change_frame(
    baseline: SliceAnalysis,
    candidate: SliceAnalysis,
    *,
    pairs: list[tuple[str | None, str | None, str, float | None]],
    direction_tolerance: float,
    baseline_masks: dict[str, np.ndarray] | None,
    candidate_masks: dict[str, np.ndarray] | None,
    reference_row_count: int | None,
) -> pl.DataFrame:
    baseline_rows = _analysis_rows(baseline)
    candidate_rows = _analysis_rows(candidate)
    baseline_predicates = _predicate_sets(baseline)
    candidate_predicates = _predicate_sets(candidate)
    rows = []
    for baseline_id, candidate_id, match_type, similarity in pairs:
        baseline_row = (
            baseline_rows.get(baseline_id) if baseline_id is not None else None
        )
        candidate_row = (
            candidate_rows.get(candidate_id)
            if candidate_id is not None
            else None
        )
        baseline_tokens = (
            baseline_predicates[baseline_id]
            if baseline_id is not None
            else frozenset()
        )
        candidate_tokens = (
            candidate_predicates[candidate_id]
            if candidate_id is not None
            else frozenset()
        )
        removed = sorted(baseline_tokens - candidate_tokens)
        added = sorted(candidate_tokens - baseline_tokens)
        shared = len(baseline_tokens & candidate_tokens)
        membership = _membership_change(
            baseline_masks.get(baseline_id)
            if baseline_masks and baseline_id
            else None,
            candidate_masks.get(candidate_id)
            if candidate_masks and candidate_id
            else None,
            reference_row_count=reference_row_count,
        )
        error_lift_delta = _delta(baseline_row, candidate_row, "error_lift")
        status = _comparison_status(
            baseline_id,
            candidate_id,
            error_lift_delta=error_lift_delta,
            tolerance=direction_tolerance,
        )
        row = {
            "comparison_id": _comparison_id(
                baseline_id, candidate_id, match_type
            ),
            "match_type": match_type,
            "similarity": similarity,
            "comparison_status": status,
            "baseline_id": baseline_id,
            "baseline_rule": baseline_row["__ginsu_rule"]
            if baseline_row
            else None,
            "baseline_rank": baseline_row["rank"] if baseline_row else None,
            "candidate_id": candidate_id,
            "candidate_rule": candidate_row["__ginsu_rule"]
            if candidate_row
            else None,
            "candidate_rank": candidate_row["rank"] if candidate_row else None,
            "baseline_slice_score": _value(baseline_row, "slice_score"),
            "candidate_slice_score": _value(candidate_row, "slice_score"),
            "slice_score_delta": _delta(
                baseline_row, candidate_row, "slice_score"
            ),
            "baseline_support_count": _value(baseline_row, "support_count"),
            "candidate_support_count": _value(candidate_row, "support_count"),
            "support_count_delta": _integer_delta(
                baseline_row, candidate_row, "support_count"
            ),
            "baseline_support_fraction": _value(
                baseline_row, "support_fraction"
            ),
            "candidate_support_fraction": _value(
                candidate_row, "support_fraction"
            ),
            "support_fraction_delta": _delta(
                baseline_row, candidate_row, "support_fraction"
            ),
            "baseline_error_lift": _value(baseline_row, "error_lift"),
            "candidate_error_lift": _value(candidate_row, "error_lift"),
            "error_lift_delta": error_lift_delta,
            "baseline_excess_error": _value(baseline_row, "excess_error"),
            "candidate_excess_error": _value(candidate_row, "excess_error"),
            "excess_error_delta": _delta(
                baseline_row, candidate_row, "excess_error"
            ),
            "rank_delta": _integer_delta(baseline_row, candidate_row, "rank"),
            "baseline_predicate_count": _value(
                baseline_row, "predicate_count"
            ),
            "candidate_predicate_count": _value(
                candidate_row, "predicate_count"
            ),
            "shared_predicate_count": shared,
            "removed_predicates": removed,
            "added_predicates": added,
            **membership,
        }
        rows.append(row)
    return (
        pl.DataFrame(rows, schema=COMPARISON_SCHEMA)
        if rows
        else pl.DataFrame(schema=COMPARISON_SCHEMA)
    )


def _analysis_rows(analysis: SliceAnalysis) -> dict[str, dict[str, Any]]:
    return {
        row["__ginsu_id"]: row
        for row in analysis.slices.select("__ginsu_id", "__ginsu_rule")
        .join(
            analysis.slice_statistics,
            on="__ginsu_id",
            how="left",
            validate="1:1",
        )
        .iter_rows(named=True)
    }


def _membership_change(
    baseline: np.ndarray | None,
    candidate: np.ndarray | None,
    *,
    reference_row_count: int | None,
) -> dict[str, Any]:
    names = (
        "reference_baseline_only_count",
        "reference_candidate_only_count",
        "reference_both_count",
        "reference_neither_count",
        "reference_union_count",
        "reference_jaccard",
    )
    if reference_row_count is None:
        return dict.fromkeys(names)
    false_mask = np.zeros(reference_row_count, dtype=bool)
    left = baseline if baseline is not None else false_mask
    right = candidate if candidate is not None else false_mask
    both = int(np.count_nonzero(left & right))
    baseline_only = int(np.count_nonzero(left & ~right))
    candidate_only = int(np.count_nonzero(~left & right))
    union = both + baseline_only + candidate_only
    return {
        "reference_baseline_only_count": baseline_only,
        "reference_candidate_only_count": candidate_only,
        "reference_both_count": both,
        "reference_neither_count": reference_row_count - union,
        "reference_union_count": union,
        "reference_jaccard": both / union if union else None,
    }


def _summary_frame(
    changes: pl.DataFrame,
    *,
    comparable: bool,
    compatibility_status: str,
    compatibility_reason: str | None,
    method: ComparisonMethod,
    similarity_threshold: float,
    direction_tolerance: float,
    baseline_count: int,
    candidate_count: int,
    reference_id: str | None,
    reference_row_count: int | None,
) -> pl.DataFrame:
    def count(column: str, value: str) -> int:
        return changes.filter(pl.col(column) == value).height

    return pl.DataFrame(
        [
            {
                "comparable": comparable,
                "compatibility_status": compatibility_status,
                "compatibility_reason": compatibility_reason,
                "matching_method": method,
                "similarity_threshold": similarity_threshold,
                "direction_metric": "error_lift",
                "direction_tolerance": direction_tolerance,
                "baseline_slice_count": baseline_count,
                "candidate_slice_count": candidate_count,
                "exact_match_count": count("match_type", "exact"),
                "related_match_count": count("match_type", method)
                if method != "exact"
                else 0,
                "emerged_count": count("comparison_status", "emerged"),
                "resolved_count": count("comparison_status", "resolved"),
                "regressed_count": count("comparison_status", "regressed"),
                "improved_count": count("comparison_status", "improved"),
                "unchanged_count": count("comparison_status", "unchanged"),
                "reference_id": reference_id,
                "reference_row_count": reference_row_count,
            }
        ],
        schema=COMPARISON_SUMMARY_SCHEMA,
    )


def _comparison_status(
    baseline_id: str | None,
    candidate_id: str | None,
    *,
    error_lift_delta: float | None,
    tolerance: float,
) -> str:
    if baseline_id is None:
        return "emerged"
    if candidate_id is None:
        return "resolved"
    if error_lift_delta is None:
        raise RuntimeError("Matched comparisons require an error-lift delta.")
    if error_lift_delta > tolerance:
        return "regressed"
    if error_lift_delta < -tolerance:
        return "improved"
    return "unchanged"


def _rank_lookup(analysis: SliceAnalysis) -> dict[str, int]:
    return dict(
        analysis.slice_statistics.select("__ginsu_id", "rank").iter_rows()
    )


def _jaccard(left: Any, right: Any) -> float | None:
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        union = int(np.count_nonzero(left | right))
        return float(np.count_nonzero(left & right) / union) if union else None
    union = left | right
    return len(left & right) / len(union) if union else None


def _value(row: dict[str, Any] | None, column: str) -> Any:
    return row[column] if row is not None else None


def _delta(
    baseline: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
    column: str,
) -> float | None:
    if baseline is None or candidate is None:
        return None
    return float(candidate[column] - baseline[column])


def _integer_delta(
    baseline: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
    column: str,
) -> int | None:
    if baseline is None or candidate is None:
        return None
    return int(candidate[column] - baseline[column])


def _comparison_id(
    baseline_id: str | None, candidate_id: str | None, match_type: str
) -> str:
    payload = json.dumps(
        {
            "baseline_id": baseline_id,
            "candidate_id": candidate_id,
            "match_type": match_type,
            "version": 1,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return f"ginsu:comparison:v1:{hashlib.sha256(payload).hexdigest()}"


def _validate_similarity_threshold(value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
        or value > 1
    ):
        raise ValueError("similarity_threshold must be finite and in (0, 1].")


def _validate_direction_tolerance(value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError("direction_tolerance must be finite and nonnegative.")
