"""Aggregate exact-rule stability across caller-controlled discovery runs."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import polars as pl
from sklearn.utils.validation import check_is_fitted

from ginsu._domain import canonical_value, value_from_canonical_json
from ginsu._frame import normalize_frame, validate_schema
from ginsu.diagnostics import AnalysisLimitError

SuccessfulRunStatus = Literal["complete", "no_valid_slices"]
FailedRunStatus = Literal["failed", "limit_reached"]
StabilityRunStatus = SuccessfulRunStatus | FailedRunStatus
StabilitySimilarityMethod = Literal["predicate", "membership"]

RUN_OBSERVATION_SCHEMA = {
    "__ginsu_id": pl.String,
    "__ginsu_rule": pl.String,
    "rank": pl.UInt32,
    "slice_score": pl.Float64,
    "support_fraction": pl.Float64,
    "error_lift": pl.Float64,
}

RUN_PREDICATE_SCHEMA: dict[str, Any] = {
    "__ginsu_id": pl.String,
    "predicate_tokens": pl.List(pl.String),
}

RUN_REFERENCE_MEMBERSHIP_SCHEMA: dict[str, Any] = {
    "__ginsu_id": pl.String,
    "membership": pl.List(pl.Boolean),
}

STABILITY_RUN_SCHEMA = {
    "run_id": pl.String,
    "status": pl.String,
    "partition_id": pl.String,
    "resampling_unit": pl.String,
    "seed": pl.Int64,
    "parameters_json": pl.String,
    "slice_count": pl.UInt64,
    "failure_reason": pl.String,
}

SLICE_RUN_SCHEMA = {
    "__ginsu_id": pl.String,
    "__ginsu_rule": pl.String,
    "run_id": pl.String,
    "run_status": pl.String,
    "selected": pl.Boolean,
    "rank": pl.UInt32,
    "slice_score": pl.Float64,
    "support_fraction": pl.Float64,
    "error_lift": pl.Float64,
}

STABILITY_SUMMARY_SCHEMA = {
    "__ginsu_id": pl.String,
    "__ginsu_rule": pl.String,
    "attempted_run_count": pl.UInt64,
    "successful_run_count": pl.UInt64,
    "unavailable_run_count": pl.UInt64,
    "selected_run_count": pl.UInt64,
    "selection_frequency_successful": pl.Float64,
    "selection_frequency_attempted": pl.Float64,
    "rank_mean": pl.Float64,
    "rank_std": pl.Float64,
    "rank_min": pl.UInt32,
    "rank_max": pl.UInt32,
    "slice_score_mean": pl.Float64,
    "slice_score_std": pl.Float64,
    "support_fraction_mean": pl.Float64,
    "support_fraction_std": pl.Float64,
    "error_lift_mean": pl.Float64,
    "error_lift_std": pl.Float64,
    "stability_status": pl.String,
}

SIMILARITY_MATCH_SCHEMA = {
    "anchor_id": pl.String,
    "anchor_rule": pl.String,
    "run_id": pl.String,
    "run_status": pl.String,
    "evidence_available": pl.Boolean,
    "candidate_id": pl.String,
    "candidate_rule": pl.String,
    "similarity": pl.Float64,
    "matched": pl.Boolean,
    "exact_selected": pl.Boolean,
    "exact_match": pl.Boolean,
    "rank": pl.UInt32,
    "slice_score": pl.Float64,
    "support_fraction": pl.Float64,
    "error_lift": pl.Float64,
}

SIMILARITY_SUMMARY_SCHEMA = {
    "anchor_id": pl.String,
    "anchor_rule": pl.String,
    "similarity_method": pl.String,
    "similarity_threshold": pl.Float64,
    "attempted_run_count": pl.UInt64,
    "successful_run_count": pl.UInt64,
    "unavailable_run_count": pl.UInt64,
    "matched_run_count": pl.UInt64,
    "exact_run_count": pl.UInt64,
    "match_frequency_successful": pl.Float64,
    "match_frequency_attempted": pl.Float64,
    "similarity_mean": pl.Float64,
    "similarity_min": pl.Float64,
    "rank_mean": pl.Float64,
    "rank_std": pl.Float64,
    "slice_score_mean": pl.Float64,
    "support_fraction_mean": pl.Float64,
    "error_lift_mean": pl.Float64,
    "stability_status": pl.String,
}


@dataclass(frozen=True, slots=True)
class StabilityLimits:
    """Bounds checked before stability cross-run tables are materialized."""

    max_runs: int = 1_000
    max_union_slices: int = 100_000
    max_slice_run_cells: int = 5_000_000
    max_similarity_comparisons: int = 5_000_000

    def __post_init__(self) -> None:
        for name in (
            "max_runs",
            "max_union_slices",
            "max_slice_run_cells",
            "max_similarity_comparisons",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer.")


@dataclass(frozen=True, slots=True)
class StabilityReferenceLimits:
    """Bounds checked before common-reference membership is captured."""

    max_rows: int = 1_000_000
    max_membership_cells: int = 10_000_000

    def __post_init__(self) -> None:
        for name in ("max_rows", "max_membership_cells"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer.")


@dataclass(frozen=True, slots=True)
class StabilityRun:
    """One successful or unavailable caller-controlled discovery run."""

    run_id: str
    status: StabilityRunStatus
    partition_id: str
    resampling_unit: str
    seed: int | None
    parameters_json: str
    observations: pl.DataFrame
    failure_reason: str | None = None
    predicates: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame(schema=RUN_PREDICATE_SCHEMA)
    )
    reference_memberships: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame(
            schema=RUN_REFERENCE_MEMBERSHIP_SCHEMA
        )
    )
    reference_id: str | None = None
    reference_row_fingerprint: str | None = None
    reference_row_count: int | None = None

    def __post_init__(self) -> None:
        for name in ("run_id", "partition_id", "resampling_unit"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a nonempty string.")
        if self.status not in (
            "complete",
            "no_valid_slices",
            "failed",
            "limit_reached",
        ):
            raise ValueError("Unsupported stability run status.")
        if self.seed is not None and (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a nonnegative integer or None.")
        if self.observations.schema != RUN_OBSERVATION_SCHEMA:
            raise ValueError(
                "observations does not match the stability schema."
            )
        if self.predicates.schema != RUN_PREDICATE_SCHEMA:
            raise ValueError("predicates does not match the stability schema.")
        if (
            self.reference_memberships.schema
            != RUN_REFERENCE_MEMBERSHIP_SCHEMA
        ):
            raise ValueError(
                "reference_memberships does not match the stability schema."
            )
        _validate_parameters_json(self.parameters_json)
        successful = self.status in ("complete", "no_valid_slices")
        if successful and self.failure_reason is not None:
            raise ValueError("Successful runs must not have a failure reason.")
        if not successful:
            if not self.failure_reason:
                raise ValueError("Unavailable runs require a failure reason.")
            if self.observations.height:
                raise ValueError(
                    "Unavailable runs must not contain observations."
                )
        if self.status == "no_valid_slices" and self.observations.height:
            raise ValueError("no_valid_slices runs must have no observations.")
        if (
            self.observations["__ginsu_id"].n_unique()
            != self.observations.height
        ):
            raise ValueError("A run must contain unique slice IDs.")
        observation_ids = set(self.observations["__ginsu_id"].to_list())
        predicate_ids = self.predicates["__ginsu_id"].to_list()
        if len(set(predicate_ids)) != len(predicate_ids):
            raise ValueError("A run must contain unique predicate-set IDs.")
        if predicate_ids and set(predicate_ids) != observation_ids:
            raise ValueError(
                "Predicate evidence must cover every observed slice exactly."
            )
        for tokens in self.predicates["predicate_tokens"].to_list():
            if not tokens or tokens != sorted(set(tokens)):
                raise ValueError(
                    "Predicate tokens must be nonempty, unique, and sorted."
                )
            for token in tokens:
                _validate_predicate_token(token)
        membership_ids = self.reference_memberships["__ginsu_id"].to_list()
        if len(set(membership_ids)) != len(membership_ids):
            raise ValueError("A run must contain unique membership IDs.")
        reference_metadata = (
            self.reference_id,
            self.reference_row_fingerprint,
            self.reference_row_count,
        )
        has_reference = any(value is not None for value in reference_metadata)
        if has_reference and any(
            value is None for value in reference_metadata
        ):
            raise ValueError("Common-reference metadata must be complete.")
        if has_reference:
            if not isinstance(self.reference_id, str) or not self.reference_id:
                raise ValueError("reference_id must be a nonempty string.")
            if (
                not isinstance(self.reference_row_fingerprint, str)
                or not self.reference_row_fingerprint
            ):
                raise ValueError(
                    "reference_row_fingerprint must be a nonempty string."
                )
            if (
                isinstance(self.reference_row_count, bool)
                or not isinstance(self.reference_row_count, int)
                or self.reference_row_count <= 0
            ):
                raise ValueError("reference_row_count must be positive.")
            if set(membership_ids) != observation_ids:
                raise ValueError(
                    "Reference membership must cover every observed slice."
                )
            for membership in self.reference_memberships[
                "membership"
            ].to_list():
                if len(membership) != self.reference_row_count:
                    raise ValueError(
                        "Reference membership lengths must match row count."
                    )
        elif membership_ids:
            raise ValueError(
                "Reference membership requires complete reference metadata."
            )
        object.__setattr__(self, "observations", self.observations.clone())
        object.__setattr__(self, "predicates", self.predicates.clone())
        object.__setattr__(
            self, "reference_memberships", self.reference_memberships.clone()
        )

    @classmethod
    def from_finder(
        cls,
        finder: Any,
        *,
        run_id: str,
        partition_id: str,
        resampling_unit: str,
        seed: int | None = None,
        parameters: Mapping[str, Any] | None = None,
        reference_frame: Any | None = None,
        reference_id: str | None = None,
        reference_row_id: str | pl.Series | None = None,
        reference_limits: StabilityReferenceLimits | None = None,
    ) -> StabilityRun:
        """Capture canonical metrics from one exhaustive fitted search."""
        check_is_fitted(
            finder,
            ("slices_", "slice_statistics_", "search_report_"),
        )
        status = finder.search_report_.status
        if status not in ("complete", "no_valid_slices"):
            raise ValueError(
                "Only exhaustive fitted searches can be captured."
            )
        observations = (
            finder.slice_statistics_.join(
                finder.slices_.select("__ginsu_id", "__ginsu_rule"),
                on="__ginsu_id",
                how="left",
                validate="1:1",
            )
            .sort("rank")
            .select(*RUN_OBSERVATION_SCHEMA)
        )
        active_parameters = parameters or {
            "alpha": finder.alpha,
            "k": finder.k,
            "max_l": finder.max_l,
            "min_sup": finder.min_sup,
        }
        predicates = _predicate_frame(finder)
        (
            memberships,
            captured_reference_id,
            reference_row_fingerprint,
            reference_row_count,
        ) = _capture_reference(
            finder,
            reference_frame=reference_frame,
            reference_id=reference_id,
            reference_row_id=reference_row_id,
            limits=reference_limits,
        )
        return cls(
            run_id=run_id,
            status=status,
            partition_id=partition_id,
            resampling_unit=resampling_unit,
            seed=seed,
            parameters_json=_parameters_json(active_parameters),
            observations=observations,
            predicates=predicates,
            reference_memberships=memberships,
            reference_id=captured_reference_id,
            reference_row_fingerprint=reference_row_fingerprint,
            reference_row_count=reference_row_count,
        )

    @classmethod
    def unavailable(
        cls,
        *,
        run_id: str,
        status: FailedRunStatus,
        partition_id: str,
        resampling_unit: str,
        failure_reason: str,
        seed: int | None = None,
        parameters: Mapping[str, Any] | None = None,
    ) -> StabilityRun:
        """Record a failed or limit-terminated run without hiding it."""
        if status not in ("failed", "limit_reached"):
            raise ValueError(
                "Unavailable status must be failed or limit_reached."
            )
        return cls(
            run_id=run_id,
            status=status,
            partition_id=partition_id,
            resampling_unit=resampling_unit,
            seed=seed,
            parameters_json=_parameters_json(parameters or {}),
            observations=pl.DataFrame(schema=RUN_OBSERVATION_SCHEMA),
            failure_reason=failure_reason,
        )


@dataclass(frozen=True, slots=True)
class StabilityReport:
    """Run evidence, per-run presence, and exact-rule stability summaries."""

    runs: pl.DataFrame
    slice_runs: pl.DataFrame
    summary: pl.DataFrame
    minimum_successful_runs: int
    stable_frequency: float
    warning_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.runs.schema != STABILITY_RUN_SCHEMA:
            raise ValueError("runs does not match the stability schema.")
        if self.slice_runs.schema != SLICE_RUN_SCHEMA:
            raise ValueError("slice_runs does not match the stability schema.")
        if self.summary.schema != STABILITY_SUMMARY_SCHEMA:
            raise ValueError("summary does not match the stability schema.")
        object.__setattr__(self, "runs", self.runs.clone())
        object.__setattr__(self, "slice_runs", self.slice_runs.clone())
        object.__setattr__(self, "summary", self.summary.clone())


@dataclass(frozen=True, slots=True)
class SimilarityStabilityReport:
    """Anchor-based related-rule recurrence alongside exact stability."""

    exact: StabilityReport
    matches: pl.DataFrame
    summary: pl.DataFrame
    similarity_method: StabilitySimilarityMethod
    similarity_threshold: float
    reference_id: str | None = None
    reference_row_fingerprint: str | None = None
    reference_row_count: int | None = None
    warning_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.exact, StabilityReport):
            raise TypeError("exact must be a StabilityReport.")
        if self.matches.schema != SIMILARITY_MATCH_SCHEMA:
            raise ValueError("matches does not match the stability schema.")
        if self.summary.schema != SIMILARITY_SUMMARY_SCHEMA:
            raise ValueError("summary does not match the stability schema.")
        if self.similarity_method not in ("predicate", "membership"):
            raise ValueError("Unsupported stability similarity method.")
        _validate_similarity_threshold(self.similarity_threshold)
        object.__setattr__(self, "matches", self.matches.clone())
        object.__setattr__(self, "summary", self.summary.clone())


def evaluate_stability(
    runs: Sequence[StabilityRun],
    *,
    minimum_successful_runs: int = 2,
    stable_frequency: float = 0.8,
    limits: StabilityLimits | None = None,
) -> StabilityReport:
    """Aggregate exact canonical slice recurrence across supplied runs."""
    if not isinstance(runs, Sequence):
        raise TypeError("runs must be a sequence of StabilityRun instances.")
    if not runs:
        raise ValueError("runs must contain at least one StabilityRun.")
    if any(not isinstance(run, StabilityRun) for run in runs):
        raise TypeError("runs must contain only StabilityRun instances.")
    if (
        isinstance(minimum_successful_runs, bool)
        or not isinstance(minimum_successful_runs, int)
        or minimum_successful_runs <= 0
    ):
        raise ValueError("minimum_successful_runs must be a positive integer.")
    if (
        isinstance(stable_frequency, bool)
        or not isinstance(stable_frequency, (int, float))
        or not math.isfinite(stable_frequency)
        or not 0 < stable_frequency <= 1
    ):
        raise ValueError("stable_frequency must be finite and in (0, 1].")
    active_limits = limits or StabilityLimits()
    if not isinstance(active_limits, StabilityLimits):
        raise TypeError("limits must be a StabilityLimits instance or None.")
    if len(runs) > active_limits.max_runs:
        raise AnalysisLimitError(
            "GINSU_MAX_STABILITY_RUNS",
            observed=len(runs),
            limit=active_limits.max_runs,
            stage="stability analysis",
        )
    run_ids = [run.run_id for run in runs]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("Stability run IDs must be unique.")

    rules = _union_rules(runs)
    if len(rules) > active_limits.max_union_slices:
        raise AnalysisLimitError(
            "GINSU_MAX_STABILITY_SLICES",
            observed=len(rules),
            limit=active_limits.max_union_slices,
            stage="stability analysis",
        )
    cells = len(runs) * len(rules)
    if cells > active_limits.max_slice_run_cells:
        raise AnalysisLimitError(
            "GINSU_MAX_STABILITY_SLICE_RUN_CELLS",
            observed=cells,
            limit=active_limits.max_slice_run_cells,
            stage="stability analysis",
        )

    run_frame = _run_frame(runs)
    slice_runs = _slice_run_frame(runs, rules)
    successful_count = sum(
        run.status in ("complete", "no_valid_slices") for run in runs
    )
    summary = _summary_frame(
        slice_runs,
        rules=rules,
        attempted_count=len(runs),
        successful_count=successful_count,
        minimum_successful_runs=minimum_successful_runs,
        stable_frequency=float(stable_frequency),
    )
    warning_codes = []
    if successful_count < len(runs):
        warning_codes.append("GINSU_STABILITY_UNAVAILABLE_RUNS")
    if successful_count < minimum_successful_runs:
        warning_codes.append("GINSU_STABILITY_INSUFFICIENT_RUNS")
    return StabilityReport(
        runs=run_frame,
        slice_runs=slice_runs,
        summary=summary,
        minimum_successful_runs=minimum_successful_runs,
        stable_frequency=float(stable_frequency),
        warning_codes=tuple(warning_codes),
    )


def evaluate_similarity_stability(
    runs: Sequence[StabilityRun],
    *,
    method: StabilitySimilarityMethod = "predicate",
    similarity_threshold: float = 0.8,
    minimum_successful_runs: int = 2,
    stable_frequency: float = 0.8,
    limits: StabilityLimits | None = None,
) -> SimilarityStabilityReport:
    """Match each canonical anchor to the best related rule in every run.

    Matching is deliberately anchor-based and does not create transitive fuzzy
    clusters. A run's best candidate is chosen by similarity, then discovery
    rank, then canonical ID. Failed and limit-terminated runs remain unknown.
    """
    if method not in ("predicate", "membership"):
        raise ValueError("method must be 'predicate' or 'membership'.")
    _validate_similarity_threshold(similarity_threshold)
    active_limits = limits or StabilityLimits()
    exact = evaluate_stability(
        runs,
        minimum_successful_runs=minimum_successful_runs,
        stable_frequency=stable_frequency,
        limits=active_limits,
    )
    rules = _union_rules(runs)
    comparisons = len(rules) * sum(
        run.observations.height
        for run in runs
        if run.status in ("complete", "no_valid_slices")
    )
    if comparisons > active_limits.max_similarity_comparisons:
        raise AnalysisLimitError(
            "GINSU_MAX_STABILITY_SIMILARITY_COMPARISONS",
            observed=comparisons,
            limit=active_limits.max_similarity_comparisons,
            stage="stability similarity matching",
        )

    reference_id: str | None = None
    reference_fingerprint: str | None = None
    reference_row_count: int | None = None
    if method == "predicate":
        evidence = _predicate_evidence(runs, rules)
    else:
        (
            evidence,
            reference_id,
            reference_fingerprint,
            reference_row_count,
        ) = _membership_evidence(runs, rules)

    matches = _similarity_matches(
        runs,
        rules=rules,
        evidence=evidence,
        threshold=float(similarity_threshold),
    )
    summary = _similarity_summary(
        matches,
        rules=rules,
        method=method,
        threshold=float(similarity_threshold),
        attempted_count=len(runs),
        successful_count=sum(
            run.status in ("complete", "no_valid_slices") for run in runs
        ),
        minimum_successful_runs=minimum_successful_runs,
        stable_frequency=float(stable_frequency),
    )
    warning_codes = list(exact.warning_codes)
    empty_membership_comparisons = matches.filter(
        pl.col("evidence_available")
        & pl.col("candidate_id").is_not_null()
        & pl.col("similarity").is_null()
    ).height
    if method == "membership" and empty_membership_comparisons:
        warning_codes.append("GINSU_STABILITY_EMPTY_REFERENCE_MEMBERSHIP")
    return SimilarityStabilityReport(
        exact=exact,
        matches=matches,
        summary=summary,
        similarity_method=method,
        similarity_threshold=float(similarity_threshold),
        reference_id=reference_id,
        reference_row_fingerprint=reference_fingerprint,
        reference_row_count=reference_row_count,
        warning_codes=tuple(dict.fromkeys(warning_codes)),
    )


def _predicate_frame(finder: Any) -> pl.DataFrame:
    tokens_by_id: dict[str, list[str]] = {}
    for row in finder.predicates_.iter_rows(named=True):
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
        tokens_by_id.setdefault(row["__ginsu_id"], []).append(token)
    rows = [
        {
            "__ginsu_id": identifier,
            "predicate_tokens": sorted(tokens_by_id[identifier]),
        }
        for identifier in finder.slices_["__ginsu_id"].to_list()
    ]
    return (
        pl.DataFrame(rows, schema=RUN_PREDICATE_SCHEMA)
        if rows
        else pl.DataFrame(schema=RUN_PREDICATE_SCHEMA)
    )


def _validate_predicate_token(token: str) -> None:
    if not isinstance(token, str):
        raise TypeError("Predicate tokens must be strings.")
    try:
        decoded = json.loads(token)
    except json.JSONDecodeError as error:
        raise ValueError(
            "Predicate tokens must contain valid JSON."
        ) from error
    if not isinstance(decoded, dict) or set(decoded) != {
        "feature",
        "operator",
        "value_json",
    }:
        raise ValueError("Predicate tokens have an invalid schema.")
    if not isinstance(decoded["feature"], str) or not decoded["feature"]:
        raise ValueError("Predicate token features must be nonempty strings.")
    if decoded["operator"] != "eq":
        raise ValueError("Predicate token operators must be 'eq'.")
    value_from_canonical_json(decoded["value_json"])
    canonical = json.dumps(
        decoded,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    if token != canonical:
        raise ValueError("Predicate tokens must use canonical JSON.")


def _capture_reference(
    finder: Any,
    *,
    reference_frame: Any | None,
    reference_id: str | None,
    reference_row_id: str | pl.Series | None,
    limits: StabilityReferenceLimits | None,
) -> tuple[pl.DataFrame, str | None, str | None, int | None]:
    empty = pl.DataFrame(schema=RUN_REFERENCE_MEMBERSHIP_SCHEMA)
    if reference_frame is None:
        if reference_id is not None or reference_row_id is not None:
            raise ValueError(
                "reference_id and reference_row_id require reference_frame."
            )
        if limits is not None and not isinstance(
            limits, StabilityReferenceLimits
        ):
            raise TypeError(
                "reference_limits must be a StabilityReferenceLimits "
                "instance or None."
            )
        return empty, None, None, None
    if not isinstance(reference_id, str) or not reference_id:
        raise ValueError(
            "reference_id must be a nonempty string when capturing membership."
        )
    if reference_row_id is None:
        raise ValueError(
            "reference_row_id is required to verify cross-run row alignment."
        )
    active_limits = limits or StabilityReferenceLimits()
    if not isinstance(active_limits, StabilityReferenceLimits):
        raise TypeError(
            "reference_limits must be a StabilityReferenceLimits instance "
            "or None."
        )
    normalized = normalize_frame(reference_frame)
    validate_schema(normalized.frame, expected=finder._feature_schema)
    row_count = normalized.frame.height
    slice_count = finder.slices_.height
    if row_count > active_limits.max_rows:
        raise AnalysisLimitError(
            "GINSU_MAX_STABILITY_REFERENCE_ROWS",
            observed=row_count,
            limit=active_limits.max_rows,
            stage="stability reference capture",
        )
    cells = row_count * slice_count
    if cells > active_limits.max_membership_cells:
        raise AnalysisLimitError(
            "GINSU_MAX_STABILITY_REFERENCE_MEMBERSHIP_CELLS",
            observed=cells,
            limit=active_limits.max_membership_cells,
            stage="stability reference capture",
        )
    row_ids = _reference_row_ids(normalized.frame, reference_row_id)
    fingerprint = _row_identity_fingerprint(row_ids)
    membership = finder.membership_frame(normalized.frame, row_id=row_ids)
    rows = [
        {
            "__ginsu_id": identifier,
            "membership": membership[identifier].to_list(),
        }
        for identifier in finder.slices_["__ginsu_id"].to_list()
    ]
    captured = (
        pl.DataFrame(rows, schema=RUN_REFERENCE_MEMBERSHIP_SCHEMA)
        if rows
        else empty
    )
    return captured, reference_id, fingerprint, row_count


def _reference_row_ids(
    frame: pl.DataFrame, row_id: str | pl.Series
) -> pl.Series:
    if isinstance(row_id, str):
        if row_id not in frame.columns:
            raise ValueError(f"Unknown reference_row_id column {row_id!r}.")
        values = frame[row_id]
    elif isinstance(row_id, pl.Series):
        if row_id.len() != frame.height:
            raise ValueError(
                "reference_row_id length must match the reference frame."
            )
        values = row_id
    else:
        raise TypeError(
            "reference_row_id must be a column name or Polars Series."
        )
    if values.null_count():
        raise ValueError("reference_row_id must not contain null values.")
    if values.n_unique() != frame.height:
        raise ValueError("reference_row_id values must be unique.")
    return values.alias("__ginsu_row")


def _row_identity_fingerprint(values: pl.Series) -> str:
    encoded = {
        "dtype": str(values.dtype),
        "values": [canonical_value(value) for value in values.to_list()],
        "version": 1,
    }
    payload = json.dumps(
        encoded,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return f"ginsu:reference-rows:v1:{hashlib.sha256(payload).hexdigest()}"


SimilarityEvidence = frozenset[str] | np.ndarray


def _predicate_evidence(
    runs: Sequence[StabilityRun], rules: Mapping[str, str]
) -> dict[str, dict[str, SimilarityEvidence]]:
    evidence: dict[str, dict[str, SimilarityEvidence]] = {}
    for run in runs:
        if run.observations.height and not run.predicates.height:
            raise ValueError(
                f"Run {run.run_id!r} has no captured predicate evidence."
            )
        evidence[run.run_id] = {
            row["__ginsu_id"]: frozenset(row["predicate_tokens"])
            for row in run.predicates.iter_rows(named=True)
        }
    _validate_anchor_evidence(evidence, rules)
    return evidence


def _membership_evidence(
    runs: Sequence[StabilityRun], rules: Mapping[str, str]
) -> tuple[
    dict[str, dict[str, SimilarityEvidence]],
    str | None,
    str | None,
    int | None,
]:
    populated = [run for run in runs if run.observations.height]
    if not populated:
        return ({run.run_id: {} for run in runs}, None, None, None)
    for run in populated:
        if not run.reference_memberships.height:
            raise ValueError(
                f"Run {run.run_id!r} has no common-reference membership "
                "evidence."
            )
    successful = [
        run for run in runs if run.status in ("complete", "no_valid_slices")
    ]
    if any(run.reference_id is None for run in successful):
        raise ValueError(
            "Every successful run requires common-reference metadata for "
            "membership similarity."
        )
    reference_keys = {
        (
            run.reference_id,
            run.reference_row_fingerprint,
            run.reference_row_count,
        )
        for run in successful
    }
    if len(reference_keys) != 1:
        raise ValueError(
            "Membership similarity requires one reference_id and identical "
            "ordered reference-row fingerprints across runs."
        )
    reference_id, fingerprint, row_count = next(iter(reference_keys))
    evidence: dict[str, dict[str, SimilarityEvidence]] = {}
    for run in runs:
        evidence[run.run_id] = {
            str(row["__ginsu_id"]): np.asarray(row["membership"], dtype=bool)
            for row in run.reference_memberships.iter_rows(named=True)
        }
    _validate_anchor_evidence(evidence, rules)
    return evidence, reference_id, fingerprint, row_count


def _validate_anchor_evidence(
    evidence: Mapping[str, Mapping[str, SimilarityEvidence]],
    rules: Mapping[str, str],
) -> None:
    anchors: dict[str, SimilarityEvidence] = {}
    for run_evidence in evidence.values():
        for identifier, item in run_evidence.items():
            previous = anchors.setdefault(identifier, item)
            if not _evidence_equal(previous, item):
                raise ValueError(
                    f"Slice ID {identifier!r} has inconsistent similarity "
                    "evidence across runs."
                )
    missing = set(rules).difference(anchors)
    if missing:
        raise ValueError("Similarity evidence does not cover every anchor.")


def _evidence_equal(
    left: SimilarityEvidence, right: SimilarityEvidence
) -> bool:
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return bool(np.array_equal(left, right))
    return left == right


def _anchor_evidence(
    evidence: Mapping[str, Mapping[str, SimilarityEvidence]], identifier: str
) -> SimilarityEvidence:
    return next(
        run_evidence[identifier]
        for run_evidence in evidence.values()
        if identifier in run_evidence
    )


def _jaccard(
    left: SimilarityEvidence, right: SimilarityEvidence
) -> float | None:
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        union = int(np.count_nonzero(left | right))
        if not union:
            return None
        return float(np.count_nonzero(left & right) / union)
    if isinstance(left, frozenset) and isinstance(right, frozenset):
        union_set = left | right
        return len(left & right) / len(union_set) if union_set else None
    raise TypeError("Similarity evidence types do not match.")


def _similarity_matches(
    runs: Sequence[StabilityRun],
    *,
    rules: Mapping[str, str],
    evidence: Mapping[str, Mapping[str, SimilarityEvidence]],
    threshold: float,
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for anchor_id, anchor_rule in rules.items():
        anchor = _anchor_evidence(evidence, anchor_id)
        for run in runs:
            available = run.status in ("complete", "no_valid_slices")
            observations = {
                row["__ginsu_id"]: row
                for row in run.observations.iter_rows(named=True)
            }
            candidates: list[tuple[float | None, dict[str, Any]]] = []
            if available:
                for candidate_id, observation in observations.items():
                    similarity = _jaccard(
                        anchor, evidence[run.run_id][candidate_id]
                    )
                    candidates.append((similarity, observation))
            comparable = [item for item in candidates if item[0] is not None]
            if comparable:
                similarity, candidate = min(
                    comparable,
                    key=lambda item: (
                        -(item[0] if item[0] is not None else -math.inf),
                        item[1]["rank"],
                        item[1]["__ginsu_id"],
                    ),
                )
            elif candidates:
                similarity, candidate = min(
                    candidates,
                    key=lambda item: (
                        item[1]["rank"],
                        item[1]["__ginsu_id"],
                    ),
                )
            else:
                similarity, candidate = None, None
            matched = (
                similarity is not None and similarity >= threshold
                if available
                else None
            )
            rows.append(
                {
                    "anchor_id": anchor_id,
                    "anchor_rule": anchor_rule,
                    "run_id": run.run_id,
                    "run_status": run.status,
                    "evidence_available": available,
                    "candidate_id": (
                        candidate["__ginsu_id"] if candidate else None
                    ),
                    "candidate_rule": (
                        candidate["__ginsu_rule"] if candidate else None
                    ),
                    "similarity": similarity,
                    "matched": matched,
                    "exact_selected": (
                        anchor_id in observations if available else None
                    ),
                    "exact_match": (
                        candidate["__ginsu_id"] == anchor_id
                        if candidate
                        else None
                    ),
                    "rank": candidate["rank"] if candidate else None,
                    "slice_score": (
                        candidate["slice_score"] if candidate else None
                    ),
                    "support_fraction": (
                        candidate["support_fraction"] if candidate else None
                    ),
                    "error_lift": (
                        candidate["error_lift"] if candidate else None
                    ),
                }
            )
    return (
        pl.DataFrame(rows, schema=SIMILARITY_MATCH_SCHEMA)
        if rows
        else pl.DataFrame(schema=SIMILARITY_MATCH_SCHEMA)
    )


def _similarity_summary(
    matches: pl.DataFrame,
    *,
    rules: Mapping[str, str],
    method: StabilitySimilarityMethod,
    threshold: float,
    attempted_count: int,
    successful_count: int,
    minimum_successful_runs: int,
    stable_frequency: float,
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for anchor_id, anchor_rule in rules.items():
        anchor_matches = matches.filter(pl.col("anchor_id") == anchor_id)
        matched = anchor_matches.filter(pl.col("matched"))
        matched_count = matched.height
        frequency = (
            matched_count / successful_count if successful_count else None
        )
        if successful_count < minimum_successful_runs:
            status = "insufficient_successful_runs"
        elif frequency is not None and frequency >= stable_frequency:
            status = "stable"
        else:
            status = "fragile"
        rows.append(
            {
                "anchor_id": anchor_id,
                "anchor_rule": anchor_rule,
                "similarity_method": method,
                "similarity_threshold": threshold,
                "attempted_run_count": attempted_count,
                "successful_run_count": successful_count,
                "unavailable_run_count": attempted_count - successful_count,
                "matched_run_count": matched_count,
                "exact_run_count": anchor_matches.filter(
                    pl.col("exact_selected")
                ).height,
                "match_frequency_successful": frequency,
                "match_frequency_attempted": matched_count / attempted_count,
                "similarity_mean": _mean(matched, "similarity"),
                "similarity_min": _minimum(matched, "similarity"),
                "rank_mean": _mean(matched, "rank"),
                "rank_std": _std(matched, "rank"),
                "slice_score_mean": _mean(matched, "slice_score"),
                "support_fraction_mean": _mean(matched, "support_fraction"),
                "error_lift_mean": _mean(matched, "error_lift"),
                "stability_status": status,
            }
        )
    return (
        pl.DataFrame(rows, schema=SIMILARITY_SUMMARY_SCHEMA).sort(
            "match_frequency_successful",
            "rank_mean",
            "anchor_id",
            descending=(True, False, False),
            nulls_last=True,
        )
        if rows
        else pl.DataFrame(schema=SIMILARITY_SUMMARY_SCHEMA)
    )


def _validate_similarity_threshold(value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 < value <= 1
    ):
        raise ValueError("similarity_threshold must be finite and in (0, 1].")


def _parameters_json(parameters: Mapping[str, Any]) -> str:
    normalized: dict[str, Any] = {}
    for key, value in parameters.items():
        if not isinstance(key, str) or not key:
            raise ValueError("Parameter names must be nonempty strings.")
        if isinstance(value, np.generic):
            value = value.item()
        if value is not None and not isinstance(
            value, (bool, int, float, str)
        ):
            raise TypeError("Stability parameters must be JSON scalar values.")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Stability parameters must be finite.")
        normalized[key] = value
    return json.dumps(
        normalized,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _validate_parameters_json(payload: str) -> None:
    if not isinstance(payload, str):
        raise TypeError("parameters_json must be a string.")
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError("parameters_json must contain valid JSON.") from error
    if not isinstance(decoded, dict) or _parameters_json(decoded) != payload:
        raise ValueError("parameters_json must be a canonical scalar object.")


def _union_rules(runs: Sequence[StabilityRun]) -> dict[str, str]:
    rules: dict[str, str] = {}
    for run in runs:
        for identifier, rule in run.observations.select(
            "__ginsu_id", "__ginsu_rule"
        ).iter_rows():
            previous = rules.setdefault(identifier, rule)
            if previous != rule:
                raise ValueError(
                    f"Slice ID {identifier!r} has inconsistent display rules."
                )
    return dict(sorted(rules.items()))


def _run_frame(runs: Sequence[StabilityRun]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "run_id": run.run_id,
                "status": run.status,
                "partition_id": run.partition_id,
                "resampling_unit": run.resampling_unit,
                "seed": run.seed,
                "parameters_json": run.parameters_json,
                "slice_count": run.observations.height,
                "failure_reason": run.failure_reason,
            }
            for run in runs
        ],
        schema=STABILITY_RUN_SCHEMA,
    )


def _slice_run_frame(
    runs: Sequence[StabilityRun], rules: Mapping[str, str]
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in runs:
        available = run.status in ("complete", "no_valid_slices")
        by_id = {
            row["__ginsu_id"]: row
            for row in run.observations.iter_rows(named=True)
        }
        for identifier, rule in rules.items():
            observation = by_id.get(identifier)
            rows.append(
                {
                    "__ginsu_id": identifier,
                    "__ginsu_rule": rule,
                    "run_id": run.run_id,
                    "run_status": run.status,
                    "selected": (
                        observation is not None if available else None
                    ),
                    "rank": observation["rank"] if observation else None,
                    "slice_score": (
                        observation["slice_score"] if observation else None
                    ),
                    "support_fraction": (
                        observation["support_fraction"]
                        if observation
                        else None
                    ),
                    "error_lift": (
                        observation["error_lift"] if observation else None
                    ),
                }
            )
    return pl.DataFrame(rows, schema=SLICE_RUN_SCHEMA)


def _summary_frame(
    slice_runs: pl.DataFrame,
    *,
    rules: Mapping[str, str],
    attempted_count: int,
    successful_count: int,
    minimum_successful_runs: int,
    stable_frequency: float,
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for identifier, rule in rules.items():
        observed = slice_runs.filter(
            (pl.col("__ginsu_id") == identifier) & pl.col("selected")
        )
        selected_count = observed.height
        frequency_successful = (
            selected_count / successful_count if successful_count else None
        )
        if successful_count < minimum_successful_runs:
            status = "insufficient_successful_runs"
        elif (
            frequency_successful is not None
            and frequency_successful >= stable_frequency
        ):
            status = "stable"
        else:
            status = "fragile"
        rows.append(
            {
                "__ginsu_id": identifier,
                "__ginsu_rule": rule,
                "attempted_run_count": attempted_count,
                "successful_run_count": successful_count,
                "unavailable_run_count": attempted_count - successful_count,
                "selected_run_count": selected_count,
                "selection_frequency_successful": frequency_successful,
                "selection_frequency_attempted": selected_count
                / attempted_count,
                "rank_mean": _mean(observed, "rank"),
                "rank_std": _std(observed, "rank"),
                "rank_min": _integer_min(observed, "rank"),
                "rank_max": _integer_max(observed, "rank"),
                "slice_score_mean": _mean(observed, "slice_score"),
                "slice_score_std": _std(observed, "slice_score"),
                "support_fraction_mean": _mean(observed, "support_fraction"),
                "support_fraction_std": _std(observed, "support_fraction"),
                "error_lift_mean": _mean(observed, "error_lift"),
                "error_lift_std": _std(observed, "error_lift"),
                "stability_status": status,
            }
        )
    return (
        pl.DataFrame(rows, schema=STABILITY_SUMMARY_SCHEMA).sort(
            "selection_frequency_successful",
            "rank_mean",
            "__ginsu_id",
            descending=(True, False, False),
            nulls_last=True,
        )
        if rows
        else pl.DataFrame(schema=STABILITY_SUMMARY_SCHEMA)
    )


def _mean(frame: pl.DataFrame, column: str) -> float | None:
    if not frame.height:
        return None
    values = frame.get_column(column).to_numpy().astype(np.float64)
    return float(values.mean())


def _std(frame: pl.DataFrame, column: str) -> float | None:
    if not frame.height:
        return None
    values = frame.get_column(column).to_numpy().astype(np.float64)
    return float(values.std(ddof=0))


def _minimum(frame: pl.DataFrame, column: str) -> float | None:
    if not frame.height:
        return None
    values = frame.get_column(column).to_numpy().astype(np.float64)
    return float(values.min())


def _integer_min(frame: pl.DataFrame, column: str) -> int | None:
    if not frame.height:
        return None
    return int(frame.get_column(column).to_numpy().min())


def _integer_max(frame: pl.DataFrame, column: str) -> int | None:
    if not frame.height:
        return None
    return int(frame.get_column(column).to_numpy().max())
