"""Aggregate exact-rule stability across caller-controlled discovery runs."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import polars as pl
from sklearn.utils.validation import check_is_fitted

from ginsu.diagnostics import AnalysisLimitError

SuccessfulRunStatus = Literal["complete", "no_valid_slices"]
FailedRunStatus = Literal["failed", "limit_reached"]
StabilityRunStatus = SuccessfulRunStatus | FailedRunStatus

RUN_OBSERVATION_SCHEMA = {
    "__ginsu_id": pl.String,
    "__ginsu_rule": pl.String,
    "rank": pl.UInt32,
    "slice_score": pl.Float64,
    "support_fraction": pl.Float64,
    "error_lift": pl.Float64,
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


@dataclass(frozen=True, slots=True)
class StabilityLimits:
    """Bounds checked before stability cross-run tables are materialized."""

    max_runs: int = 1_000
    max_union_slices: int = 100_000
    max_slice_run_cells: int = 5_000_000

    def __post_init__(self) -> None:
        for name in ("max_runs", "max_union_slices", "max_slice_run_cells"):
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
        object.__setattr__(self, "observations", self.observations.clone())

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
        return cls(
            run_id=run_id,
            status=status,
            partition_id=partition_id,
            resampling_unit=resampling_unit,
            seed=seed,
            parameters_json=_parameters_json(active_parameters),
            observations=observations,
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


def _integer_min(frame: pl.DataFrame, column: str) -> int | None:
    if not frame.height:
        return None
    return int(frame.get_column(column).to_numpy().min())


def _integer_max(frame: pl.DataFrame, column: str) -> int | None:
    if not frame.height:
        return None
    return int(frame.get_column(column).to_numpy().max())
