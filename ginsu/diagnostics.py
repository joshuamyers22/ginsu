"""Read-only search diagnostics and bounded execution policies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

SearchStatus = Literal[
    "complete",
    "no_valid_slices",
    "limit_reached",
    "cancelled",
    "failed",
]
SearchStageStatus = Literal["complete", "terminated"]


def _require_positive(name: str, value: int | float | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or value <= 0:
        raise ValueError(
            f"{name} must be positive or None; received {value!r}."
        )


@dataclass(frozen=True, slots=True)
class SearchLimits:
    """Conservative limits applied to one slice search.

    Set a limit to ``None`` only when the caller has independently bounded the
    input and accepts the corresponding resource risk.
    """

    max_feature_cardinality: int | None = 10_000
    max_encoded_features: int | None = 50_000
    max_pair_matrix_bytes: int | None = 256 * 1024 * 1024
    max_candidates_per_level: int | None = 1_000_000
    max_total_candidates: int | None = 5_000_000
    max_search_seconds: float | None = 300.0
    max_tied_slices: int | None = 10_000

    def __post_init__(self) -> None:
        for name in (
            "max_feature_cardinality",
            "max_encoded_features",
            "max_pair_matrix_bytes",
            "max_candidates_per_level",
            "max_total_candidates",
            "max_search_seconds",
            "max_tied_slices",
        ):
            _require_positive(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class SearchLevelReport:
    """Candidate funnel for one lattice level."""

    level: int
    source_slices: int
    potential_pairs: int
    compatible_pairs: int
    candidates_after_pruning: int
    evaluated_candidates: int
    valid_candidates: int


@dataclass(frozen=True, slots=True)
class SearchStageReport:
    """Timing and optional boundary-memory evidence for one search stage."""

    stage: str
    status: SearchStageStatus
    elapsed_seconds: float
    memory_start_bytes: int | None = None
    memory_end_bytes: int | None = None
    observed_peak_memory_bytes: int | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.stage, str)
            or not self.stage.strip()
            or len(self.stage) > 200
        ):
            raise ValueError(
                "stage must be a non-empty string of at most 200 characters."
            )
        if self.status not in ("complete", "terminated"):
            raise ValueError("status must be 'complete' or 'terminated'.")
        if (
            isinstance(self.elapsed_seconds, bool)
            or not isinstance(self.elapsed_seconds, (int, float))
            or not math.isfinite(self.elapsed_seconds)
            or self.elapsed_seconds < 0
        ):
            raise ValueError("elapsed_seconds must be finite and nonnegative.")
        memory_values = (
            self.memory_start_bytes,
            self.memory_end_bytes,
            self.observed_peak_memory_bytes,
        )
        if any(
            value is not None
            and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            )
            for value in memory_values
        ):
            raise ValueError(
                "memory observations must be nonnegative integers."
            )
        observed = [
            value
            for value in (
                self.memory_start_bytes,
                self.memory_end_bytes,
            )
            if value is not None
        ]
        expected_peak = max(observed) if observed else None
        if self.observed_peak_memory_bytes != expected_peak:
            raise ValueError(
                "observed_peak_memory_bytes must equal the largest boundary "
                "observation."
            )


@dataclass(frozen=True, slots=True)
class SearchReport:
    """Immutable evidence describing one completed or terminated fit."""

    status: SearchStatus
    backend: Literal["numba", "numpy"]
    numba_used: bool
    input_kind: str
    input_schema: tuple[tuple[str, str], ...]
    row_count: int
    feature_count: int
    feature_cardinalities: tuple[tuple[str, int], ...]
    encoded_feature_count: int | None
    copy_boundaries: tuple[str, ...]
    levels: tuple[SearchLevelReport, ...]
    elapsed_seconds: float
    limits: SearchLimits
    warning_codes: tuple[str, ...] = ()
    termination_reason: str | None = None
    stages: tuple[SearchStageReport, ...] = ()
    memory_measurement: str | None = None
    observed_peak_memory_bytes: int | None = None

    def __post_init__(self) -> None:
        if any(
            not isinstance(stage, SearchStageReport) for stage in self.stages
        ):
            raise TypeError(
                "stages must contain only SearchStageReport values."
            )
        stage_names = [stage.stage for stage in self.stages]
        if len(set(stage_names)) != len(stage_names):
            raise ValueError("search stage names must be unique.")
        terminated = [
            index
            for index, stage in enumerate(self.stages)
            if stage.status == "terminated"
        ]
        if terminated and terminated != [len(self.stages) - 1]:
            raise ValueError("only the final search stage may be terminated.")
        if self.exhaustive and terminated:
            raise ValueError(
                "an exhaustive report cannot contain a terminated stage."
            )
        if self.stages and not math.isclose(
            math.fsum(stage.elapsed_seconds for stage in self.stages),
            self.elapsed_seconds,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "stage elapsed time must equal report elapsed_seconds."
            )
        for previous, current in zip(
            self.stages, self.stages[1:], strict=False
        ):
            if previous.memory_end_bytes != current.memory_start_bytes:
                raise ValueError(
                    "adjacent search stages must share their boundary-memory "
                    "observation."
                )
        if self.memory_measurement is not None and (
            not isinstance(self.memory_measurement, str)
            or not self.memory_measurement.strip()
            or len(self.memory_measurement) > 200
        ):
            raise ValueError(
                "memory_measurement must be a non-empty string of at most 200 "
                "characters."
            )
        if self.observed_peak_memory_bytes is not None and (
            isinstance(self.observed_peak_memory_bytes, bool)
            or not isinstance(self.observed_peak_memory_bytes, int)
            or self.observed_peak_memory_bytes < 0
        ):
            raise ValueError(
                "observed_peak_memory_bytes must be a nonnegative integer."
            )
        stage_peaks = [
            stage.observed_peak_memory_bytes
            for stage in self.stages
            if stage.observed_peak_memory_bytes is not None
        ]
        expected_peak = max(stage_peaks) if stage_peaks else None
        if self.memory_measurement is None:
            if self.observed_peak_memory_bytes is not None or stage_peaks:
                raise ValueError(
                    "memory observations require memory_measurement."
                )
        elif self.observed_peak_memory_bytes != expected_peak:
            raise ValueError(
                "observed_peak_memory_bytes must equal the largest stage "
                "observation."
            )

    @property
    def exhaustive(self) -> bool:
        """Whether the configured search completed without truncation."""
        return self.status in ("complete", "no_valid_slices")


class ResourceLimitError(RuntimeError):
    """Base error raised before a bounded operation exceeds its limit."""

    def __init__(
        self,
        code: str,
        *,
        observed: int | float,
        limit: int | float,
        stage: str,
    ) -> None:
        self.code = code
        self.observed = observed
        self.limit = limit
        self.stage = stage
        super().__init__(
            f"{code}: {stage} requires {observed!r}, exceeding limit {limit!r}."
        )


class SearchLimitError(ResourceLimitError):
    """Raised before a configured search resource limit is exceeded."""


class AnalysisLimitError(ResourceLimitError):
    """Raised before a derived analysis exceeds its explicit size limit."""
