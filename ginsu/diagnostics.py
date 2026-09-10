"""Read-only search diagnostics and bounded execution policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SearchStatus = Literal[
    "complete",
    "no_valid_slices",
    "limit_reached",
    "cancelled",
    "failed",
]


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
