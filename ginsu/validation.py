"""Fixed-rule descriptive and inferential validation on untouched data."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import polars as pl
from sklearn.utils.validation import check_is_fitted

from ginsu._frame import normalize_frame, validate_schema
from ginsu._validation import normalize_errors
from ginsu.diagnostics import AnalysisLimitError

VALIDATION_STATISTICS_SCHEMA = {
    "__ginsu_id": pl.String,
    "discovery_rank": pl.UInt32,
    "__ginsu_rule": pl.String,
    "discovery_slice_score": pl.Float64,
    "discovery_support_count": pl.UInt64,
    "discovery_support_fraction": pl.Float64,
    "discovery_error_mean": pl.Float64,
    "discovery_error_lift": pl.Float64,
    "validation_status": pl.String,
    "validation_min_support_count": pl.UInt64,
    "validation_support_count": pl.UInt64,
    "validation_support_fraction": pl.Float64,
    "validation_error_sum": pl.Float64,
    "validation_error_max": pl.Float64,
    "validation_error_mean": pl.Float64,
    "validation_complement_error_mean": pl.Float64,
    "validation_mean_difference": pl.Float64,
    "validation_baseline_error_mean": pl.Float64,
    "validation_error_lift": pl.Float64,
    "validation_excess_error": pl.Float64,
    "error_mean_delta": pl.Float64,
    "error_lift_delta": pl.Float64,
    "predicate_count": pl.UInt32,
    "inference_status": pl.String,
    "validation_error_mean_ci_lower": pl.Float64,
    "validation_error_mean_ci_upper": pl.Float64,
    "validation_permutation_p_value": pl.Float64,
    "validation_adjusted_p_value": pl.Float64,
    "validation_significant": pl.Boolean,
}


@dataclass(frozen=True, slots=True)
class ValidationLimits:
    """Resource limits applied before validation membership is materialized."""

    max_slices: int = 1_000
    max_membership_cells: int = 10_000_000
    max_resample_work: int = 100_000_000
    max_resample_batch_cells: int = 1_000_000

    def __post_init__(self) -> None:
        for name in (
            "max_slices",
            "max_membership_cells",
            "max_resample_work",
            "max_resample_batch_cells",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer.")


@dataclass(frozen=True, slots=True)
class ValidationInference:
    """Deterministic uncertainty and fixed-rule hypothesis-test settings.

    The one-sided permutation null is that validation losses are exchangeable
    with respect to each fixed slice membership. Holm correction covers the
    fixed rules that are testable on this validation partition.
    """

    confidence_level: float = 0.95
    bootstrap_resamples: int = 1_000
    permutation_resamples: int = 1_000
    random_seed: int = 0
    multiplicity_method: Literal["holm"] = "holm"

    def __post_init__(self) -> None:
        if (
            isinstance(self.confidence_level, bool)
            or not isinstance(self.confidence_level, (int, float))
            or not math.isfinite(self.confidence_level)
            or not 0 < self.confidence_level < 1
        ):
            raise ValueError("confidence_level must be finite and in (0, 1).")
        for name in ("bootstrap_resamples", "permutation_resamples"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 100
            ):
                raise ValueError(f"{name} must be an integer of at least 100.")
        if (
            isinstance(self.random_seed, bool)
            or not isinstance(self.random_seed, int)
            or self.random_seed < 0
        ):
            raise ValueError("random_seed must be a nonnegative integer.")
        if self.multiplicity_method != "holm":
            raise ValueError("multiplicity_method must be 'holm'.")


@dataclass(frozen=True, slots=True)
class SliceValidation:
    """Holdout evidence for an unchanged set of discovered rules.

    The table remains in discovery-rank order. ``evidence_kind`` explicitly
    distinguishes descriptive metrics from requested fixed-rule inference.
    """

    statistics: pl.DataFrame
    row_count: int
    baseline_error_mean: float
    minimum_support_count: int
    input_kind: str
    warning_codes: tuple[str, ...] = ()
    evidence_kind: str = "holdout_descriptive"
    interval_method: str = "none"
    test_method: str = "none"
    multiplicity_method: str = "none"
    confidence_level: float | None = None
    bootstrap_resamples: int = 0
    permutation_resamples: int = 0
    random_seed: int | None = None

    def __post_init__(self) -> None:
        if self.statistics.schema != VALIDATION_STATISTICS_SCHEMA:
            raise ValueError(
                "statistics does not match the validation schema."
            )
        if self.row_count <= 0:
            raise ValueError("row_count must be positive.")
        if self.minimum_support_count <= 0:
            raise ValueError("minimum_support_count must be positive.")
        if not math.isfinite(self.baseline_error_mean):
            raise ValueError("baseline_error_mean must be finite.")
        if self.baseline_error_mean < 0:
            raise ValueError("baseline_error_mean must be nonnegative.")
        if self.evidence_kind not in (
            "holdout_descriptive",
            "holdout_inferential",
        ):
            raise ValueError("Unsupported validation evidence kind.")
        if self.evidence_kind == "holdout_descriptive" and any(
            (
                self.interval_method != "none",
                self.test_method != "none",
                self.multiplicity_method != "none",
                self.confidence_level is not None,
                self.bootstrap_resamples != 0,
                self.permutation_resamples != 0,
                self.random_seed is not None,
            )
        ):
            raise ValueError(
                "Descriptive validation must not claim inferential methods."
            )
        if self.evidence_kind == "holdout_inferential":
            if self.interval_method != "bootstrap_percentile":
                raise ValueError("Unsupported validation interval method.")
            if self.test_method != "permutation_mean_difference_greater":
                raise ValueError("Unsupported validation test method.")
            if self.multiplicity_method != "holm":
                raise ValueError("Unsupported validation multiplicity method.")
            if self.confidence_level is None:
                raise ValueError(
                    "Inferential validation requires a confidence level."
                )
        object.__setattr__(self, "statistics", self.statistics.clone())

    @property
    def slice_count(self) -> int:
        """Number of fixed discovery rules evaluated."""
        return self.statistics.height

    @property
    def sufficient_support_count(self) -> int:
        """Number of rules meeting the declared validation support threshold."""
        return self.statistics.filter(
            pl.col("validation_status") == "sufficient_support"
        ).height


def validate_slices(
    finder: Any,
    X: Any,
    errors: Any,
    *,
    min_support: int | float | None = None,
    limits: ValidationLimits | None = None,
    inference: ValidationInference | None = None,
) -> SliceValidation:
    """Evaluate fixed discovered slices on a separate labeled partition.

    This function never searches, reranks, filters, or modifies discovery
    results. It computes descriptive loss metrics for every discovered rule in
    original rank order. Optional inference adds percentile bootstrap intervals
    for the conditional slice mean and one-sided permutation tests of the slice
    versus complement mean, corrected across testable fixed rules with Holm's
    method.

    Parameters
    ----------
    finder:
        A fitted :class:`ginsu.Slicefinder`.
    X:
        Schema-compatible validation observations.
    errors:
        One finite, nonnegative realized loss per validation observation. An
        all-zero validation vector is valid and produces null lift values.
    min_support:
        Minimum validation support. An integer is a row count; a float in
        ``(0, 1]`` is rounded upward against the validation row count. ``None``
        reuses the discovery estimator's configured threshold under the same
        rule, with a minimum of one row.
    limits:
        Bounds checked before the row-by-slice membership table is built.
    inference:
        ``None`` returns descriptive evidence. ``ValidationInference()`` adds
        deterministic fixed-rule inference under its documented assumptions.

    Returns
    -------
    SliceValidation
        Self-describing descriptive holdout evidence in discovery-rank order.
    """
    check_is_fitted(
        finder,
        ("slices_", "slice_statistics_", "_feature_schema"),
    )
    normalized = normalize_frame(X)
    validate_schema(normalized.frame, expected=finder._feature_schema)
    losses = normalize_errors(
        errors,
        expected_length=normalized.frame.height,
        require_positive=False,
    )
    minimum_support_count = _resolve_min_support(
        finder.min_sup if min_support is None else min_support,
        row_count=normalized.frame.height,
    )
    active_limits = limits or ValidationLimits()
    if not isinstance(active_limits, ValidationLimits):
        raise TypeError("limits must be a ValidationLimits instance or None.")
    if inference is not None and not isinstance(
        inference, ValidationInference
    ):
        raise TypeError(
            "inference must be a ValidationInference instance or None."
        )

    identifiers = finder.slices_.get_column("__ginsu_id").to_list()
    _check_validation_bounds(
        row_count=normalized.frame.height,
        slice_count=len(identifiers),
        limits=active_limits,
        inference=inference,
    )

    baseline = float(losses.mean())
    if not identifiers:
        statistics = pl.DataFrame(schema=VALIDATION_STATISTICS_SCHEMA)
    else:
        membership = finder.membership_frame(normalized.frame)
        masks = membership.select(identifiers).to_numpy().astype(bool)
        statistics = _build_statistics(
            finder,
            masks=masks,
            losses=losses,
            baseline=baseline,
            minimum_support_count=minimum_support_count,
            inference=inference,
            limits=active_limits,
        )

    statuses = set(statistics.get_column("validation_status").to_list())
    warning_codes = []
    if "no_members" in statuses:
        warning_codes.append("GINSU_VALIDATION_NO_MEMBERS")
    if "insufficient_support" in statuses:
        warning_codes.append("GINSU_VALIDATION_INSUFFICIENT_SUPPORT")
    if baseline == 0:
        warning_codes.append("GINSU_VALIDATION_ZERO_BASELINE")
    if (
        inference is not None
        and statistics.filter(pl.col("inference_status") != "tested").height
    ):
        warning_codes.append("GINSU_VALIDATION_INFERENCE_NOT_TESTED")

    inferential = inference is not None

    return SliceValidation(
        statistics=statistics,
        row_count=normalized.frame.height,
        baseline_error_mean=baseline,
        minimum_support_count=minimum_support_count,
        input_kind=normalized.kind,
        warning_codes=tuple(warning_codes),
        evidence_kind=(
            "holdout_inferential" if inferential else "holdout_descriptive"
        ),
        interval_method=("bootstrap_percentile" if inferential else "none"),
        test_method=(
            "permutation_mean_difference_greater" if inferential else "none"
        ),
        multiplicity_method=("holm" if inferential else "none"),
        confidence_level=(inference.confidence_level if inference else None),
        bootstrap_resamples=(
            inference.bootstrap_resamples if inference else 0
        ),
        permutation_resamples=(
            inference.permutation_resamples if inference else 0
        ),
        random_seed=(inference.random_seed if inference else None),
    )


def _resolve_min_support(value: int | float, *, row_count: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            "min_support must be an integer count or float fraction."
        )
    if isinstance(value, int):
        if value < 0:
            raise ValueError("Integer min_support must be nonnegative.")
        return max(1, value)
    if not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError("Float min_support must be finite and in (0, 1].")
    return max(1, math.ceil(value * row_count))


def _check_validation_bounds(
    *,
    row_count: int,
    slice_count: int,
    limits: ValidationLimits,
    inference: ValidationInference | None,
) -> None:
    if slice_count > limits.max_slices:
        raise AnalysisLimitError(
            "GINSU_MAX_VALIDATION_SLICES",
            observed=slice_count,
            limit=limits.max_slices,
            stage="holdout validation",
        )
    cell_count = row_count * slice_count
    if cell_count > limits.max_membership_cells:
        raise AnalysisLimitError(
            "GINSU_MAX_VALIDATION_MEMBERSHIP_CELLS",
            observed=cell_count,
            limit=limits.max_membership_cells,
            stage="holdout validation",
        )
    if inference is not None:
        if row_count > limits.max_resample_batch_cells:
            raise AnalysisLimitError(
                "GINSU_MAX_VALIDATION_RESAMPLE_BATCH_CELLS",
                observed=row_count,
                limit=limits.max_resample_batch_cells,
                stage="holdout validation inference",
            )
        resample_work = (
            row_count
            * slice_count
            * (inference.bootstrap_resamples + inference.permutation_resamples)
        )
        if resample_work > limits.max_resample_work:
            raise AnalysisLimitError(
                "GINSU_MAX_VALIDATION_RESAMPLE_WORK",
                observed=resample_work,
                limit=limits.max_resample_work,
                stage="holdout validation inference",
            )


def _build_statistics(
    finder: Any,
    *,
    masks: np.ndarray,
    losses: np.ndarray,
    baseline: float,
    minimum_support_count: int,
    inference: ValidationInference | None,
    limits: ValidationLimits,
) -> pl.DataFrame:
    discovery = finder.slice_statistics_.join(
        finder.slices_.select("__ginsu_id", "__ginsu_rule"),
        on="__ginsu_id",
        how="left",
        validate="1:1",
    ).sort("rank")

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(discovery.iter_rows(named=True)):
        mask = masks[:, index]
        support = int(np.count_nonzero(mask))
        selected_losses = losses[mask]
        error_sum = float(selected_losses.sum())
        error_mean = float(selected_losses.mean()) if support else None
        error_max = float(selected_losses.max()) if support else None
        complement = losses[~mask]
        complement_mean = float(complement.mean()) if complement.size else None
        mean_difference = (
            error_mean - complement_mean
            if error_mean is not None and complement_mean is not None
            else None
        )
        error_lift = (
            error_mean / baseline
            if error_mean is not None and baseline > 0
            else None
        )
        if support >= minimum_support_count:
            status = "sufficient_support"
        elif support:
            status = "insufficient_support"
        else:
            status = "no_members"
        rows.append(
            {
                "__ginsu_id": item["__ginsu_id"],
                "discovery_rank": item["rank"],
                "__ginsu_rule": item["__ginsu_rule"],
                "discovery_slice_score": item["slice_score"],
                "discovery_support_count": item["support_count"],
                "discovery_support_fraction": item["support_fraction"],
                "discovery_error_mean": item["error_mean"],
                "discovery_error_lift": item["error_lift"],
                "validation_status": status,
                "validation_min_support_count": minimum_support_count,
                "validation_support_count": support,
                "validation_support_fraction": support / losses.size,
                "validation_error_sum": error_sum,
                "validation_error_max": error_max,
                "validation_error_mean": error_mean,
                "validation_complement_error_mean": complement_mean,
                "validation_mean_difference": mean_difference,
                "validation_baseline_error_mean": baseline,
                "validation_error_lift": error_lift,
                "validation_excess_error": error_sum - support * baseline,
                "error_mean_delta": (
                    error_mean - item["error_mean"]
                    if error_mean is not None
                    else None
                ),
                "error_lift_delta": (
                    error_lift - item["error_lift"]
                    if error_lift is not None
                    else None
                ),
                "predicate_count": item["predicate_count"],
                "inference_status": "not_requested",
                "validation_error_mean_ci_lower": None,
                "validation_error_mean_ci_upper": None,
                "validation_permutation_p_value": None,
                "validation_adjusted_p_value": None,
                "validation_significant": None,
            }
        )
    if inference is not None:
        _add_inference(
            rows,
            masks=masks,
            losses=losses,
            minimum_support_count=minimum_support_count,
            config=inference,
            max_batch_cells=limits.max_resample_batch_cells,
        )
    return pl.DataFrame(rows, schema=VALIDATION_STATISTICS_SCHEMA)


def _add_inference(
    rows: list[dict[str, Any]],
    *,
    masks: np.ndarray,
    losses: np.ndarray,
    minimum_support_count: int,
    config: ValidationInference,
    max_batch_cells: int,
) -> None:
    inference_minimum = max(2, minimum_support_count)
    testable: list[int] = []
    for index, row in enumerate(rows):
        support = int(row["validation_support_count"])
        complement_support = losses.size - support
        if support < inference_minimum:
            row["inference_status"] = "not_tested_slice_support"
        elif complement_support < inference_minimum:
            row["inference_status"] = "not_tested_complement_support"
        else:
            row["inference_status"] = "tested"
            testable.append(index)

    if not testable:
        return

    seed_sequence = np.random.SeedSequence(config.random_seed)
    bootstrap_seed, permutation_seed = seed_sequence.spawn(2)
    bootstrap_rng = np.random.default_rng(bootstrap_seed)
    permutation_rng = np.random.default_rng(permutation_seed)
    tail = (1 - config.confidence_level) / 2

    for index in testable:
        selected = losses[masks[:, index]]
        bootstrap_means = _bootstrap_means(
            selected,
            resamples=config.bootstrap_resamples,
            rng=bootstrap_rng,
            max_batch_cells=max_batch_cells,
        )
        lower, upper = np.quantile(
            bootstrap_means,
            (tail, 1 - tail),
        )
        rows[index]["validation_error_mean_ci_lower"] = float(lower)
        rows[index]["validation_error_mean_ci_upper"] = float(upper)

    tested_masks = masks[:, testable].astype(np.float64)
    supports = tested_masks.sum(axis=0)
    complement_supports = losses.size - supports
    observed = np.asarray(
        [rows[index]["validation_mean_difference"] for index in testable],
        dtype=np.float64,
    )
    exceedances = np.zeros(len(testable), dtype=np.int64)
    for _ in range(config.permutation_resamples):
        permuted = permutation_rng.permutation(losses)
        selected_sums = tested_masks.T @ permuted
        differences = selected_sums / supports - (
            (permuted.sum() - selected_sums) / complement_supports
        )
        exceedances += differences >= observed
    raw_p_values = (exceedances + 1) / (config.permutation_resamples + 1)
    adjusted_p_values = _holm_adjust(raw_p_values)
    alpha = 1 - config.confidence_level
    for position, index in enumerate(testable):
        raw = float(raw_p_values[position])
        adjusted = float(adjusted_p_values[position])
        rows[index]["validation_permutation_p_value"] = raw
        rows[index]["validation_adjusted_p_value"] = adjusted
        rows[index]["validation_significant"] = adjusted <= alpha


def _bootstrap_means(
    values: np.ndarray,
    *,
    resamples: int,
    rng: np.random.Generator,
    max_batch_cells: int,
) -> np.ndarray:
    result = np.empty(resamples, dtype=np.float64)
    batch_size = max(1, min(resamples, max_batch_cells // values.size))
    for start in range(0, resamples, batch_size):
        stop = min(resamples, start + batch_size)
        indices = rng.integers(
            0,
            values.size,
            size=(stop - start, values.size),
        )
        result[start:stop] = values[indices].mean(axis=1)
    return result


def _holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Return monotone Holm-adjusted p-values in original order."""
    order = np.argsort(p_values, kind="stable")
    adjusted = np.empty_like(p_values, dtype=np.float64)
    running_max = 0.0
    count = p_values.size
    for position, original_index in enumerate(order):
        candidate = min(1.0, (count - position) * p_values[original_index])
        running_max = max(running_max, candidate)
        adjusted[original_index] = running_max
    return adjusted
