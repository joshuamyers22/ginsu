"""
The slicefinder module implements the Slicefinder class.
"""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Callable
from numbers import Integral
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import polars as pl
from scipy import sparse as sp
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted

from ginsu._domain import Predicate, Slice
from ginsu._frame import normalize_frame, schema_signature, validate_schema
from ginsu._validation import normalize_errors, to_engine_array
from ginsu.diagnostics import (
    SearchLevelReport,
    SearchLimitError,
    SearchLimits,
    SearchReport,
    SearchStageReport,
    SearchStageStatus,
    SearchStatus,
)

if TYPE_CHECKING:
    from ginsu.selection import (
        SelectionLimits,
        SelectionMethod,
        SliceSelection,
    )
    from ginsu.validation import (
        SliceValidation,
        ValidationInference,
        ValidationLimits,
    )

ArrayLike = npt.ArrayLike
NDArray = npt.NDArray[Any]

logger = logging.getLogger(__name__)

# Numba availability detection
score_slices_numba: Any = None
score_ub_batch_numba: Any = None
compute_slice_ids_numba: Any = None
try:
    from ginsu._numba_ops import (
        compute_slice_ids_numba as _compute_slice_ids_numba,
    )
    from ginsu._numba_ops import (
        score_slices_numba as _score_slices_numba,
    )
    from ginsu._numba_ops import (
        score_ub_batch_numba as _score_ub_batch_numba,
    )

    compute_slice_ids_numba = _compute_slice_ids_numba
    score_slices_numba = _score_slices_numba
    score_ub_batch_numba = _score_ub_batch_numba
    NUMBA_AVAILABLE = True
except (ImportError, RuntimeError):
    NUMBA_AVAILABLE = False


def _column_cardinality(series: pl.Series) -> int:
    """Count values, including NumPy-origin Polars Object columns."""
    try:
        return int(series.n_unique())
    except pl.exceptions.InvalidOperationError:
        return int(np.unique(series.to_numpy()).size)


def is_numba_available() -> bool:
    """Check if numba is available for acceleration.

    Returns
    -------
    bool
        True if numba is installed and can be used for acceleration.
    """
    return NUMBA_AVAILABLE


def _warn_numba_not_available() -> None:
    """Issue a warning if numba is not available."""
    warnings.warn(
        "Numba not available. Install with: pip install numba\n"
        "Or: pip install ginsu[optimized]\n"
        "Performance will be 5-50x slower without Numba optimization.",
        UserWarning,
        stacklevel=3,
    )


class Slicefinder(BaseEstimator, TransformerMixin):
    """Find high-loss subpopulations in categorical feature data.

    Given an input dataset (``X``) and one observed model-loss value per row,
    Ginsu returns the highest-scoring slices where loss is elevated. A slice
    is a subpopulation defined by one or more equality predicates. Discovery
    scores are descriptive ranking values, not tests of statistical
    significance.

    The maximal dimension of this subspace is controlled by `max_l`.

    The slice scoring function is the linear combination of two objectives:
        - Find sufficiently large slices, with more than `min_sup` elements
          (high impact on the overall model)
        - With substantial errors
          (high negative impact on sub-group/model)

    The importance of each objective is controlled through a single parameter `alpha`.

    Slice enumeration and pruning techniques are done via sparse linear algebra.

    Parameters
    ----------
    alpha: float, default=0.6
        Weight parameter for the importance of the average slice error.
        0 < `alpha` <= 1.

    k: int, default=1
        Maximum number of slices to return.
        Note: in case of equality between `k`-th slice score and the following ones,
        all those slices are returned, leading to `_n_features_out` slices returned.
        (`_n_features_out` >= `k`)

    max_l: int, default=4
        Maximum lattice level.
        In other words: the maximum number of predicate to define a slice.

    min_sup: int or float, default=10
        Minimum support threshold. An integer is a row count. A float in
        ``(0, 1)`` is a fraction of the discovery rows and is rounded upward.
        This threshold excludes small candidates; it does not establish
        statistical significance.

    verbose: bool, default=True
        Controls the verbosity.

    limits: SearchLimits or None, default=None
        Resource policy for cardinality, candidate generation, compatibility
        matrices, elapsed search time, and tied output. ``None`` uses the
        conservative :class:`ginsu.SearchLimits` defaults.

    clock: callable, default=time.perf_counter
        Injected monotonic clock used for elapsed-time limits and diagnostics.

    memory_sampler: callable or None, default=None
        Optional caller-supplied function returning a nonnegative byte count at
        search-stage boundaries. Samples are observations, not allocations
        attributed to Ginsu.

    memory_measurement: str or None, default=None
        Caller-declared semantics for ``memory_sampler`` such as
        ``"process_peak_rss_bytes"``. Required exactly when a sampler is set.

    Attributes
    ----------
    slices\\_ : polars.DataFrame
        Ranked rules with stable ``__ginsu_id`` values, display rules, and one
        nullable predicate-value column per input feature.
    slice_statistics\\_ : polars.DataFrame
        Canonical discovery metrics, including rank, score, support, observed
        error summaries, error lift, excess error, and predicate count.
    predicates\\_ : polars.DataFrame
        Long-form predicate table keyed by stable slice ID.
    average_error\\_ : float
        Mean discovery loss.
    feature_names_in\\_ : numpy.ndarray
        Ordered feature names captured during fitting.
    input_kind\\_ : str
        Normalized producer kind: ``polars``, ``numpy``, ``arrow``, or
        ``interchange``.
    top_slices\\_ : numpy.ndarray
        Legacy positional rule representation. ``None`` means that a feature
        is unused by a rule. Prefer ``slices_`` for application code.
    top_slices_statistics\\_ : list of dict
        Legacy list-of-dictionaries discovery statistics. Prefer
        ``slice_statistics_`` for application code.

    search_report\\_ : SearchReport
        Immutable execution evidence. Limit-terminated and failed searches
        expose this report but do not leave the estimator in a fitted state.

    References
    ----------
    `SliceLine: Fast, Linear-Algebra-based Slice Finding for ML Model Debugging
    <https://mboehm7.github.io/resources/sigmod2021b_sliceline.pdf>`__,
    from *Svetlana Sagadeeva* and *Matthias Boehm* of Graz University of Technology.
    """

    def __init__(
        self,
        alpha: float = 0.6,
        k: int = 1,
        max_l: int = 4,
        min_sup: int | float = 10,
        verbose: bool = True,
        limits: SearchLimits | None = None,
        clock: Callable[[], float] = perf_counter,
        memory_sampler: Callable[[], int] | None = None,
        memory_measurement: str | None = None,
    ) -> None:
        self.alpha = alpha
        self.k = k
        self.max_l = max_l
        self.min_sup = min_sup
        self.verbose = verbose
        self.limits = limits
        self.clock = clock
        self.memory_sampler = memory_sampler
        self.memory_measurement = memory_measurement

        self._one_hot_encoder: OneHotEncoder | None = None
        self._top_slices_enc: sp.csr_matrix | None = None
        self._min_sup_actual = min_sup

        if self.verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        # Warn user once if Numba optimization is not available
        if not NUMBA_AVAILABLE and verbose:
            warnings.warn(
                "Numba JIT optimization not available. "
                "Install with 'pip install ginsu[optimized]' "
                "for 5-50x performance improvements on scoring operations. "
                "See the Ginsu performance documentation for details.",
                UserWarning,
                stacklevel=2,
            )

    def _check_params(self) -> None:
        """Check transformer parameters."""
        if not 0 < self.alpha <= 1:
            raise ValueError(f"Invalid 'alpha' parameter: {self.alpha}")

        if self.k <= 0:
            raise ValueError(f"Invalid 'k' parameter: {self.k}")

        if self.max_l <= 0:
            raise ValueError(f"Invalid 'max_l' parameter: {self.max_l}")

        if self.min_sup < 0 or (
            isinstance(self.min_sup, float) and self.min_sup >= 1
        ):
            raise ValueError(f"Invalid 'min_sup' parameter: {self.min_sup}")

        if self.limits is not None and not isinstance(
            self.limits, SearchLimits
        ):
            raise TypeError("limits must be a SearchLimits instance or None.")
        if not callable(self.clock):
            raise TypeError("clock must be callable.")
        if self.memory_sampler is not None and not callable(
            self.memory_sampler
        ):
            raise TypeError("memory_sampler must be callable or None.")
        if (self.memory_sampler is None) != (self.memory_measurement is None):
            raise ValueError(
                "memory_sampler and memory_measurement must be provided together."
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

    def _clear_fitted_state(self) -> None:
        """Remove prior results before starting a new fit."""
        for name in (
            "feature_names_in_",
            "_feature_schema",
            "input_kind_",
            "average_error_",
            "top_slices_",
            "top_slices_statistics_",
            "slices_",
            "slice_statistics_",
            "predicates_",
            "_slice_objects",
            "search_report_",
        ):
            if hasattr(self, name):
                delattr(self, name)
        self._one_hot_encoder = None
        self._top_slices_enc = None

    def _raise_limit(
        self,
        code: str,
        *,
        observed: int | float,
        limit: int | float,
        stage: str,
    ) -> None:
        raise SearchLimitError(
            code, observed=observed, limit=limit, stage=stage
        )

    def _read_memory_sample(self) -> int | None:
        if self.memory_sampler is None:
            return None
        value: Any = self.memory_sampler()
        if (
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or value < 0
        ):
            raise ValueError(
                "memory_sampler must return a nonnegative integer byte count."
            )
        return int(value)

    def _initialize_search_diagnostics(
        self, started_at: float, *, stage: str
    ) -> None:
        initial_memory = self._read_memory_sample()
        self._search_stages: list[SearchStageReport] = []
        self._active_search_stage: str | None = stage
        self._active_stage_started_at = started_at
        self._active_stage_memory_start = initial_memory
        self._diagnostic_last_at = started_at
        self._diagnostic_last_memory = initial_memory
        self._observed_peak_memory_bytes = initial_memory

    def _finish_active_search_stage(
        self,
        status: SearchStageStatus,
        *,
        check_elapsed: bool = True,
        sample_memory: bool = True,
    ) -> None:
        if self._active_search_stage is None:
            return
        finished_at = self.clock()
        if finished_at < self._active_stage_started_at:
            raise ValueError("clock must be monotonic.")
        memory_end = self._read_memory_sample() if sample_memory else None
        observed = [
            value
            for value in (self._active_stage_memory_start, memory_end)
            if value is not None
        ]
        stage_peak = max(observed) if observed else None
        report = SearchStageReport(
            stage=self._active_search_stage,
            status=status,
            elapsed_seconds=finished_at - self._active_stage_started_at,
            memory_start_bytes=self._active_stage_memory_start,
            memory_end_bytes=memory_end,
            observed_peak_memory_bytes=stage_peak,
        )
        self._search_stages.append(report)
        if stage_peak is not None:
            current_peak = self._observed_peak_memory_bytes
            self._observed_peak_memory_bytes = (
                stage_peak
                if current_peak is None
                else max(current_peak, stage_peak)
            )
        self._diagnostic_last_at = finished_at
        self._diagnostic_last_memory = memory_end
        self._active_search_stage = None
        if check_elapsed:
            limit = self._active_search_limits.max_search_seconds
            elapsed = finished_at - self._search_started_at
            if limit is not None and elapsed > limit:
                self._raise_limit(
                    "GINSU_MAX_SEARCH_SECONDS",
                    observed=elapsed,
                    limit=limit,
                    stage=report.stage.replace("_", " "),
                )

    def _transition_search_stage(self, stage: str) -> None:
        self._finish_active_search_stage("complete")
        self._active_search_stage = stage
        self._active_stage_started_at = self._diagnostic_last_at
        self._active_stage_memory_start = self._diagnostic_last_memory
        if self._active_stage_memory_start is not None:
            current_peak = self._observed_peak_memory_bytes
            self._observed_peak_memory_bytes = (
                self._active_stage_memory_start
                if current_peak is None
                else max(current_peak, self._active_stage_memory_start)
            )

    def _terminate_active_search_stage(self) -> None:
        try:
            self._finish_active_search_stage(
                "terminated", check_elapsed=False, sample_memory=False
            )
        except Exception:
            self._active_search_stage = None

    def _check_elapsed(self, stage: str) -> None:
        limit = self._active_search_limits.max_search_seconds
        if limit is None:
            return
        observed_at = self.clock()
        if observed_at < self._search_started_at:
            raise ValueError("clock must be monotonic.")
        elapsed = observed_at - self._search_started_at
        if elapsed > limit:
            self._raise_limit(
                "GINSU_MAX_SEARCH_SECONDS",
                observed=elapsed,
                limit=limit,
                stage=stage,
            )

    def _check_tie_limit(self, count: int, stage: str) -> None:
        limit = self._active_search_limits.max_tied_slices
        if limit is not None and count > limit:
            self._raise_limit(
                "GINSU_MAX_TIED_SLICES",
                observed=count,
                limit=limit,
                stage=stage,
            )

    def _check_top_slices(self) -> None:
        """Check if slices have been found."""
        # Check if fit has been called
        check_is_fitted(self, ("top_slices_", "_feature_schema"))

        # Check if a slice has been found
        if self.top_slices_.size == 0:
            raise ValueError("No transform: Ginsu did not find any slice.")

    def __sklearn_is_fitted__(self) -> bool:
        """Keep diagnostic-only failed fits from satisfying fitted checks."""
        return hasattr(self, "top_slices_") and hasattr(
            self, "_feature_schema"
        )

    def fit(self, X: ArrayLike, errors: ArrayLike) -> Slicefinder:
        """Search for slices on ``X`` using one observed loss per row.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            Categorical or pre-discretized discovery features. Named inputs
            preserve their ordered schema; NumPy-like inputs receive generated
            ``column_<index>`` names.

        errors: array-like of shape (n_samples, )
            Finite, nonnegative per-row model losses. At least one value must
            be positive.

        Returns
        -------
        self: object
            Returns the instance itself.
        """
        self._clear_fitted_state()
        self._check_params()
        started_at = self.clock()
        active_limits = self.limits or SearchLimits()
        self._active_search_limits = active_limits
        self._search_started_at = started_at
        self._initialize_search_diagnostics(
            started_at, stage="input_normalization"
        )

        # Validate and normalize inputs before deriving any fitted state.
        normalized = normalize_frame(X)
        signature = schema_signature(normalized.frame)
        cardinalities = tuple(
            (name, _column_cardinality(normalized.frame.get_column(name)))
            for name in normalized.frame.columns
        )
        copy_boundaries = tuple(
            (
                [f"{normalized.kind}->polars"]
                if normalized.kind != "polars"
                else []
            )
            + ["polars->numpy", "numpy->scipy-csr"]
        )

        self._search_levels: list[SearchLevelReport] = []
        self._encoded_feature_count: int | None = None
        self._numba_used = False

        def build_report(
            status: SearchStatus,
            *,
            reason: str | None = None,
            warning_codes: tuple[str, ...] = (),
        ) -> SearchReport:
            return SearchReport(
                status=status,
                backend="numba" if self._numba_used else "numpy",
                numba_used=self._numba_used,
                input_kind=normalized.kind,
                input_schema=signature,
                row_count=normalized.frame.height,
                feature_count=normalized.frame.width,
                feature_cardinalities=cardinalities,
                encoded_feature_count=self._encoded_feature_count,
                copy_boundaries=copy_boundaries,
                levels=tuple(self._search_levels),
                elapsed_seconds=self._diagnostic_last_at - started_at,
                limits=active_limits,
                warning_codes=warning_codes,
                termination_reason=reason,
                stages=tuple(self._search_stages),
                memory_measurement=self.memory_measurement,
                observed_peak_memory_bytes=(self._observed_peak_memory_bytes),
            )

        try:
            cardinality_limit = active_limits.max_feature_cardinality
            if cardinality_limit is not None:
                for name, cardinality in cardinalities:
                    if cardinality > cardinality_limit:
                        self._raise_limit(
                            "GINSU_MAX_FEATURE_CARDINALITY",
                            observed=cardinality,
                            limit=cardinality_limit,
                            stage=f"feature {name!r}",
                        )

            self._transition_search_stage("engine_conversion")
            X_array = to_engine_array(normalized.frame)
            normalized_errors = normalize_errors(
                errors, expected_length=normalized.frame.height
            )

            # Compute actual min_sup value (convert fraction to count if needed)
            if 0 < self.min_sup < 1:
                self._min_sup_actual = max(
                    1, math.ceil(self.min_sup * X_array.shape[0])
                )
            else:
                self._min_sup_actual = self.min_sup

            self._transition_search_stage("one_hot_encoding")
            self._search_slices(
                X_array, normalized_errors, started_at=started_at
            )
            self._transition_search_stage("result_materialization")
            self._build_result_frames(normalized.frame)
            self._finish_active_search_stage("complete")
        except SearchLimitError as error:
            self._terminate_active_search_stage()
            self._clear_fitted_state()
            self.search_report_ = build_report(
                "limit_reached",
                reason=str(error),
                warning_codes=(error.code,),
            )
            raise
        except Exception as error:
            self._terminate_active_search_stage()
            self._clear_fitted_state()
            self.search_report_ = build_report("failed", reason=str(error))
            raise

        self.feature_names_in_ = np.asarray(
            normalized.frame.columns, dtype=object
        )
        self._feature_schema = signature
        self.input_kind_ = normalized.kind
        status: SearchStatus = (
            "no_valid_slices" if self.slices_.height == 0 else "complete"
        )
        self.search_report_ = build_report(status)

        return self

    def transform(self, X: ArrayLike) -> NDArray | Any:
        """Generate slices masks for `X`.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        Returns
        -------
        slices_masks: np.ndarray or polars.DataFrame
            NumPy-like input produces an ndarray. Polars, Arrow, and dataframe
            interchange inputs produce a Polars DataFrame.

            The result has shape (n_samples, _n_features_out), and
            `slices_masks[i, j] == 1`: the `i`-th sample of `X` is in the `j`-th `top_slices_`.
        """
        self._check_top_slices()

        normalized = normalize_frame(X)
        validate_schema(normalized.frame, expected=self._feature_schema)
        X_array = to_engine_array(normalized.frame)

        slices_masks = self._get_slices_masks(X_array).T.astype(bool)

        if normalized.kind == "numpy":
            return slices_masks

        return pl.DataFrame(
            slices_masks,
            schema=self.get_feature_names_out().tolist(),
            orient="row",
        )

    def get_slice(self, X: ArrayLike, slice_index: int) -> NDArray | Any:
        """Filter `X` samples according to the `slice_index`-th slice.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            Dataset, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        slice_index: int
            Index of the slice to get from `top_slices_`.

        Returns
        -------
        X_slice: np.ndarray or polars.DataFrame
            Filtered samples. NumPy-like input produces an ndarray. Polars,
            Arrow, and dataframe interchange inputs produce a Polars DataFrame.
        """
        self._check_top_slices()

        normalized = normalize_frame(X)
        validate_schema(normalized.frame, expected=self._feature_schema)
        X_array = to_engine_array(normalized.frame)

        slices_masks = self._get_slices_masks(X_array)
        row_mask = slices_masks[slice_index].astype(bool)

        if normalized.kind == "numpy":
            return X_array[row_mask, :]

        return normalized.frame.filter(row_mask)

    def get_feature_names_out(self) -> NDArray:
        """Get output feature names for transformation.

        Returns
        -------
        feature_names_out : ndarray of str objects
            The following output feature names are generated:
            `["slice_0", "slice_1", ..., "slice_(_n_features_out)"]`.
        """
        check_is_fitted(self)

        feature_names = [f"slice_{i}" for i in range(self._n_features_out)]

        return np.array(feature_names, dtype=object)

    def membership_frame(
        self, X: ArrayLike, *, row_id: str | pl.Series | None = None
    ) -> pl.DataFrame:
        """Return positional row identity and stable Boolean slice columns.

        Parameters
        ----------
        X:
            Schema-compatible observations.
        row_id:
            ``None`` generates a positional UInt64 row number. A string uses
            that input column as row identity. A Polars Series supplies an
            explicit identifier with one value per observation.

        Returns
        -------
        polars.DataFrame
            ``__ginsu_row`` followed by one Boolean column per stable slice ID.
        """
        check_is_fitted(self, ("slices_", "_feature_schema"))
        normalized = normalize_frame(X)
        validate_schema(normalized.frame, expected=self._feature_schema)
        X_array = to_engine_array(normalized.frame)

        if row_id is None:
            row_values = pl.Series(
                "__ginsu_row",
                range(normalized.frame.height),
                dtype=pl.UInt64,
            )
        elif isinstance(row_id, str):
            if row_id not in normalized.frame.columns:
                raise ValueError(f"Unknown row_id column {row_id!r}.")
            row_values = normalized.frame.get_column(row_id).alias(
                "__ginsu_row"
            )
        elif isinstance(row_id, pl.Series):
            if row_id.len() != normalized.frame.height:
                raise ValueError(
                    "row_id length must match the number of observations."
                )
            row_values = row_id.alias("__ginsu_row")
        else:
            raise TypeError(
                "row_id must be None, a column name, or a Polars Series."
            )

        if row_values.n_unique() != normalized.frame.height:
            raise ValueError("row_id values must be unique.")

        result = pl.DataFrame({"__ginsu_row": row_values})
        if self.slices_.height == 0:
            return result

        masks = self._get_slices_masks(X_array).T.astype(bool)
        identifiers = self.slices_.get_column("__ginsu_id").to_list()
        return result.with_columns(
            [
                pl.Series(identifier, masks[:, index], dtype=pl.Boolean)
                for index, identifier in enumerate(identifiers)
            ]
        )

    def _get_slices_masks(self, X: NDArray) -> NDArray:
        """Private utilities function generating slices masks for `X`."""
        if self._one_hot_encoder is None or self._top_slices_enc is None:
            raise RuntimeError("Fitted encoder state is unavailable.")
        X_encoded = self._one_hot_encoder.transform(X)

        # Shape X_encoded: (X.shape[0], total number of modalities in _one_hot_encoder.categories_)
        # Shape _top_slices_enc: (top_slices_.shape[0], X_encoded[1])
        slice_candidates = self._top_slices_enc @ X_encoded.T

        # self._top_slices_enc.sum(axis=1) is the number of predicate(s) for each top_slices_
        slices_masks = (
            slice_candidates == self._top_slices_enc.sum(axis=1)
        ).A.astype(int)

        return slices_masks

    def _build_result_frames(self, input_frame: pl.DataFrame) -> None:
        """Build canonical Polars result tables from characterized engine output."""
        slices = []
        for row in self.top_slices_:
            predicates = tuple(
                Predicate(feature, value)
                for feature, value in zip(
                    input_frame.columns, row, strict=True
                )
                if value is not None
            )
            slices.append(Slice(predicates))

        identifiers = [item.id for item in slices]
        ranks = list(range(1, len(slices) + 1))
        result_columns: dict[str, pl.Series] = {
            "__ginsu_id": pl.Series(
                "__ginsu_id", identifiers, dtype=pl.String
            ),
            "__ginsu_rank": pl.Series("__ginsu_rank", ranks, dtype=pl.UInt32),
            "__ginsu_rule": pl.Series(
                "__ginsu_rule",
                [item.rule for item in slices],
                dtype=pl.String,
            ),
        }
        for column_index, (name, dtype) in enumerate(
            input_frame.schema.items()
        ):
            values = [row[column_index] for row in self.top_slices_]
            result_columns[name] = pl.Series(
                name,
                values,
                dtype=dtype if dtype != pl.Object else pl.Object,
                strict=False,
            )
        self.slices_ = pl.DataFrame(result_columns)

        statistics = self.top_slices_statistics_
        support = [int(item["slice_size"]) for item in statistics]
        error_sum = [item["sum_slice_error"] for item in statistics]
        error_mean = [item["slice_average_error"] for item in statistics]
        self.slice_statistics_ = pl.DataFrame(
            {
                "__ginsu_id": pl.Series(identifiers, dtype=pl.String),
                "rank": pl.Series(ranks, dtype=pl.UInt32),
                "slice_score": pl.Series(
                    [item["slice_score"] for item in statistics],
                    dtype=pl.Float64,
                ),
                "support_count": pl.Series(support, dtype=pl.UInt64),
                "support_fraction": pl.Series(
                    [value / input_frame.height for value in support],
                    dtype=pl.Float64,
                ),
                "error_sum": pl.Series(error_sum, dtype=pl.Float64),
                "error_max": pl.Series(
                    [item["max_slice_error"] for item in statistics],
                    dtype=pl.Float64,
                ),
                "error_mean": pl.Series(error_mean, dtype=pl.Float64),
                "baseline_error_mean": pl.Series(
                    [self.average_error_] * len(slices), dtype=pl.Float64
                ),
                "error_lift": pl.Series(
                    [value / self.average_error_ for value in error_mean],
                    dtype=pl.Float64,
                ),
                "excess_error": pl.Series(
                    [
                        total - count * self.average_error_
                        for total, count in zip(
                            error_sum, support, strict=True
                        )
                    ],
                    dtype=pl.Float64,
                ),
                "predicate_count": pl.Series(
                    [len(item.predicates) for item in slices], dtype=pl.UInt32
                ),
            }
        )

        predicate_rows = []
        dtype_lookup = {
            name: str(dtype) for name, dtype in input_frame.schema.items()
        }
        for rank, item in enumerate(slices, start=1):
            for position, predicate in enumerate(item.predicates):
                predicate_rows.append(
                    {
                        "__ginsu_id": item.id,
                        "rank": rank,
                        "predicate_position": position,
                        "feature": predicate.feature,
                        "operator": predicate.operator,
                        "value_json": predicate.value_json,
                        "display_value": predicate.display_value,
                        "source_dtype": dtype_lookup[predicate.feature],
                    }
                )
        self.predicates_ = pl.DataFrame(
            predicate_rows,
            schema={
                "__ginsu_id": pl.String,
                "rank": pl.UInt32,
                "predicate_position": pl.UInt32,
                "feature": pl.String,
                "operator": pl.String,
                "value_json": pl.String,
                "display_value": pl.String,
                "source_dtype": pl.String,
            },
        )
        self._slice_objects = tuple(slices)

    def equivalence_groups(
        self,
        X: ArrayLike,
        *,
        max_slices: int = 100,
        max_membership_cells: int = 10_000_000,
    ) -> pl.DataFrame:
        """Group returned rules with exactly equal membership on ``X``."""
        from ginsu._plot_data import equivalence_groups

        return equivalence_groups(
            self,
            X,
            max_slices=max_slices,
            max_membership_cells=max_membership_cells,
        )

    def overlap_frame(
        self,
        X: ArrayLike,
        *,
        metric: str = "jaccard",
        max_slices: int = 100,
        max_cells: int = 10_000,
        max_membership_cells: int = 10_000_000,
    ) -> pl.DataFrame:
        """Return a bounded pairwise overlap table for discovered rules."""
        from ginsu._plot_data import overlap_data

        return overlap_data(
            self,
            X,
            metric=metric,
            max_slices=max_slices,
            max_cells=max_cells,
            max_membership_cells=max_membership_cells,
        )

    def lattice_edges(self, *, max_nodes: int = 100) -> pl.DataFrame:
        """Return exact one-predicate refinement edges among returned rules."""
        from ginsu._plot_data import lattice_edges_data

        return lattice_edges_data(self, max_nodes=max_nodes)

    def validate_slices(
        self,
        X: ArrayLike,
        errors: ArrayLike,
        *,
        min_support: int | float | None = None,
        limits: ValidationLimits | None = None,
        inference: ValidationInference | None = None,
    ) -> SliceValidation:
        """Evaluate fixed discovered rules descriptively on holdout data."""
        from ginsu.validation import validate_slices

        return validate_slices(
            self,
            X,
            errors,
            min_support=min_support,
            limits=limits,
            inference=inference,
        )

    def select_slices(
        self,
        X: ArrayLike,
        *,
        method: SelectionMethod = "score",
        k: int = 20,
        max_jaccard: float = 0.8,
        limits: SelectionLimits | None = None,
    ) -> SliceSelection:
        """Build an auditable post-selection view over fixed discoveries."""
        from ginsu.selection import select_slices

        return select_slices(
            self,
            X,
            method=method,
            k=k,
            max_jaccard=max_jaccard,
            limits=limits,
        )

    @property
    def _n_features_out(self) -> int:
        """Number of transformed output features."""
        return self.top_slices_.shape[0]

    @staticmethod
    def _dummify(array: NDArray, n_col_x_encoded: int) -> sp.csr_matrix:
        """Dummify `array` with respect to `n_col_x_encoded`.

        Creates a sparse one-hot encoding matrix where each row corresponds
        to an element in array and has a single True value in the column
        specified by that element (adjusted for 1-based indexing).

        Args:
            array: 1-based indices to encode (must not contain 0)
            n_col_x_encoded: Number of columns in output matrix

        Returns:
            Sparse CSR matrix of shape (len(array), n_col_x_encoded)

        Raises:
            ValueError: If array contains 0, which cannot be one-hot encoded.
        """
        if 0 in array:
            raise ValueError(
                "Modality 0 is not expected to be one-hot encoded."
            )

        # Direct CSR construction: 2-3x faster than lil_matrix approach
        n = array.size
        return sp.csr_matrix(
            (np.ones(n, dtype=np.bool_), (np.arange(n), array - 1)),
            shape=(n, n_col_x_encoded),
            dtype=np.bool_,
        )

    def _maintain_top_k(
        self,
        slices: sp.csr_matrix,
        statistics: NDArray,
        top_k_slices: sp.csr_matrix,
        top_k_statistics: NDArray,
    ) -> tuple[sp.csr_matrix, NDArray]:
        """Add new `slices` to `top_k_slices` and update the top-k slices."""
        # prune invalid min_sup and scores
        valid_slices_mask = (statistics[:, 3] >= self._min_sup_actual) & (
            statistics[:, 0] > 0
        )
        if np.sum(valid_slices_mask) != 0:
            slices, statistics = (
                slices[valid_slices_mask],
                statistics[valid_slices_mask],
            )

            if (slices.shape[1] != top_k_slices.shape[1]) and (
                slices.shape[1] == 1
            ):
                slices, statistics = slices.T, statistics.T

            # evaluated candidates and previous top-k
            slices = sp.vstack([top_k_slices, slices])
            statistics = np.concatenate([top_k_statistics, statistics])

            # extract top-k
            top_slices_bool = (
                rankdata(-statistics[:, 0], method="min") <= self.k
            )
            top_k_slices, top_k_statistics = (
                slices[top_slices_bool],
                statistics[top_slices_bool],
            )
            # Sort by score (descending), then lexicographically by slice representation
            # to ensure deterministic ordering when scores are equal
            scores = -top_k_statistics[:, 0]
            slice_keys = tuple(
                top_k_slices.toarray()[:, i]
                for i in range(top_k_slices.shape[1])
            )
            top_slices_indices = np.lexsort(slice_keys[::-1] + (scores,))
            top_k_slices, top_k_statistics = (
                top_k_slices[top_slices_indices],
                top_k_statistics[top_slices_indices],
            )
        return top_k_slices, top_k_statistics

    def _score_ub(
        self,
        slice_sizes_ub: NDArray,
        slice_errors_ub: NDArray,
        max_slice_errors_ub: NDArray,
        n_col_x_encoded: int,
    ) -> NDArray:
        """Compute the upper-bound score for all the slices.

        Uses Numba JIT compilation when available for 5-10x speedup.
        """
        if NUMBA_AVAILABLE and score_ub_batch_numba is not None:
            self._numba_used = True
            return score_ub_batch_numba(
                slice_sizes_ub.astype(np.float64),
                slice_errors_ub.astype(np.float64),
                max_slice_errors_ub.astype(np.float64),
                n_col_x_encoded,
                float(self._min_sup_actual),
                self.alpha,
                self.average_error_,
            )

        # Fallback to NumPy implementation
        # Since slice_scores is either monotonically increasing or decreasing, we
        # probe interesting points of slice_scores in the interval [min_sup, ss],
        # and compute the maximum to serve as the upper bound
        potential_solutions = np.column_stack(
            (
                self._min_sup_actual * np.ones(slice_sizes_ub.shape[0]),
                np.maximum(
                    slice_errors_ub / max_slice_errors_ub, self._min_sup_actual
                ),
                slice_sizes_ub,
            )
        )
        slice_scores_ub = np.amax(
            (
                self.alpha
                * (
                    np.minimum(
                        potential_solutions.T * max_slice_errors_ub,
                        slice_errors_ub,
                    ).T
                    / self.average_error_
                    - potential_solutions
                )
                - (1 - self.alpha) * (n_col_x_encoded - potential_solutions)
            )
            / potential_solutions,
            axis=1,
        )
        return slice_scores_ub

    @staticmethod
    def _analyse_top_k(top_k_statistics: NDArray) -> tuple[float, float]:
        """Get the maximum and the minimum slices scores."""
        max_slice_scores = min_slice_scores = -np.inf
        if top_k_statistics.shape[0] > 0:
            max_slice_scores = top_k_statistics[0, 0]
            min_slice_scores = top_k_statistics[
                top_k_statistics.shape[0] - 1, 0
            ]
        return max_slice_scores, min_slice_scores

    def _score(
        self,
        slice_sizes: NDArray,
        slice_errors: NDArray,
        n_row_x_encoded: int,
    ) -> NDArray:
        """Compute the score for all the slices.

        Uses Numba JIT compilation when available for 5-10x speedup.
        """
        if NUMBA_AVAILABLE and score_slices_numba is not None:
            self._numba_used = True
            # Ensure inputs are float64 for numba
            sizes = np.asarray(slice_sizes, dtype=np.float64)
            errors = np.asarray(slice_errors, dtype=np.float64)
            return score_slices_numba(
                sizes,
                errors,
                n_row_x_encoded,
                self.alpha,
                self.average_error_,
            )

        # Fallback to NumPy implementation
        with np.errstate(divide="ignore", invalid="ignore"):
            slice_scores = self.alpha * (
                (slice_errors / slice_sizes) / self.average_error_ - 1
            ) - (1 - self.alpha) * (n_row_x_encoded / slice_sizes - 1)
        return np.nan_to_num(slice_scores, nan=-np.inf)

    def _eval_slice(
        self,
        x_encoded: sp.csr_matrix,
        errors: NDArray,
        slices: sp.csr_matrix,
        level: int,
    ) -> NDArray:
        """Compute several statistics for all the slices."""
        slice_candidates = x_encoded @ slices.T == level
        slice_sizes = slice_candidates.sum(axis=0).A[0]
        slice_errors = errors @ slice_candidates
        # Here we can't use the .A shorthand because it is not
        # implemented in all scipy versions for coo_matrix objects
        max_slice_errors = (
            slice_candidates.T.multiply(errors).max(axis=1).toarray()
        )

        # score of relative error and relative size
        slice_scores = self._score(
            slice_sizes, slice_errors, x_encoded.shape[0]
        )
        return np.column_stack(
            [slice_scores, slice_errors, max_slice_errors, slice_sizes]
        )

    def _create_and_score_basic_slices(
        self,
        x_encoded: sp.csr_matrix,
        n_col_x_encoded: int,
        errors: NDArray,
    ) -> tuple[sp.csr_matrix, NDArray]:
        """Initialise 1-slices, i.e. slices with one predicate."""
        slice_sizes = x_encoded.sum(axis=0).A[0]
        slice_errors = errors @ x_encoded
        # Here we can't use the .A shorthand because it is not
        # implemented in all scipy versions for coo_matrix objects
        max_slice_errors = (
            x_encoded.T.multiply(errors).max(axis=1).toarray()[:, 0]
        )

        # working set of active slices (#attr x #slices) and top-k
        valid_slices_mask = (slice_sizes >= self._min_sup_actual) & (
            slice_errors > 0
        )
        attr = np.arange(1, n_col_x_encoded + 1)[valid_slices_mask]
        slice_sizes = slice_sizes[valid_slices_mask]
        slice_errors = slice_errors[valid_slices_mask]
        max_slice_errors = max_slice_errors[valid_slices_mask]
        slices = self._dummify(attr, n_col_x_encoded)

        # score 1-slices and create initial top-k
        slice_scores = self._score(
            slice_sizes, slice_errors, x_encoded.shape[0]
        )
        statistics = np.column_stack(
            (slice_scores, slice_errors, max_slice_errors, slice_sizes)
        )

        n_col_dropped = n_col_x_encoded - sum(valid_slices_mask)
        logger.debug(
            "Dropping %i/%i features below min_sup = %i.",
            n_col_dropped,
            n_col_x_encoded,
            self._min_sup_actual,
        )

        return slices, statistics

    def _get_pruned_s_r(
        self, slices: sp.csr_matrix, statistics: NDArray
    ) -> tuple[sp.csr_matrix, NDArray]:
        """Prune invalid slices.
        Do not affect overall pruning effectiveness due to handling of missing parents.
        """
        valid_slices_mask = (statistics[:, 3] >= self._min_sup_actual) & (
            statistics[:, 1] > 0
        )
        return slices[valid_slices_mask], statistics[valid_slices_mask]

    @staticmethod
    def _join_compatible_slices(
        slices: sp.csr_matrix,
        level: int,
        *,
        max_pair_matrix_bytes: int | None = None,
    ) -> sp.csr_matrix:
        """Join compatible slices keeping sparse format when beneficial.

        Returns a sparse boolean matrix where entry (i,j) is True if slices
        i and j are compatible for joining at the given level. Only upper
        triangular entries (i < j) are populated.

        For level==2 (looking for disjoint slices), uses dense format since
        most pairs are compatible. For higher levels, keeps sparse format.
        """
        n_slices = slices.shape[0]
        if n_slices == 0:
            return sp.csr_matrix((0, 0), dtype=np.bool_)

        # ``join_counts.toarray()``, the comparison, and ``np.triu`` coexist
        # briefly. Estimate all three dense buffers before multiplication.
        estimated_bytes = (
            n_slices
            * n_slices
            * (np.dtype(np.int64).itemsize + 2 * np.dtype(np.bool_).itemsize)
        )
        if (
            max_pair_matrix_bytes is not None
            and estimated_bytes > max_pair_matrix_bytes
        ):
            raise SearchLimitError(
                "GINSU_MAX_PAIR_MATRIX_BYTES",
                observed=estimated_bytes,
                limit=max_pair_matrix_bytes,
                stage=f"level {level} compatibility matrix",
            )

        slices_int = slices.astype(int)
        join_counts = slices_int @ slices_int.T

        if level == 2:
            # For level 2, we're looking for pairs with dot product == 0
            # Most pairs will match, so dense is more efficient
            join_dense = join_counts.toarray() == 0
        else:
            # For higher levels, most pairs won't match, so sparse is better
            # Use dense conversion for smaller matrices to ensure consistent ordering
            # This matches the original behavior and ensures deterministic results
            join_dense = join_counts.toarray() == level - 2

        join_upper = np.triu(join_dense, 1)
        rows, cols = np.where(join_upper)
        return sp.csr_matrix(
            (np.ones(len(rows), dtype=np.bool_), (rows, cols)),
            shape=join_counts.shape,
            dtype=np.bool_,
        )

    @staticmethod
    def _combine_slices(
        slices: sp.csr_matrix,
        statistics: NDArray,
        compatible_slices: sp.csr_matrix,
    ) -> tuple[sp.csr_matrix, NDArray, NDArray, NDArray]:
        """Combine slices by exploiting parents node statistics.

        Works with sparse compatible_slices matrix returned by
        _join_compatible_slices.
        """
        parent_1_idx, parent_2_idx = compatible_slices.nonzero()
        pair_candidates = slices[parent_1_idx] + slices[parent_2_idx]

        slice_errors = np.minimum(
            statistics[parent_1_idx, 1], statistics[parent_2_idx, 1]
        )
        max_slice_errors = np.minimum(
            statistics[parent_1_idx, 2], statistics[parent_2_idx, 2]
        )
        slice_sizes = np.minimum(
            statistics[parent_1_idx, 3], statistics[parent_2_idx, 3]
        )
        return pair_candidates, slice_sizes, slice_errors, max_slice_errors

    @staticmethod
    def _prune_invalid_self_joins(
        feature_offset_start: NDArray,
        feature_offset_end: NDArray,
        pair_candidates: sp.csr_matrix,
        slice_sizes: NDArray,
        slice_errors: NDArray,
        max_slice_errors: NDArray,
    ) -> tuple[sp.csr_matrix, NDArray, NDArray, NDArray]:
        """Prune invalid self joins (>1 bit per feature)."""
        valid_slices_mask = np.full(pair_candidates.shape[0], True)
        for start, end in zip(
            feature_offset_start, feature_offset_end, strict=False
        ):
            valid_slices_mask = (
                valid_slices_mask
                * (pair_candidates[:, start:end].sum(axis=1) <= 1).A[:, 0]
            )
        return (
            pair_candidates[valid_slices_mask],
            slice_sizes[valid_slices_mask],
            slice_errors[valid_slices_mask],
            max_slice_errors[valid_slices_mask],
        )

    def _prepare_deduplication_and_pruning(
        self,
        feature_offset_start: NDArray,
        feature_offset_end: NDArray,
        feature_domains: NDArray,
        pair_candidates: sp.csr_matrix,
    ) -> NDArray:
        """Prepare IDs for deduplication and pruning.

        Uses Numba JIT compilation when available for 10-50x speedup.
        """
        if NUMBA_AVAILABLE and compute_slice_ids_numba is not None:
            self._numba_used = True
            return compute_slice_ids_numba(
                pair_candidates.data.astype(np.float64),
                pair_candidates.indices.astype(np.int64),
                pair_candidates.indptr.astype(np.int64),
                feature_offset_start.astype(np.int64),
                feature_offset_end.astype(np.int64),
                feature_domains.astype(np.float64),
            )

        # Fallback to Python implementation
        ids = np.zeros(pair_candidates.shape[0])
        dom = feature_domains + 1
        for j, (start, end) in enumerate(
            zip(feature_offset_start, feature_offset_end, strict=False)
        ):
            sub_pair_candidates = pair_candidates[:, start:end]
            # sub_p should not contain multiple True on the same line
            i = sub_pair_candidates.argmax(axis=1).T + np.any(
                # Here we can't use the .A shorthand because it is not
                # implemented in all scipy versions for coo_matrix objects
                sub_pair_candidates.toarray(),
                axis=1,
            )
            ids = ids + i.A * np.prod(dom[(j + 1) : dom.shape[0]])
        return ids

    def _get_pair_candidates(
        self,
        slices: sp.csr_matrix,
        statistics: NDArray,
        top_k_statistics: NDArray,
        level: int,
        n_col_x_encoded: int,
        feature_domains: NDArray,
        feature_offset_start: NDArray,
        feature_offset_end: NDArray,
    ) -> sp.csr_matrix:
        """Compute and prune plausible slices candidates."""
        if not hasattr(self, "_active_search_limits"):
            self._active_search_limits = self.limits or SearchLimits()
        if not hasattr(self, "_search_started_at"):
            self._search_started_at = self.clock()
        if not hasattr(self, "_generated_candidate_total"):
            self._generated_candidate_total = 0

        self._check_elapsed(f"level {level} join")
        self._last_potential_pairs = (
            slices.shape[0] * (slices.shape[0] - 1) // 2
        )
        compatible_slices = self._join_compatible_slices(
            slices,
            level,
            max_pair_matrix_bytes=(
                self._active_search_limits.max_pair_matrix_bytes
            ),
        )
        compatible_count = int(compatible_slices.nnz)
        self._last_compatible_pairs = compatible_count
        self._last_candidates_after_pruning = 0

        level_limit = self._active_search_limits.max_candidates_per_level
        if level_limit is not None and compatible_count > level_limit:
            self._raise_limit(
                "GINSU_MAX_CANDIDATES_PER_LEVEL",
                observed=compatible_count,
                limit=level_limit,
                stage=f"level {level} compatible pairs",
            )
        total = self._generated_candidate_total + compatible_count
        total_limit = self._active_search_limits.max_total_candidates
        if total_limit is not None and total > total_limit:
            self._raise_limit(
                "GINSU_MAX_TOTAL_CANDIDATES",
                observed=total,
                limit=total_limit,
                stage=f"level {level} cumulative compatible pairs",
            )
        self._generated_candidate_total = total

        if compatible_slices.nnz == 0:
            return sp.csr_matrix(np.empty((0, slices.shape[1])))

        (
            pair_candidates,
            slice_sizes,
            slice_errors,
            max_slice_errors,
        ) = self._combine_slices(slices, statistics, compatible_slices)

        (
            pair_candidates,
            slice_sizes,
            slice_errors,
            max_slice_errors,
        ) = self._prune_invalid_self_joins(
            feature_offset_start,
            feature_offset_end,
            pair_candidates,
            slice_sizes,
            slice_errors,
            max_slice_errors,
        )

        if pair_candidates.shape[0] == 0:
            return sp.csr_matrix(np.empty((0, slices.shape[1])))

        ids = self._prepare_deduplication_and_pruning(
            feature_offset_start,
            feature_offset_end,
            feature_domains,
            pair_candidates,
        )

        # remove duplicate candidates and select corresponding statistics
        _, unique_candidate_indices, duplicate_counts = np.unique(
            ids, return_index=True, return_counts=True
        )

        # Slices at level i normally have i parents (cf. section 3.1 in the paper)
        # We want to keep only slices whose parents have not been pruned.
        # If all the parents are present they are going to get combined 2 by 2 in i*(i-1)/2 ways
        # So, we select only candidates which appear with the correct cardinality.
        all_parents_mask = duplicate_counts == level * (level - 1) / 2
        unique_candidate_indices = unique_candidate_indices[all_parents_mask]

        pair_candidates = pair_candidates[unique_candidate_indices]
        slice_sizes = slice_sizes[unique_candidate_indices]
        slice_errors = slice_errors[unique_candidate_indices]
        max_slice_errors = max_slice_errors[unique_candidate_indices]

        slice_scores = self._score_ub(
            slice_sizes,
            slice_errors,
            max_slice_errors,
            n_col_x_encoded,
        )

        # Seems to be always fully True
        # Due to maintain_top_k that apply slice_sizes filter
        pruning_sizes = slice_sizes >= self._min_sup_actual

        _, min_slice_scores = self._analyse_top_k(top_k_statistics)

        pruning_scores = (slice_scores > min_slice_scores) & (slice_scores > 0)

        result = pair_candidates[pruning_scores & pruning_sizes]
        self._last_candidates_after_pruning = result.shape[0]
        return result

    def _search_slices(
        self,
        input_x: NDArray,
        errors: NDArray,
        *,
        started_at: float | None = None,
    ) -> None:
        """Main function of the SliceLine algorithm."""
        self._active_search_limits = self.limits or SearchLimits()
        self._search_started_at = (
            self.clock() if started_at is None else started_at
        )
        if not hasattr(self, "_active_search_stage"):
            self._initialize_search_diagnostics(
                self._search_started_at, stage="one_hot_encoding"
            )
        self._search_levels = []
        self._generated_candidate_total = 0
        self._encoded_feature_count = None
        self._numba_used = False

        # prepare offset vectors and one-hot encoded input_x
        encoder = OneHotEncoder(handle_unknown="ignore")
        self._one_hot_encoder = encoder
        x_encoded = encoder.fit_transform(input_x)
        feature_domains: NDArray = np.array(
            [len(sub_array) for sub_array in encoder.categories_]
        )
        feature_offset_end = np.cumsum(feature_domains)
        feature_offset_start = feature_offset_end - feature_domains

        # initialize statistics and basic slices
        n_col_x_encoded = x_encoded.shape[1]
        self._encoded_feature_count = n_col_x_encoded
        encoded_limit = self._active_search_limits.max_encoded_features
        if encoded_limit is not None and n_col_x_encoded > encoded_limit:
            self._raise_limit(
                "GINSU_MAX_ENCODED_FEATURES",
                observed=n_col_x_encoded,
                limit=encoded_limit,
                stage="one-hot encoding",
            )
        self._transition_search_stage("level_1_evaluation")
        self.average_error_ = float(np.mean(errors))
        slices, statistics = self._create_and_score_basic_slices(
            x_encoded,
            n_col_x_encoded,
            errors,
        )

        # initialize top-k
        top_k_slices, top_k_statistics = self._maintain_top_k(
            slices,
            statistics,
            sp.csr_matrix((0, n_col_x_encoded)),
            np.zeros((0, 4)),
        )
        self._check_tie_limit(top_k_slices.shape[0], "level 1 top-k")
        basic_valid = int(
            np.sum(
                (statistics[:, 3] >= self._min_sup_actual)
                & (statistics[:, 0] > 0)
            )
        )
        self._search_levels.append(
            SearchLevelReport(
                level=1,
                source_slices=n_col_x_encoded,
                potential_pairs=0,
                compatible_pairs=0,
                candidates_after_pruning=slices.shape[0],
                evaluated_candidates=n_col_x_encoded,
                valid_candidates=basic_valid,
            )
        )

        max_slice_scores, min_slice_scores = self._analyse_top_k(
            top_k_statistics
        )
        logger.debug(
            "Initial top-K: count=%i, max=%f, min=%f",
            top_k_slices.shape[0],
            max_slice_scores,
            min_slice_scores,
        )

        # lattice enumeration w/ size/error pruning, one iteration per level
        # termination condition (max #feature levels)
        level = 1
        min_condition = min(input_x.shape[1], self.max_l)
        while (
            (slices.shape[0] > 0)
            and (slices.sum() > 0)
            and (level < min_condition)
        ):
            level += 1
            self._transition_search_stage(f"level_{level}_join")

            # enumerate candidate join pairs, including size/error pruning
            slices, statistics = self._get_pruned_s_r(slices, statistics)
            nr_s = slices.shape[0]
            slices = self._get_pair_candidates(
                slices,
                statistics,
                top_k_statistics,
                level,
                n_col_x_encoded,
                feature_domains,
                feature_offset_start,
                feature_offset_end,
            )
            candidate_count = slices.shape[0]
            self._transition_search_stage(f"level_{level}_evaluation")

            logger.debug("Level %i:", level)
            logger.debug(
                " -- generated paired slice candidates: %i -> %i",
                nr_s,
                slices.shape[0],
            )

            # extract and evaluate candidate slices
            statistics = self._eval_slice(x_encoded, errors, slices, level)

            # maintain top-k after evaluation
            top_k_slices, top_k_statistics = self._maintain_top_k(
                slices, statistics, top_k_slices, top_k_statistics
            )
            self._check_tie_limit(
                top_k_slices.shape[0], f"level {level} top-k"
            )

            max_slice_scores, min_slice_scores = self._analyse_top_k(
                top_k_statistics
            )
            valid = np.sum(
                (statistics[:, 3] >= self._min_sup_actual)
                & (statistics[:, 1] > 0)
            )
            self._search_levels.append(
                SearchLevelReport(
                    level=level,
                    source_slices=nr_s,
                    potential_pairs=self._last_potential_pairs,
                    compatible_pairs=self._last_compatible_pairs,
                    candidates_after_pruning=(
                        self._last_candidates_after_pruning
                    ),
                    evaluated_candidates=candidate_count,
                    valid_candidates=int(valid),
                )
            )
            logger.debug(
                " -- valid slices after eval: %s/%i", valid, slices.shape[0]
            )
            logger.debug(
                " -- top-K: count=%i, max=%f, min=%f",
                top_k_slices.shape[0],
                max_slice_scores,
                min_slice_scores,
            )

        self._top_slices_enc = top_k_slices.copy()
        if top_k_slices.shape[0] == 0:
            self.top_slices_ = np.empty((0, input_x.shape[1]))
        else:
            self.top_slices_ = encoder.inverse_transform(top_k_slices)

        # compute slices' average errors
        top_k_statistics = np.column_stack(
            (
                top_k_statistics,
                np.divide(top_k_statistics[:, 1], top_k_statistics[:, 3]),
            )
        )

        # transform statistics to a list of dict
        statistics_names = [
            "slice_score",
            "sum_slice_error",
            "max_slice_error",
            "slice_size",
            "slice_average_error",
        ]
        self.top_slices_statistics_ = [
            {
                stat_name: float(stat_value)
                for stat_value, stat_name in zip(
                    statistic, statistics_names, strict=False
                )
            }
            for statistic in top_k_statistics
        ]

        logger.debug("Terminated at level %i.", level)
