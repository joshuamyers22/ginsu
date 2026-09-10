"""Fitted, reproducible Polars-native feature discretization."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from sklearn.exceptions import NotFittedError

from ginsu._frame import normalize_frame, schema_signature, validate_schema


def _validate_n_bins(n_bins: int) -> None:
    if isinstance(n_bins, bool) or n_bins < 2:
        raise ValueError("n_bins must be an integer of at least 2.")


def _required_float(value: Any) -> float:
    if value is None:
        raise ValueError("Cannot learn numeric bins from an empty feature.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("Numeric bin statistics must be finite.")
    return result


def _check_fitted(plan: DiscretizationPlan) -> None:
    required = ("input_schema_", "boundaries_", "retained_levels_")
    if not all(hasattr(plan, name) for name in required):
        raise NotFittedError(
            "This DiscretizationPlan instance is not fitted yet. Call 'fit' first."
        )


@dataclass(frozen=True, slots=True)
class FixedBins:
    """Caller-supplied finite, strictly increasing numeric boundaries."""

    boundaries: tuple[float, ...]

    def __post_init__(self) -> None:
        normalized = tuple(float(value) for value in self.boundaries)
        if not normalized:
            raise ValueError("FixedBins requires at least one boundary.")
        if not all(np.isfinite(normalized)):
            raise ValueError("FixedBins boundaries must be finite.")
        if any(
            left >= right
            for left, right in zip(normalized, normalized[1:], strict=False)
        ):
            raise ValueError(
                "FixedBins boundaries must be strictly increasing."
            )
        object.__setattr__(self, "boundaries", normalized)


@dataclass(frozen=True, slots=True)
class EqualWidthBins:
    """Learn equal-width boundaries from discovery observations."""

    n_bins: int = 10

    def __post_init__(self) -> None:
        _validate_n_bins(self.n_bins)


@dataclass(frozen=True, slots=True)
class QuantileBins:
    """Learn deterministic linear-quantile boundaries from discovery data."""

    n_bins: int = 10

    def __post_init__(self) -> None:
        _validate_n_bins(self.n_bins)


@dataclass(frozen=True, slots=True)
class CategoryPolicy:
    """Retain frequent categories and distinguish rare from unseen values."""

    max_categories: int = 20
    min_count: int = 1
    other_label: str = "__ginsu_other__"
    unknown_label: str = "__ginsu_unknown__"

    def __post_init__(self) -> None:
        if isinstance(self.max_categories, bool) or self.max_categories < 1:
            raise ValueError("max_categories must be a positive integer.")
        if isinstance(self.min_count, bool) or self.min_count < 1:
            raise ValueError("min_count must be a positive integer.")
        if not self.other_label or not self.unknown_label:
            raise ValueError("Category sentinel labels must not be empty.")
        if self.other_label == self.unknown_label:
            raise ValueError("Category sentinel labels must be distinct.")


NumericBins = FixedBins | EqualWidthBins | QuantileBins


class DiscretizationPlan:
    """Fit feature transformations on discovery data and reuse them unchanged."""

    def __init__(
        self,
        *,
        numeric: Mapping[str, NumericBins] | None = None,
        categorical: Mapping[str, CategoryPolicy] | None = None,
    ) -> None:
        self.numeric = dict(numeric or {})
        self.categorical = dict(categorical or {})
        overlap = self.numeric.keys() & self.categorical.keys()
        if overlap:
            raise ValueError(
                "Features cannot be both numeric and categorical: "
                f"{sorted(overlap)!r}."
            )

    def fit(self, X: Any) -> DiscretizationPlan:
        """Learn boundaries and category sets from discovery observations."""
        normalized = normalize_frame(X)
        frame = normalized.frame
        configured = self.numeric.keys() | self.categorical.keys()
        missing = sorted(configured - set(frame.columns))
        if missing:
            raise ValueError(
                f"Configured features are missing from X: {missing!r}."
            )

        self.input_schema_ = schema_signature(frame)
        self.boundaries_: dict[str, tuple[float, ...]] = {}
        self.retained_levels_: dict[str, tuple[str, ...]] = {}
        self.observed_levels_: dict[str, tuple[str, ...]] = {}
        metadata: list[dict[str, Any]] = []

        for feature, strategy in self.numeric.items():
            series = frame.get_column(feature)
            if not series.dtype.is_numeric() or series.dtype == pl.Boolean:
                raise TypeError(
                    f"Numeric feature {feature!r} must have a numeric dtype."
                )
            boundaries = _fit_numeric_boundaries(series, strategy)
            self.boundaries_[feature] = boundaries
            metadata.append(
                {
                    "feature": feature,
                    "strategy": type(strategy).__name__,
                    "boundaries_json": json.dumps(
                        boundaries, separators=(",", ":")
                    ),
                    "retained_levels_json": None,
                }
            )

        for feature, policy in self.categorical.items():
            values = frame.get_column(feature).cast(pl.String).to_list()
            if policy.other_label in values or policy.unknown_label in values:
                raise ValueError(
                    f"Category sentinel labels collide with values in {feature!r}."
                )
            counts = Counter(values)
            ordered = sorted(counts, key=lambda value: (-counts[value], value))
            retained = tuple(
                value for value in ordered if counts[value] >= policy.min_count
            )[: policy.max_categories]
            observed = tuple(sorted(counts))
            self.retained_levels_[feature] = retained
            self.observed_levels_[feature] = observed
            metadata.append(
                {
                    "feature": feature,
                    "strategy": type(policy).__name__,
                    "boundaries_json": None,
                    "retained_levels_json": json.dumps(
                        retained, ensure_ascii=False, separators=(",", ":")
                    ),
                }
            )

        self.metadata_ = pl.DataFrame(
            metadata,
            schema={
                "feature": pl.String,
                "strategy": pl.String,
                "boundaries_json": pl.String,
                "retained_levels_json": pl.String,
            },
        )
        return self

    def transform(self, X: Any) -> pl.DataFrame:
        """Apply only the boundaries and category sets learned by ``fit``."""
        _check_fitted(self)
        normalized = normalize_frame(X)
        validate_schema(normalized.frame, expected=self.input_schema_)
        expressions: list[pl.Expr] = []

        for feature, boundaries in self.boundaries_.items():
            expression = pl.lit(0, dtype=pl.UInt32)
            for boundary in boundaries:
                expression += (
                    pl.col(feature).cast(pl.Float64) > boundary
                ).cast(pl.UInt32)
            expressions.append(expression.alias(feature))

        for feature, policy in self.categorical.items():
            values = pl.col(feature).cast(pl.String)
            retained = self.retained_levels_[feature]
            observed = self.observed_levels_[feature]
            expressions.append(
                pl.when(values.is_in(retained))
                .then(values)
                .when(values.is_in(observed))
                .then(pl.lit(policy.other_label))
                .otherwise(pl.lit(policy.unknown_label))
                .alias(feature)
            )

        return normalized.frame.with_columns(expressions)

    def fit_transform(self, X: Any) -> pl.DataFrame:
        """Fit on discovery data and transform that same data."""
        return self.fit(X).transform(X)

    def to_spec(self) -> dict[str, Any]:
        """Return the fitted plan as closed, JSON-compatible data."""
        _check_fitted(self)
        numeric = []
        for feature, strategy in self.numeric.items():
            if isinstance(strategy, FixedBins):
                strategy_spec: dict[str, Any] = {
                    "name": "fixed",
                    "boundaries": list(strategy.boundaries),
                }
            elif isinstance(strategy, EqualWidthBins):
                strategy_spec = {
                    "name": "equal_width",
                    "n_bins": strategy.n_bins,
                }
            else:
                strategy_spec = {
                    "name": "quantile",
                    "n_bins": strategy.n_bins,
                }
            numeric.append(
                {
                    "feature": feature,
                    "strategy": strategy_spec,
                    "learned_boundaries": list(self.boundaries_[feature]),
                }
            )

        categorical = []
        for feature, policy in self.categorical.items():
            categorical.append(
                {
                    "feature": feature,
                    "policy": {
                        "max_categories": policy.max_categories,
                        "min_count": policy.min_count,
                        "other_label": policy.other_label,
                        "unknown_label": policy.unknown_label,
                    },
                    "retained_levels": list(self.retained_levels_[feature]),
                    "observed_levels": list(self.observed_levels_[feature]),
                }
            )

        return {
            "version": 1,
            "input_schema": [
                {"name": name, "dtype": dtype}
                for name, dtype in self.input_schema_
            ],
            "numeric": numeric,
            "categorical": categorical,
        }

    @classmethod
    def from_spec(cls, spec: Any) -> DiscretizationPlan:
        """Restore a fitted plan from validated declarative data."""
        root = _closed_mapping(
            spec,
            keys={"version", "input_schema", "numeric", "categorical"},
            context="discretization plan",
        )
        if _plain_int(root["version"], "discretization version") != 1:
            raise ValueError("Unsupported discretization plan version.")

        schema_rows = _plain_list(root["input_schema"], "input_schema")
        input_schema = []
        for index, value in enumerate(schema_rows):
            row = _closed_mapping(
                value,
                keys={"name", "dtype"},
                context=f"input_schema[{index}]",
            )
            input_schema.append(
                (
                    _plain_string(row["name"], f"input_schema[{index}].name"),
                    _plain_string(
                        row["dtype"], f"input_schema[{index}].dtype"
                    ),
                )
            )
        if not input_schema or len({name for name, _ in input_schema}) != len(
            input_schema
        ):
            raise ValueError(
                "Discretization input_schema must contain unique features."
            )
        schema_names = {name for name, _ in input_schema}

        numeric_config: dict[str, NumericBins] = {}
        learned_boundaries: dict[str, tuple[float, ...]] = {}
        for index, value in enumerate(_plain_list(root["numeric"], "numeric")):
            row = _closed_mapping(
                value,
                keys={"feature", "strategy", "learned_boundaries"},
                context=f"numeric[{index}]",
            )
            feature = _plain_string(
                row["feature"], f"numeric[{index}].feature"
            )
            if feature not in schema_names or feature in numeric_config:
                raise ValueError(
                    f"Invalid or duplicate numeric feature {feature!r}."
                )
            strategy = _restore_numeric_strategy(
                row["strategy"], context=f"numeric[{index}].strategy"
            )
            boundaries = _finite_float_tuple(
                row["learned_boundaries"],
                context=f"numeric[{index}].learned_boundaries",
                allow_empty=True,
            )
            if (
                isinstance(strategy, FixedBins)
                and boundaries != strategy.boundaries
            ):
                raise ValueError(
                    f"Fixed boundaries do not match learned boundaries for {feature!r}."
                )
            numeric_config[feature] = strategy
            learned_boundaries[feature] = boundaries

        categorical_config: dict[str, CategoryPolicy] = {}
        retained_levels: dict[str, tuple[str, ...]] = {}
        observed_levels: dict[str, tuple[str, ...]] = {}
        for index, value in enumerate(
            _plain_list(root["categorical"], "categorical")
        ):
            row = _closed_mapping(
                value,
                keys={
                    "feature",
                    "policy",
                    "retained_levels",
                    "observed_levels",
                },
                context=f"categorical[{index}]",
            )
            feature = _plain_string(
                row["feature"], f"categorical[{index}].feature"
            )
            if (
                feature not in schema_names
                or feature in categorical_config
                or feature in numeric_config
            ):
                raise ValueError(
                    f"Invalid or duplicate categorical feature {feature!r}."
                )
            policy_data = _closed_mapping(
                row["policy"],
                keys={
                    "max_categories",
                    "min_count",
                    "other_label",
                    "unknown_label",
                },
                context=f"categorical[{index}].policy",
            )
            policy = CategoryPolicy(
                max_categories=_plain_int(
                    policy_data["max_categories"], "max_categories"
                ),
                min_count=_plain_int(policy_data["min_count"], "min_count"),
                other_label=_plain_string(
                    policy_data["other_label"], "other_label"
                ),
                unknown_label=_plain_string(
                    policy_data["unknown_label"], "unknown_label"
                ),
            )
            retained = _string_tuple(
                row["retained_levels"],
                context=f"categorical[{index}].retained_levels",
            )
            observed = _string_tuple(
                row["observed_levels"],
                context=f"categorical[{index}].observed_levels",
            )
            if not set(retained).issubset(observed):
                raise ValueError(
                    f"Retained levels must be observed for {feature!r}."
                )
            if len(retained) > policy.max_categories:
                raise ValueError(
                    f"Retained levels exceed max_categories for {feature!r}."
                )
            if (
                policy.other_label in observed
                or policy.unknown_label in observed
            ):
                raise ValueError(
                    f"Category sentinels collide with observed levels for {feature!r}."
                )
            categorical_config[feature] = policy
            retained_levels[feature] = retained
            observed_levels[feature] = observed

        result = cls(
            numeric=numeric_config,
            categorical=categorical_config,
        )
        result.input_schema_ = tuple(input_schema)
        result.boundaries_ = learned_boundaries
        result.retained_levels_ = retained_levels
        result.observed_levels_ = observed_levels
        result.metadata_ = pl.DataFrame(
            [
                {
                    "feature": feature,
                    "strategy": type(strategy).__name__,
                    "boundaries_json": json.dumps(
                        learned_boundaries[feature], separators=(",", ":")
                    ),
                    "retained_levels_json": None,
                }
                for feature, strategy in numeric_config.items()
            ]
            + [
                {
                    "feature": feature,
                    "strategy": type(policy).__name__,
                    "boundaries_json": None,
                    "retained_levels_json": json.dumps(
                        retained_levels[feature],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                }
                for feature, policy in categorical_config.items()
            ],
            schema={
                "feature": pl.String,
                "strategy": pl.String,
                "boundaries_json": pl.String,
                "retained_levels_json": pl.String,
            },
        )
        return result


def _fit_numeric_boundaries(
    series: pl.Series, strategy: NumericBins
) -> tuple[float, ...]:
    if isinstance(strategy, FixedBins):
        return strategy.boundaries

    values = series.cast(pl.Float64)
    minimum = _required_float(values.min())
    maximum = _required_float(values.max())
    if minimum == maximum:
        return ()

    if isinstance(strategy, EqualWidthBins):
        width = (maximum - minimum) / strategy.n_bins
        candidates = (
            minimum + index * width for index in range(1, strategy.n_bins)
        )
    else:
        candidates = (
            _required_float(
                values.quantile(
                    index / strategy.n_bins, interpolation="linear"
                )
            )
            for index in range(1, strategy.n_bins)
        )

    return tuple(
        sorted(
            {
                boundary
                for boundary in candidates
                if minimum < boundary < maximum and np.isfinite(boundary)
            }
        )
    )


def _closed_mapping(
    value: Any, *, keys: set[str], context: str
) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{context} must be a JSON object.")
    if set(value) != keys:
        raise ValueError(
            f"{context} keys must be {sorted(keys)!r}; received {sorted(value)!r}."
        )
    return value


def _plain_list(value: Any, context: str) -> list[Any]:
    if type(value) is not list:
        raise TypeError(f"{context} must be a JSON array.")
    return value


def _plain_string(value: Any, context: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{context} must be a non-empty string.")
    return value


def _plain_int(value: Any, context: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{context} must be an integer.")
    return value


def _finite_float_tuple(
    value: Any, *, context: str, allow_empty: bool
) -> tuple[float, ...]:
    values = _plain_list(value, context)
    result = tuple(float(item) for item in values)
    if (not allow_empty and not result) or not all(np.isfinite(result)):
        raise ValueError(f"{context} must contain finite boundaries.")
    if any(
        left >= right for left, right in zip(result, result[1:], strict=False)
    ):
        raise ValueError(f"{context} must be strictly increasing.")
    return result


def _string_tuple(value: Any, *, context: str) -> tuple[str, ...]:
    values = _plain_list(value, context)
    result = tuple(_plain_string(item, context) for item in values)
    if len(set(result)) != len(result):
        raise ValueError(f"{context} must not contain duplicate values.")
    return result


def _restore_numeric_strategy(value: Any, *, context: str) -> NumericBins:
    if type(value) is not dict or "name" not in value:
        raise TypeError(f"{context} must be a strategy object.")
    name = _plain_string(value["name"], f"{context}.name")
    if name == "fixed":
        row = _closed_mapping(
            value, keys={"name", "boundaries"}, context=context
        )
        return FixedBins(
            _finite_float_tuple(
                row["boundaries"],
                context=f"{context}.boundaries",
                allow_empty=False,
            )
        )
    if name in ("equal_width", "quantile"):
        row = _closed_mapping(value, keys={"name", "n_bins"}, context=context)
        n_bins = _plain_int(row["n_bins"], f"{context}.n_bins")
        return (
            EqualWidthBins(n_bins)
            if name == "equal_width"
            else QuantileBins(n_bins)
        )
    raise ValueError(f"Unknown numeric strategy {name!r}.")
