"""Tests for descriptive evaluation of fixed slices on holdout data."""

from copy import deepcopy

import numpy as np
import polars as pl
import pytest

from ginsu import (
    AnalysisLimitError,
    Slicefinder,
    ValidationInference,
    ValidationLimits,
    validate_slices,
)
from ginsu.validation import VALIDATION_STATISTICS_SCHEMA, _holm_adjust


@pytest.fixture
def validation_example():
    discovery = pl.DataFrame(
        {
            "region": ["east"] * 4 + ["west"] * 4,
            "tier": [1, 2, 1, 2, 1, 2, 1, 2],
        }
    )
    finder = Slicefinder(
        alpha=0.95,
        k=4,
        max_l=2,
        min_sup=1,
        verbose=False,
    ).fit(discovery, [4.0, 3.0, 5.0, 4.0, 1.0, 1.0, 1.0, 1.0])
    holdout = pl.DataFrame(
        {
            "region": ["east", "east", "west", "west", "north"],
            "tier": [1, 2, 1, 2, 3],
        }
    )
    errors = np.array([5.0, 3.0, 1.0, 1.0, 2.0])
    return finder, holdout, errors


def test_holdout_validation_preserves_rules_and_discovery_order(
    validation_example,
):
    finder, holdout, errors = validation_example
    slices_before = finder.slices_.clone()
    statistics_before = finder.slice_statistics_.clone()
    legacy_slices_before = finder.top_slices_.copy()
    legacy_statistics_before = deepcopy(finder.top_slices_statistics_)

    result = finder.validate_slices(holdout, errors, min_support=1)

    assert result.statistics.get_column("__ginsu_id").to_list() == (
        finder.slices_.get_column("__ginsu_id").to_list()
    )
    assert result.statistics.get_column("discovery_rank").to_list() == list(
        range(1, finder.slices_.height + 1)
    )
    assert finder.slices_.equals(slices_before)
    assert finder.slice_statistics_.equals(statistics_before)
    np.testing.assert_array_equal(finder.top_slices_, legacy_slices_before)
    assert finder.top_slices_statistics_ == legacy_statistics_before


def test_validation_metrics_match_fixed_membership(validation_example):
    finder, holdout, errors = validation_example

    result = validate_slices(finder, holdout, errors, min_support=1)
    membership = finder.membership_frame(holdout)
    baseline = float(errors.mean())

    assert result.row_count == holdout.height
    assert result.baseline_error_mean == pytest.approx(baseline)
    assert result.evidence_kind == "holdout_descriptive"
    assert result.interval_method == "none"
    assert result.test_method == "none"
    assert result.multiplicity_method == "none"
    assert result.statistics.schema == VALIDATION_STATISTICS_SCHEMA
    for row in result.statistics.iter_rows(named=True):
        mask = membership.get_column(row["__ginsu_id"]).to_numpy()
        selected = errors[mask]
        assert row["validation_support_count"] == selected.size
        assert row["validation_support_fraction"] == pytest.approx(
            selected.size / errors.size
        )
        assert row["validation_error_sum"] == pytest.approx(selected.sum())
        assert row["validation_error_mean"] == pytest.approx(selected.mean())
        assert row["validation_error_lift"] == pytest.approx(
            selected.mean() / baseline
        )
        assert row["validation_excess_error"] == pytest.approx(
            selected.sum() - selected.size * baseline
        )


def test_fractional_validation_support_rounds_up(validation_example):
    finder, holdout, errors = validation_example

    result = finder.validate_slices(holdout, errors, min_support=0.41)

    assert result.minimum_support_count == 3
    assert result.statistics.get_column(
        "validation_min_support_count"
    ).unique().to_list() == [3]
    expected_sufficient = result.statistics.filter(
        pl.col("validation_support_count") >= 3
    ).height
    assert result.sufficient_support_count == expected_sufficient


def test_no_members_are_explicit_and_metrics_are_null(validation_example):
    finder, _, _ = validation_example
    unseen = pl.DataFrame({"region": ["north", "north"], "tier": [99, 99]})

    result = finder.validate_slices(unseen, [1.0, 2.0], min_support=1)

    assert set(result.statistics["validation_status"].unique()) == {
        "no_members"
    }
    assert result.statistics["validation_error_mean"].null_count() == (
        finder.slices_.height
    )
    assert result.statistics["validation_error_max"].null_count() == (
        finder.slices_.height
    )
    assert "GINSU_VALIDATION_NO_MEMBERS" in result.warning_codes


def test_insufficient_support_is_not_silently_dropped(validation_example):
    finder, holdout, errors = validation_example

    result = finder.validate_slices(
        holdout, errors, min_support=holdout.height
    )

    assert result.slice_count == finder.slices_.height
    assert result.sufficient_support_count == 0
    assert "insufficient_support" in set(
        result.statistics["validation_status"]
    )
    assert "GINSU_VALIDATION_INSUFFICIENT_SUPPORT" in result.warning_codes


def test_zero_baseline_has_null_lift_without_invalid_arithmetic(
    validation_example,
):
    finder, holdout, _ = validation_example

    result = finder.validate_slices(holdout, np.zeros(holdout.height))

    assert result.baseline_error_mean == 0
    assert result.statistics["validation_error_lift"].null_count() == (
        finder.slices_.height
    )
    assert result.statistics["error_lift_delta"].null_count() == (
        finder.slices_.height
    )
    assert "GINSU_VALIDATION_ZERO_BASELINE" in result.warning_codes


@pytest.mark.parametrize(
    "errors, error_type, message",
    [
        ([1.0], ValueError, "inconsistent row counts"),
        ([1.0, 1.0, -1.0, 1.0, 1.0], ValueError, "nonnegative"),
        ([1.0, 1.0, np.nan, 1.0, 1.0], ValueError, "finite"),
        (["bad"] * 5, TypeError, "numeric or Boolean"),
    ],
)
def test_validation_rejects_invalid_loss_vectors(
    validation_example, errors, error_type, message
):
    finder, holdout, _ = validation_example

    with pytest.raises(error_type, match=message):
        finder.validate_slices(holdout, errors)


def test_validation_rejects_schema_drift(validation_example):
    finder, holdout, errors = validation_example

    with pytest.raises(ValueError, match="schema does not match"):
        finder.validate_slices(holdout.select("tier", "region"), errors)


def test_limits_fail_before_membership_materialization(
    monkeypatch, validation_example
):
    finder, holdout, errors = validation_example
    monkeypatch.setattr(
        finder,
        "membership_frame",
        lambda *_args, **_kwargs: pytest.fail("membership was materialized"),
    )

    with pytest.raises(AnalysisLimitError) as raised:
        finder.validate_slices(
            holdout,
            errors,
            limits=ValidationLimits(max_slices=1),
        )

    assert raised.value.code == "GINSU_MAX_VALIDATION_SLICES"

    with pytest.raises(AnalysisLimitError) as raised:
        finder.validate_slices(
            holdout,
            errors,
            limits=ValidationLimits(max_membership_cells=1),
        )

    assert raised.value.code == "GINSU_MAX_VALIDATION_MEMBERSHIP_CELLS"


def test_empty_discovery_result_keeps_validation_schema():
    discovery = pl.DataFrame({"region": ["east", "west"]})
    finder = Slicefinder(min_sup=3, verbose=False).fit(discovery, [1.0, 1.0])

    result = finder.validate_slices(discovery, [0.0, 0.0])

    assert result.slice_count == 0
    assert result.sufficient_support_count == 0
    assert result.statistics.schema == VALIDATION_STATISTICS_SCHEMA


@pytest.mark.parametrize("min_support", [-1, 1.1, float("nan"), True, "one"])
def test_invalid_validation_support_is_rejected(
    validation_example, min_support
):
    finder, holdout, errors = validation_example

    with pytest.raises((TypeError, ValueError), match="min_support"):
        finder.validate_slices(holdout, errors, min_support=min_support)


def test_arrow_holdout_uses_same_polars_contract(validation_example):
    pytest.importorskip("pyarrow")
    finder, holdout, errors = validation_example

    polars_result = finder.validate_slices(holdout, errors)
    arrow_result = finder.validate_slices(holdout.to_arrow(), errors)

    assert arrow_result.input_kind == "arrow"
    assert arrow_result.statistics.equals(polars_result.statistics)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_slices": 0},
        {"max_slices": True},
        {"max_slices": 1.5},
        {"max_membership_cells": -1},
    ],
)
def test_invalid_validation_limits_are_rejected(kwargs):
    with pytest.raises(ValueError, match="positive integer"):
        ValidationLimits(**kwargs)


def test_validation_limits_are_immutable_value_objects():
    limits = ValidationLimits()
    assert limits.max_slices > 0
    with pytest.raises(AttributeError):
        limits.max_slices = 2


def test_validation_rejects_an_untyped_limits_object(validation_example):
    finder, holdout, errors = validation_example

    with pytest.raises(TypeError, match="ValidationLimits"):
        finder.validate_slices(holdout, errors, limits=object())


def test_inference_detects_a_strong_fixed_holdout_signal():
    discovery = pl.DataFrame({"region": ["east"] * 30 + ["west"] * 30})
    discovery_errors = [5.0, 6.0, 4.0] * 10 + [1.0] * 30
    finder = Slicefinder(
        alpha=0.95,
        k=1,
        max_l=1,
        min_sup=5,
        verbose=False,
    ).fit(discovery, discovery_errors)
    holdout = discovery.clone()
    holdout_errors = np.array([5.0, 6.0, 4.0] * 10 + [1.0] * 30)

    result = finder.validate_slices(
        holdout,
        holdout_errors,
        inference=ValidationInference(
            bootstrap_resamples=499,
            permutation_resamples=499,
            random_seed=7,
        ),
    )
    row = result.statistics.row(0, named=True)

    assert result.evidence_kind == "holdout_inferential"
    assert result.interval_method == "bootstrap_percentile"
    assert result.test_method == "permutation_mean_difference_greater"
    assert result.multiplicity_method == "holm"
    assert result.confidence_level == pytest.approx(0.95)
    assert row["inference_status"] == "tested"
    assert row["validation_error_mean_ci_lower"] <= 5.0
    assert row["validation_error_mean_ci_upper"] >= 5.0
    assert row["validation_mean_difference"] == pytest.approx(4.0)
    assert row["validation_adjusted_p_value"] <= 0.05
    assert row["validation_significant"] is True


def test_inference_is_deterministic_for_a_recorded_seed(validation_example):
    finder, holdout, errors = validation_example
    config = ValidationInference(
        bootstrap_resamples=199,
        permutation_resamples=199,
        random_seed=42,
    )

    first = finder.validate_slices(holdout, errors, inference=config)
    second = finder.validate_slices(holdout, errors, inference=config)

    assert first.statistics.equals(second.statistics)
    assert first.random_seed == second.random_seed == 42


def test_constant_losses_produce_valid_nonsignificant_inference():
    frame = pl.DataFrame({"region": ["east"] * 10 + ["west"] * 10})
    finder = Slicefinder(
        alpha=0.95,
        k=1,
        max_l=1,
        min_sup=2,
        verbose=False,
    ).fit(frame, [2.0] * 10 + [1.0] * 10)

    result = finder.validate_slices(
        frame,
        [1.0] * frame.height,
        inference=ValidationInference(
            bootstrap_resamples=199,
            permutation_resamples=199,
        ),
    )
    row = result.statistics.row(0, named=True)

    assert row["validation_permutation_p_value"] == 1.0
    assert row["validation_adjusted_p_value"] == 1.0
    assert row["validation_significant"] is False
    assert row["validation_error_mean_ci_lower"] == 1.0
    assert row["validation_error_mean_ci_upper"] == 1.0


def test_inference_retains_rules_that_cannot_be_tested(validation_example):
    finder, _, _ = validation_example
    all_east = pl.DataFrame({"region": ["east"] * 4, "tier": [1, 2, 1, 2]})

    result = finder.validate_slices(
        all_east,
        [2.0, 3.0, 2.0, 3.0],
        inference=ValidationInference(
            bootstrap_resamples=100,
            permutation_resamples=100,
        ),
    )

    not_tested = result.statistics.filter(
        pl.col("inference_status") != "tested"
    )
    assert not_tested.height > 0
    assert not_tested["validation_adjusted_p_value"].null_count() == (
        not_tested.height
    )
    assert "GINSU_VALIDATION_INFERENCE_NOT_TESTED" in result.warning_codes


def test_inference_work_limit_fails_before_membership(
    monkeypatch, validation_example
):
    finder, holdout, errors = validation_example
    monkeypatch.setattr(
        finder,
        "membership_frame",
        lambda *_args, **_kwargs: pytest.fail("membership was materialized"),
    )

    with pytest.raises(AnalysisLimitError) as raised:
        finder.validate_slices(
            holdout,
            errors,
            inference=ValidationInference(
                bootstrap_resamples=100,
                permutation_resamples=100,
            ),
            limits=ValidationLimits(max_resample_work=1),
        )

    assert raised.value.code == "GINSU_MAX_VALIDATION_RESAMPLE_WORK"

    with pytest.raises(AnalysisLimitError) as raised:
        finder.validate_slices(
            holdout,
            errors,
            inference=ValidationInference(
                bootstrap_resamples=100,
                permutation_resamples=100,
            ),
            limits=ValidationLimits(max_resample_batch_cells=1),
        )

    assert raised.value.code == ("GINSU_MAX_VALIDATION_RESAMPLE_BATCH_CELLS")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"confidence_level": 0},
        {"confidence_level": 1},
        {"confidence_level": float("nan")},
        {"bootstrap_resamples": 99},
        {"permutation_resamples": True},
        {"random_seed": -1},
        {"multiplicity_method": "none"},
    ],
)
def test_invalid_inference_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ValidationInference(**kwargs)


def test_validation_rejects_an_untyped_inference_object(validation_example):
    finder, holdout, errors = validation_example

    with pytest.raises(TypeError, match="ValidationInference"):
        finder.validate_slices(holdout, errors, inference=object())


def test_holm_adjustment_is_monotone_in_ranked_p_values():
    raw = np.array([0.01, 0.04, 0.03])

    adjusted = _holm_adjust(raw)

    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])
    assert np.all(adjusted >= raw)
