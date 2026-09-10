"""Tests for fitted Polars-native discretization."""

import copy

import polars as pl
import pytest
from sklearn.exceptions import NotFittedError

from ginsu import (
    CategoryPolicy,
    DiscretizationPlan,
    EqualWidthBins,
    FixedBins,
    QuantileBins,
)


def test_fixed_bins_use_stable_right_closed_boundaries():
    frame = pl.DataFrame({"age": [10.0, 20.0, 30.0, 40.0]})
    plan = DiscretizationPlan(numeric={"age": FixedBins((20.0, 30.0))})

    transformed = plan.fit_transform(frame)

    assert transformed.get_column("age").to_list() == [0, 0, 1, 2]
    assert transformed.schema["age"] == pl.UInt32


def test_equal_width_boundaries_are_learned_once():
    discovery = pl.DataFrame({"value": [0.0, 10.0, 20.0, 30.0]})
    validation = pl.DataFrame({"value": [-10.0, 5.0, 25.0, 40.0]})
    plan = DiscretizationPlan(numeric={"value": EqualWidthBins(3)})

    plan.fit(discovery)
    transformed = plan.transform(validation)

    assert plan.boundaries_["value"] == pytest.approx((10.0, 20.0))
    assert transformed.get_column("value").to_list() == [0, 0, 2, 2]


def test_quantile_bins_remove_duplicate_extreme_boundaries():
    frame = pl.DataFrame({"value": [0.0, 0.0, 0.0, 1.0]})
    plan = DiscretizationPlan(numeric={"value": QuantileBins(4)})

    transformed = plan.fit_transform(frame)

    assert len(plan.boundaries_["value"]) <= 1
    assert transformed.get_column("value").n_unique() <= 2


def test_constant_numeric_feature_has_one_bin():
    frame = pl.DataFrame({"value": [2, 2, 2]})
    plan = DiscretizationPlan(numeric={"value": EqualWidthBins(5)})

    transformed = plan.fit_transform(frame)

    assert plan.boundaries_["value"] == ()
    assert transformed.get_column("value").to_list() == [0, 0, 0]


def test_category_policy_distinguishes_rare_and_unknown():
    discovery = pl.DataFrame({"city": ["a", "a", "b", "c"]})
    validation = pl.DataFrame({"city": ["a", "b", "new", "c"]})
    plan = DiscretizationPlan(
        categorical={"city": CategoryPolicy(max_categories=1)}
    )

    plan.fit(discovery)
    transformed = plan.transform(validation)

    assert plan.retained_levels_["city"] == ("a",)
    assert transformed.get_column("city").to_list() == [
        "a",
        "__ginsu_other__",
        "__ginsu_unknown__",
        "__ginsu_other__",
    ]


def test_category_ties_have_deterministic_lexical_order():
    frame = pl.DataFrame({"city": ["z", "a", "z", "a"]})
    plan = DiscretizationPlan(
        categorical={"city": CategoryPolicy(max_categories=1)}
    ).fit(frame)

    assert plan.retained_levels_["city"] == ("a",)


def test_fit_is_error_independent_and_metadata_is_polars():
    frame = pl.DataFrame({"age": [10, 20, 30], "city": ["a", "b", "a"]})
    plan = DiscretizationPlan(
        numeric={"age": QuantileBins(2)},
        categorical={"city": CategoryPolicy()},
    ).fit(frame)

    assert plan.metadata_.shape == (2, 4)
    assert plan.metadata_.schema == {
        "feature": pl.String,
        "strategy": pl.String,
        "boundaries_json": pl.String,
        "retained_levels_json": pl.String,
    }


def test_transform_requires_fit_and_exact_schema():
    frame = pl.DataFrame({"value": [1, 2]})
    plan = DiscretizationPlan(numeric={"value": EqualWidthBins(2)})

    with pytest.raises(NotFittedError):
        plan.transform(frame)

    plan.fit(frame)
    with pytest.raises(ValueError, match="schema does not match"):
        plan.transform(frame.cast({"value": pl.Float64}))


@pytest.mark.parametrize(
    "strategy",
    [EqualWidthBins, QuantileBins],
)
def test_bin_count_must_be_at_least_two(strategy):
    with pytest.raises(ValueError, match="at least 2"):
        strategy(1)


def test_fixed_boundaries_must_be_finite_and_increasing():
    with pytest.raises(ValueError, match="strictly increasing"):
        FixedBins((2.0, 1.0))
    with pytest.raises(ValueError, match="finite"):
        FixedBins((float("inf"),))


def test_category_sentinel_collision_is_rejected():
    frame = pl.DataFrame({"city": ["a", "__ginsu_other__"]})
    plan = DiscretizationPlan(categorical={"city": CategoryPolicy()})

    with pytest.raises(ValueError, match="collide"):
        plan.fit(frame)


def test_fitted_plan_spec_round_trip_preserves_transform():
    discovery = pl.DataFrame(
        {
            "fixed": [0.0, 1.0, 2.0, 3.0],
            "equal": [0, 10, 20, 30],
            "quantile": [1, 1, 2, 9],
            "city": ["a", "a", "b", "c"],
        }
    )
    validation = pl.DataFrame(
        {
            "fixed": [-1.0, 2.0, 4.0],
            "equal": [-5, 15, 40],
            "quantile": [0, 2, 10],
            "city": ["a", "b", "new"],
        }
    )
    fitted = DiscretizationPlan(
        numeric={
            "fixed": FixedBins((1.0, 2.0)),
            "equal": EqualWidthBins(3),
            "quantile": QuantileBins(3),
        },
        categorical={"city": CategoryPolicy(max_categories=1)},
    ).fit(discovery)

    restored = DiscretizationPlan.from_spec(fitted.to_spec())

    assert restored.to_spec() == fitted.to_spec()
    assert restored.metadata_.equals(fitted.metadata_)
    assert restored.transform(validation).equals(fitted.transform(validation))


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda spec: spec.update({"unexpected": True}), "keys must be"),
        (lambda spec: spec.update({"version": 2}), "Unsupported"),
        (
            lambda spec: spec["numeric"][0].update(
                {"learned_boundaries": [2.0, 1.0]}
            ),
            "strictly increasing",
        ),
        (
            lambda spec: spec["categorical"][0].update(
                {"retained_levels": ["not-observed"]}
            ),
            "must be observed",
        ),
    ],
)
def test_discretization_spec_fails_closed(mutation, match):
    frame = pl.DataFrame({"value": [0, 1, 2], "city": ["a", "a", "b"]})
    fitted = DiscretizationPlan(
        numeric={"value": EqualWidthBins(2)},
        categorical={"city": CategoryPolicy(max_categories=1)},
    ).fit(frame)
    spec = copy.deepcopy(fitted.to_spec())
    mutation(spec)

    with pytest.raises((TypeError, ValueError), match=match):
        DiscretizationPlan.from_spec(spec)
