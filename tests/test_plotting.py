"""Semantic tests for Ginsu plot data and optional Plotly figures."""

import numpy as np
import polars as pl
import pytest

from ginsu import (
    AnalysisLimitError,
    Slicefinder,
    StabilityRun,
    evaluate_similarity_stability,
    evaluate_stability,
)
from ginsu._plot_data import (
    equivalence_groups,
    error_dependence_data,
    error_dependence_summary,
    impact_plot_data,
    lattice_edges_data,
    overlap_data,
    predicate_matrix_data,
)
from ginsu._stability_plot_data import (
    SENSITIVITY_PLOT_SCHEMA,
    STABILITY_PLOT_SCHEMA,
    sensitivity_plot_data,
    stability_plot_data,
)


@pytest.fixture
def fitted_example():
    frame = pl.DataFrame(
        {
            "age": [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75],
            "region": ["east"] * 6 + ["west"] * 6,
        }
    )
    errors = np.array([4, 3, 4, 3, 4, 3, 1, 1, 1, 1, 1, 1], dtype=float)
    finder = Slicefinder(
        alpha=0.95, k=2, max_l=2, min_sup=2, verbose=False
    ).fit(frame, errors)
    return finder, frame, errors


@pytest.fixture
def overlap_example():
    frame = pl.DataFrame(
        {
            "a": [0, 0, 0, 0, 1, 1, 1, 1],
            "b": [0, 0, 1, 1, 0, 0, 1, 1],
            "c": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    errors = np.array([0, 0, 0, 5, 0, 0, 0, 5], dtype=float)
    finder = Slicefinder(
        alpha=0.95, k=20, max_l=2, min_sup=1, verbose=False
    ).fit(frame, errors)
    return finder, frame


def _stability_finder(region: str) -> Slicefinder:
    frame = pl.DataFrame({"region": ["east"] * 10 + ["west"] * 10})
    errors = (
        [5.0] * 10 + [1.0] * 10
        if region == "east"
        else [1.0] * 10 + [5.0] * 10
    )
    return Slicefinder(alpha=0.95, k=1, max_l=1, min_sup=2, verbose=False).fit(
        frame, errors
    )


@pytest.fixture
def stability_report():
    runs = [
        StabilityRun.from_finder(
            _stability_finder("east"),
            run_id="run-1",
            partition_id="train-1",
            resampling_unit="account",
            parameters={"alpha": 0.8, "mode": "strict"},
        ),
        StabilityRun.from_finder(
            _stability_finder("west"),
            run_id="run-2",
            partition_id="train-2",
            resampling_unit="account",
            parameters={"alpha": 0.9, "mode": "lenient"},
        ),
        StabilityRun.unavailable(
            run_id="run-3",
            status="limit_reached",
            partition_id="train-3",
            resampling_unit="account",
            failure_reason="candidate limit",
            parameters={"alpha": 1.0, "mode": "strict"},
        ),
    ]
    return evaluate_stability(runs), evaluate_similarity_stability(
        runs, method="predicate", similarity_threshold=0.8
    )


def test_impact_data_uses_canonical_metrics(fitted_example):
    finder, _, _ = fitted_example

    data = impact_plot_data(finder)

    assert data.height == finder.slices_.height
    assert data.get_column("__ginsu_id").n_unique() == data.height
    assert data.get_column("support_fraction").is_between(0, 1).all()


def test_predicate_matrix_is_rectangular_and_matches_rule_sizes(
    overlap_example,
):
    finder, _ = overlap_example

    data = predicate_matrix_data(finder)
    used_by_slice = data.group_by("__ginsu_id").agg(
        pl.col("is_used").sum().alias("used")
    )
    expected = finder.slice_statistics_.select(
        "__ginsu_id", pl.col("predicate_count").alias("used")
    )

    assert data.height == finder.slices_.height * len(finder._feature_schema)
    assert (
        used_by_slice.join(
            expected,
            on=("__ginsu_id", "used"),
            how="inner",
        ).height
        == finder.slices_.height
    )


def test_predicate_matrix_limit_fails_before_cross_join(overlap_example):
    finder, _ = overlap_example

    with pytest.raises(AnalysisLimitError) as raised:
        predicate_matrix_data(finder, max_cells=1)

    assert raised.value.code == "GINSU_MAX_PREDICATE_MATRIX_CELLS"


def test_dependence_data_preserves_all_rows_and_membership(fitted_example):
    finder, frame, errors = fitted_example

    data = error_dependence_data(finder, frame, errors, feature="age")
    summary = error_dependence_summary(data, n_bins=4)

    assert data.height == frame.height
    assert data.get_column("error").to_list() == errors.tolist()
    assert summary.get_column("count").sum() == frame.height
    assert set(data.get_column("in_slice").unique()) <= {True, False}


def test_dependence_data_accepts_an_all_zero_evaluation_loss(fitted_example):
    finder, frame, _ = fitted_example

    data = error_dependence_data(
        finder, frame, np.zeros(frame.height), feature="age"
    )

    assert data.get_column("error").sum() == 0


def test_dependence_plot_is_explicitly_observational(fitted_example):
    pytest.importorskip("plotly")
    from ginsu.plotting import plot_error_dependence

    finder, frame, errors = fitted_example
    figure = plot_error_dependence(
        finder, frame, errors, feature="age", max_points=5, seed=7
    )

    assert figure.layout.title.text == "Observed error dependence: age"
    assert "not causal" in figure.layout.annotations[0].text
    assert len(figure.data) == 4


def test_impact_figure_has_population_reference(fitted_example):
    pytest.importorskip("plotly")
    from ginsu.plotting import plot_impact

    finder, _, _ = fitted_example
    figure = plot_impact(finder)

    assert figure.layout.title.text == "Ginsu slice impact"
    assert len(figure.layout.shapes) == 1


def test_overlap_data_is_symmetric_and_has_unit_diagonal(overlap_example):
    finder, frame = overlap_example

    data = overlap_data(finder, frame)
    count = finder.slices_.height
    matrix = data.get_column("jaccard").to_numpy().reshape(count, count)

    assert data.height == count**2
    np.testing.assert_allclose(matrix, matrix.T)
    np.testing.assert_allclose(np.diag(matrix), np.ones(count))
    assert data.filter(pl.col("equivalent")).height >= count


def test_equivalence_groups_are_exact_and_stable() -> None:
    frame = pl.DataFrame({"a": [0, 0, 1, 1], "b": [0, 0, 1, 1]})
    finder = Slicefinder(
        alpha=1.0, k=10, max_l=2, min_sup=1, verbose=False
    ).fit(frame, [0.0, 0.0, 1.0, 1.0])

    first = equivalence_groups(finder, frame)
    second = finder.equivalence_groups(frame)

    assert first.equals(second)
    assert first.get_column("represented_rule_count").max() >= 2
    assert first.get_column("support_count").min() > 0


def test_zero_union_jaccard_is_null(monkeypatch, overlap_example):
    finder, frame = overlap_example
    count = finder.slices_.height
    monkeypatch.setattr(
        finder,
        "_get_slices_masks",
        lambda array: np.zeros((count, array.shape[0]), dtype=int),
    )

    data = finder.overlap_frame(frame)

    assert data.get_column("jaccard").null_count() == count**2
    assert data.get_column("union_count").sum() == 0


def test_overlap_limits_fail_before_membership_materialization(
    monkeypatch, overlap_example
):
    finder, frame = overlap_example
    monkeypatch.setattr(
        finder,
        "membership_frame",
        lambda *_args, **_kwargs: pytest.fail("membership was materialized"),
    )

    with pytest.raises(AnalysisLimitError) as raised:
        finder.overlap_frame(frame, max_slices=1)

    assert raised.value.code == "GINSU_MAX_ANALYSIS_SLICES"


def test_lattice_edges_are_exact_one_predicate_refinements(overlap_example):
    finder, _ = overlap_example

    edges = lattice_edges_data(finder)

    assert edges.height > 0
    assert (
        (
            edges.get_column("child_predicate_count")
            - edges.get_column("parent_predicate_count")
        )
        .eq(1)
        .all()
    )
    assert finder.lattice_edges().equals(edges)


def test_lattice_limit_fails_before_pairwise_work(overlap_example):
    finder, _ = overlap_example

    with pytest.raises(AnalysisLimitError) as raised:
        finder.lattice_edges(max_nodes=1)

    assert raised.value.code == "GINSU_MAX_LATTICE_NODES"


def test_overlap_and_lattice_figures_have_semantic_labels(overlap_example):
    pytest.importorskip("plotly")
    from ginsu.plotting import plot_lattice, plot_overlap

    finder, frame = overlap_example
    overlap_figure = plot_overlap(finder, frame)
    lattice_figure = plot_lattice(finder)

    assert overlap_figure.layout.title.text == "Ginsu slice overlap"
    assert overlap_figure.data[0].colorbar.title.text == "Jaccard"
    assert lattice_figure.layout.title.text == "Ginsu slice lattice"
    assert "not causality" in lattice_figure.layout.annotations[0].text


def test_predicate_matrix_figure_labels_composition(overlap_example):
    pytest.importorskip("plotly")
    from ginsu.plotting import plot_predicate_matrix

    finder, _ = overlap_example
    figure = plot_predicate_matrix(finder)

    assert figure.layout.title.text == "Ginsu predicate matrix"
    assert figure.layout.xaxis.title.text == "Feature"


def test_stability_plot_data_normalizes_exact_and_similarity_reports(
    stability_report,
):
    exact, similarity = stability_report

    exact_data = stability_plot_data(exact, metric="rank")
    similar_data = stability_plot_data(similarity, metric="similarity")

    assert exact_data.schema == STABILITY_PLOT_SCHEMA
    assert similar_data.schema == STABILITY_PLOT_SCHEMA
    assert exact_data.height == exact.summary.height * exact.runs.height
    assert similar_data.height == (
        similarity.summary.height * similarity.exact.runs.height
    )
    assert set(exact_data["report_kind"]) == {"exact"}
    assert set(similar_data["report_kind"]) == {"similarity"}
    assert set(similar_data["similarity_method"]) == {"predicate"}
    unavailable = exact_data.filter(pl.col("run_id") == "run-3")
    assert unavailable["evidence_available"].to_list() == [False, False]
    assert unavailable["observation_value"].null_count() == 2


def test_sensitivity_data_preserves_numeric_parameter_and_absence(
    stability_report,
):
    exact, _ = stability_report

    data = sensitivity_plot_data(exact, parameter="alpha", metric="selected")

    assert data.schema == SENSITIVITY_PLOT_SCHEMA
    assert set(data["parameter_kind"]) == {"float"}
    assert data["parameter_is_numeric"].all()
    assert set(data["parameter_numeric"].drop_nulls()) == {0.8, 0.9, 1.0}
    assert set(data["observation_value"].drop_nulls()) == {0.0, 1.0}
    assert (
        data.filter(pl.col("run_id") == "run-3")[
            "observation_value"
        ].null_count()
        == 2
    )


def test_categorical_sensitivity_is_not_treated_as_numeric(stability_report):
    exact, _ = stability_report

    data = sensitivity_plot_data(exact, parameter="mode", metric="rank")

    assert not data["parameter_is_numeric"].any()
    assert set(data["parameter_value"]) == {"strict", "lenient"}
    assert data["parameter_numeric"].null_count() == data.height


def test_stability_plot_marks_unavailable_runs(stability_report):
    pytest.importorskip("plotly")
    from ginsu.plotting import plot_stability

    exact, similarity = stability_report
    exact_figure = plot_stability(exact)
    similarity_figure = plot_stability(similarity, metric="similarity")

    assert exact_figure.layout.title.text == "Ginsu exact-rule stability"
    assert similarity_figure.layout.title.text == (
        "Ginsu predicate-similarity stability"
    )
    assert "unavailable" in exact_figure.layout.annotations[-1].text
    assert -1 in np.asarray(exact_figure.data[-1].z)
    assert exact_figure.data[-1].colorbar.title.text == "Run evidence"


def test_sensitivity_plot_uses_unconnected_markers_and_evidence_panel(
    stability_report,
):
    pytest.importorskip("plotly")
    from ginsu.plotting import plot_sensitivity

    exact, _ = stability_report
    figure = plot_sensitivity(exact, parameter="mode", metric="rank")

    assert figure.layout.title.text == "Ginsu sensitivity: mode vs. rank"
    assert all(trace.mode == "markers" for trace in figure.data[:-1])
    assert -1 in np.asarray(figure.data[-1].z)
    assert "not a causal" in figure.layout.annotations[-1].text


def test_stability_plot_data_limits_and_invalid_options(stability_report):
    exact, _ = stability_report

    with pytest.raises(AnalysisLimitError) as raised:
        stability_plot_data(exact, max_slices=1)
    assert raised.value.code == "GINSU_MAX_STABILITY_PLOT_SLICES"

    with pytest.raises(AnalysisLimitError) as raised:
        stability_plot_data(exact, max_cells=1)
    assert raised.value.code == "GINSU_MAX_STABILITY_PLOT_CELLS"

    with pytest.raises(ValueError, match="SimilarityStabilityReport"):
        stability_plot_data(exact, metric="similarity")

    with pytest.raises(ValueError, match="missing"):
        sensitivity_plot_data(exact, parameter="unknown")
