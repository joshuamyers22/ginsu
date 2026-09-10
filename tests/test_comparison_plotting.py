"""Semantic tests for bounded analysis-comparison visualizations."""

import polars as pl
import pytest

from ginsu import (
    AnalysisLimitError,
    SliceAnalysis,
    Slicefinder,
    compare_analyses,
)
from ginsu._comparison_plot_data import (
    COMPARISON_DUMBBELL_SCHEMA,
    COMPARISON_MIGRATION_SCHEMA,
    comparison_dumbbell_data,
    comparison_migration_data,
)


def _analysis(target: str, *, scale: float = 1.0) -> SliceAnalysis:
    frame = pl.DataFrame(
        {
            "a": [0, 0, 0, 0, 1, 1, 1, 1],
            "b": [0, 0, 1, 1, 0, 0, 1, 1],
        }
    )
    if target == "a":
        errors = [5.0 * scale] * 4 + [1.0] * 4
    elif target == "b":
        errors = [5.0 * scale, 5.0 * scale, 1.0, 1.0] * 2
    else:
        errors = [1.0, 1.0, 5.0 * scale, 5.0 * scale, 1.0, 1.0, 1.0, 1.0]
    finder = Slicefinder(
        alpha=0.95, k=1, max_l=2, min_sup=1, verbose=False
    ).fit(frame, errors)
    return SliceAnalysis.from_finder(
        finder, dataset_fingerprint=f"sha256:{target}-{scale}"
    )


def test_dumbbell_data_keeps_emerged_and_resolved_endpoints():
    comparison = compare_analyses(
        _analysis("a"),
        _analysis("b"),
        method="predicate",
        similarity_threshold=0.8,
    )

    data = comparison_dumbbell_data(comparison)

    assert data.schema == COMPARISON_DUMBBELL_SCHEMA
    assert data.height == 2
    assert set(data["comparison_status"]) == {"emerged", "resolved"}
    assert data["baseline_value"].null_count() == 1
    assert data["candidate_value"].null_count() == 1
    assert any(label.endswith("(emerged)") for label in data["label"])
    assert any(label.endswith("(resolved)") for label in data["label"])


def test_dumbbell_data_selects_metric_and_preserves_delta():
    comparison = compare_analyses(
        _analysis("a"), _analysis("a", scale=1.5), method="exact"
    )

    data = comparison_dumbbell_data(comparison, metric="error_lift")

    assert data["metric"].to_list() == ["error_lift"]
    assert data["delta"][0] == comparison.changes["error_lift_delta"][0]
    assert (
        data["baseline_value"][0]
        == comparison.changes["baseline_error_lift"][0]
    )


def test_migration_data_expands_audited_reference_segments():
    reference = pl.DataFrame(
        {
            "a": [0, 0, 0, 0, 1, 1, 1, 1],
            "b": [0, 0, 1, 1, 0, 0, 1, 1],
        }
    )
    comparison = compare_analyses(
        _analysis("intersection"),
        _analysis("a"),
        method="predicate",
        similarity_threshold=0.5,
        reference_data=reference,
        reference_id="reference-v1",
    )

    data = comparison_migration_data(comparison)

    assert data.schema == COMPARISON_MIGRATION_SCHEMA
    assert data.height == 3
    assert data["membership_segment"].to_list() == [
        "baseline_only",
        "both",
        "candidate_only",
    ]
    assert data["membership_count"].to_list() == [0, 2, 2]
    assert set(data["reference_id"]) == {"reference-v1"}
    assert set(data["reference_jaccard"]) == {0.5}


def test_migration_data_keeps_unmatched_rule_memberships():
    reference = pl.DataFrame(
        {
            "a": [0, 0, 0, 0, 1, 1, 1, 1],
            "b": [0, 0, 1, 1, 0, 0, 1, 1],
        }
    )
    comparison = compare_analyses(
        _analysis("a"),
        _analysis("b"),
        method="predicate",
        similarity_threshold=0.8,
        reference_data=reference,
        reference_id="reference-v1",
    )

    data = comparison_migration_data(comparison)

    assert data["label"].n_unique() == 2
    assert data.height == 6
    resolved = data.filter(pl.col("comparison_status") == "resolved")
    emerged = data.filter(pl.col("comparison_status") == "emerged")
    assert (
        resolved.filter(pl.col("membership_segment") == "candidate_only")[
            "membership_count"
        ][0]
        == 0
    )
    assert (
        emerged.filter(pl.col("membership_segment") == "baseline_only")[
            "membership_count"
        ][0]
        == 0
    )


def test_comparison_plot_data_rejects_missing_evidence_and_excess_rows():
    comparison = compare_analyses(_analysis("a"), _analysis("b"))

    with pytest.raises(ValueError, match="reference evidence"):
        comparison_migration_data(comparison)

    with pytest.raises(AnalysisLimitError) as raised:
        comparison_dumbbell_data(comparison, max_changes=1)
    assert raised.value.code == "GINSU_MAX_COMPARISON_PLOT_CHANGES"

    with pytest.raises(ValueError, match="metric"):
        comparison_dumbbell_data(comparison, metric="unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive integer"):
        comparison_dumbbell_data(comparison, max_changes=0)
    with pytest.raises(ValueError, match="positive integer"):
        comparison_dumbbell_data(comparison, max_changes=True)
    with pytest.raises(TypeError, match="AnalysisComparison"):
        comparison_dumbbell_data(object())  # type: ignore[arg-type]


def test_comparison_figures_retain_semantics():
    pytest.importorskip("plotly")
    from ginsu.plotting import plot_comparison

    unmatched = compare_analyses(
        _analysis("a"),
        _analysis("b"),
        method="predicate",
        similarity_threshold=0.8,
    )
    dumbbell = plot_comparison(unmatched)

    assert dumbbell.layout.title.text == (
        "Ginsu analysis comparison: error_lift"
    )
    assert sum(len(trace.x) for trace in dumbbell.data) == 2
    assert "emerged and resolved" in dumbbell.layout.annotations[-1].text

    reference = pl.DataFrame(
        {
            "a": [0, 0, 0, 0, 1, 1, 1, 1],
            "b": [0, 0, 1, 1, 0, 0, 1, 1],
        }
    )
    compared = compare_analyses(
        _analysis("intersection"),
        _analysis("a"),
        method="predicate",
        similarity_threshold=0.5,
        reference_data=reference,
        reference_id="reference-v1",
    )
    migration = plot_comparison(compared, kind="migration")

    assert (
        migration.layout.title.text == "Ginsu reference membership migration"
    )
    assert migration.layout.barmode == "stack"
    assert [trace.name for trace in migration.data] == [
        "Baseline only",
        "Shared",
        "Candidate only",
    ]
    assert (
        "not an additive attribution" in migration.layout.annotations[-1].text
    )


def test_rank_plot_reverses_metric_axis_and_noncomparable_is_explicit():
    pytest.importorskip("plotly")
    from ginsu.plotting import plot_comparison

    exact = compare_analyses(_analysis("a"), _analysis("a"), method="exact")
    rank = plot_comparison(exact, metric="rank")
    assert rank.layout.xaxis.autorange == "reversed"

    different_frame = pl.DataFrame({"different": [0] * 4 + [1] * 4})
    different_finder = Slicefinder(
        alpha=0.95, k=1, max_l=1, min_sup=1, verbose=False
    ).fit(different_frame, [5.0] * 4 + [1.0] * 4)
    different = SliceAnalysis.from_finder(
        different_finder, dataset_fingerprint="sha256:different"
    )
    incompatible = compare_analyses(_analysis("a"), different)

    figure = plot_comparison(incompatible)
    migration_figure = plot_comparison(incompatible, kind="migration")

    assert (
        figure.layout.title.text == "Ginsu analysis comparison: not comparable"
    )
    assert "incompatible_feature_schema" in figure.layout.annotations[0].text
    assert migration_figure.layout.title.text == (
        "Ginsu analysis comparison: not comparable"
    )


def test_empty_comparison_plots_are_explicit():
    pytest.importorskip("plotly")
    from ginsu.plotting import plot_comparison

    frame = pl.DataFrame({"a": [0, 0, 1, 1]})
    finder = Slicefinder(
        alpha=0.95, k=10, max_l=1, min_sup=1, verbose=False
    ).fit(frame, [1.0, 1.0, 1.0, 1.0])
    analysis = SliceAnalysis.from_finder(
        finder, dataset_fingerprint="sha256:empty"
    )
    comparison = compare_analyses(
        analysis,
        analysis,
        reference_data=frame,
        reference_id="reference-empty",
    )

    dumbbell = plot_comparison(comparison)
    migration = plot_comparison(comparison, kind="migration")

    assert dumbbell.layout.annotations[0].text == "No comparison changes"
    assert migration.layout.annotations[0].text == "No comparison changes"


def test_comparison_plot_rejects_invalid_kind():
    pytest.importorskip("plotly")
    from ginsu.plotting import plot_comparison

    comparison = compare_analyses(_analysis("a"), _analysis("a"))
    with pytest.raises(ValueError, match="kind"):
        plot_comparison(comparison, kind="waterfall")


def test_comparison_plot_checks_limit_before_loading_plotly(monkeypatch):
    import ginsu.plotting as plotting

    comparison = compare_analyses(_analysis("a"), _analysis("b"))
    monkeypatch.setattr(
        plotting,
        "_plotly_graph_objects",
        lambda: pytest.fail("Plotly loaded before the plot-data limit check"),
    )

    with pytest.raises(AnalysisLimitError) as raised:
        plotting.plot_comparison(comparison, max_changes=1)

    assert raised.value.code == "GINSU_MAX_COMPARISON_PLOT_CHANGES"
