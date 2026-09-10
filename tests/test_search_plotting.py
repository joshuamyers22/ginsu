"""Semantic tests for bounded search-profile visualizations."""

from dataclasses import replace

import polars as pl
import pytest

from ginsu import (
    AnalysisLimitError,
    SearchLimitError,
    SearchLimits,
    Slicefinder,
)
from ginsu._search_plot_data import (
    SEARCH_CARDINALITY_SCHEMA,
    SEARCH_FUNNEL_SCHEMA,
    SEARCH_SUMMARY_SCHEMA,
    search_cardinality_data,
    search_funnel_data,
    search_summary_data,
)


@pytest.fixture
def search_report():
    frame = pl.DataFrame(
        {
            "segment": ["a", "a", "b", "b"],
            "region": [1, 2, 1, 2],
        }
    )
    finder = Slicefinder(
        alpha=1.0, k=2, max_l=2, min_sup=1, verbose=False
    ).fit(frame, [1.0, 1.0, 0.1, 0.1])
    return finder.search_report_


def test_search_funnel_data_preserves_completed_level_evidence(search_report):
    data = search_funnel_data(search_report)

    assert data.schema == SEARCH_FUNNEL_SCHEMA
    assert data.height == len(search_report.levels) * 6
    assert data["stage"].unique(maintain_order=True).to_list() == [
        "Source slices",
        "Potential pairs",
        "Compatible pairs",
        "After pruning",
        "Evaluated candidates",
        "Valid candidates",
    ]
    level_one_pairs = data.filter(
        (pl.col("level") == 1)
        & pl.col("stage").is_in(["Potential pairs", "Compatible pairs"])
    )
    assert level_one_pairs["count"].null_count() == 2
    level_two = data.filter(pl.col("level") == 2)
    expected = search_report.levels[1]
    assert (
        level_two.filter(pl.col("stage") == "Potential pairs")["count"][0]
        == expected.potential_pairs
    )
    assert level_two["is_last_completed_level"].all()


def test_search_cardinality_and_summary_are_typed(search_report):
    cardinality = search_cardinality_data(search_report)
    summary = search_summary_data(search_report)

    assert cardinality.schema == SEARCH_CARDINALITY_SCHEMA
    assert cardinality["feature"].to_list() == ["segment", "region"]
    assert cardinality["cardinality"].to_list() == [2, 2]
    assert cardinality["dtype"].to_list() == ["String", "Int64"]
    assert not cardinality["over_limit"].any()
    assert summary.schema == SEARCH_SUMMARY_SCHEMA
    assert summary.height == 1
    assert summary["status"][0] == "complete"
    assert summary["exhaustive"][0]
    assert summary["completed_level_count"][0] == 2
    assert summary["last_completed_level"][0] == 2
    assert summary["copy_boundaries"][0].to_list() == [
        "polars->numpy",
        "numpy->scipy-csr",
    ]


def test_search_plot_limits_fail_before_expansion(search_report):
    with pytest.raises(AnalysisLimitError) as raised:
        search_funnel_data(search_report, max_levels=1)
    assert raised.value.code == "GINSU_MAX_SEARCH_PLOT_LEVELS"

    with pytest.raises(AnalysisLimitError) as raised:
        search_funnel_data(search_report, max_cells=1)
    assert raised.value.code == "GINSU_MAX_SEARCH_PLOT_CELLS"

    with pytest.raises(AnalysisLimitError) as raised:
        search_cardinality_data(search_report, max_features=1)
    assert raised.value.code == "GINSU_MAX_SEARCH_PLOT_FEATURES"


def test_search_plot_data_rejects_invalid_inputs(search_report):
    with pytest.raises(TypeError, match="SearchReport"):
        search_summary_data(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive integer"):
        search_funnel_data(search_report, max_levels=0)
    with pytest.raises(ValueError, match="positive integer"):
        search_cardinality_data(search_report, max_features=True)
    malformed = replace(search_report, feature_count=3)
    with pytest.raises(ValueError, match="feature_count"):
        search_summary_data(malformed)


def test_search_plot_data_rejects_inconsistent_report_evidence(search_report):
    level = search_report.levels[1]
    duplicate_schema = (("duplicate", "Int64"), ("duplicate", "Int64"))
    invalid_reports = [
        (
            replace(search_report, limits=object()),
            TypeError,
            "SearchLimits",
        ),
        (replace(search_report, status="unknown"), ValueError, "status"),
        (replace(search_report, backend="gpu"), ValueError, "backend"),
        (replace(search_report, row_count=-1), ValueError, "nonnegative"),
        (
            replace(search_report, elapsed_seconds=float("nan")),
            ValueError,
            "finite",
        ),
        (
            replace(
                search_report,
                feature_cardinalities=tuple(
                    reversed(search_report.feature_cardinalities)
                ),
            ),
            ValueError,
            "input_schema order",
        ),
        (
            replace(
                search_report,
                input_schema=duplicate_schema,
                feature_cardinalities=(("duplicate", 1), ("duplicate", 1)),
            ),
            ValueError,
            "unique",
        ),
        (
            replace(
                search_report,
                feature_cardinalities=(("segment", -1), ("region", 2)),
            ),
            ValueError,
            "cardinalities",
        ),
        (
            replace(search_report, levels=(object(),)),
            TypeError,
            "SearchLevelReport",
        ),
        (
            replace(
                search_report,
                levels=(replace(level, source_slices=True),),
            ),
            ValueError,
            "integers",
        ),
        (
            replace(search_report, levels=(level, level)),
            ValueError,
            "increase",
        ),
        (
            replace(
                search_report,
                levels=(
                    replace(
                        level,
                        potential_pairs=1,
                        compatible_pairs=2,
                    ),
                ),
            ),
            ValueError,
            "potential pairs",
        ),
        (
            replace(
                search_report,
                levels=(
                    replace(
                        level,
                        compatible_pairs=1,
                        candidates_after_pruning=2,
                    ),
                ),
            ),
            ValueError,
            "compatible pairs",
        ),
        (
            replace(
                search_report,
                levels=(
                    replace(
                        level,
                        candidates_after_pruning=1,
                        evaluated_candidates=2,
                    ),
                ),
            ),
            ValueError,
            "post-pruning",
        ),
        (
            replace(
                search_report,
                levels=(
                    replace(
                        level,
                        candidates_after_pruning=1,
                        evaluated_candidates=1,
                        valid_candidates=2,
                    ),
                ),
            ),
            ValueError,
            "evaluated",
        ),
    ]

    for invalid, error_type, message in invalid_reports:
        with pytest.raises(error_type, match=message):
            search_summary_data(invalid)  # type: ignore[arg-type]


def test_search_profile_figure_has_diagnostic_semantics(search_report):
    pytest.importorskip("plotly")
    from ginsu.plotting import plot_search_report

    figure = plot_search_report(search_report)

    assert figure.layout.title.text == "Ginsu search profile: complete"
    assert figure.layout.meta["status"] == "complete"
    assert figure.layout.xaxis.title.text == "Lattice level"
    assert figure.layout.xaxis2.title.text == "Distinct values"
    assert {trace.name for trace in figure.data} >= {
        "Source slices",
        "Potential pairs",
        "Compatible pairs",
        "After pruning",
        "Evaluated candidates",
        "Valid candidates",
        "Cardinality",
    }
    annotation_text = " ".join(
        annotation.text for annotation in figure.layout.annotations
    )
    assert "stage timing and peak memory were not recorded" in annotation_text
    assert "not model-quality evidence" in annotation_text


def test_limit_terminated_profile_keeps_reason_and_cardinality_threshold():
    pytest.importorskip("plotly")
    from ginsu.plotting import plot_search_report

    finder = Slicefinder(
        limits=SearchLimits(max_feature_cardinality=1), verbose=False
    )
    with pytest.raises(SearchLimitError):
        finder.fit(pl.DataFrame({"feature": [1, 2, 3]}), [1.0, 1.0, 1.0])

    cardinality = search_cardinality_data(finder.search_report_)
    figure = plot_search_report(finder.search_report_)

    assert cardinality["over_limit"].to_list() == [True]
    assert figure.layout.title.text == "Ginsu search profile: limit_reached"
    assert len(figure.layout.shapes) == 1
    annotation_text = " ".join(
        annotation.text for annotation in figure.layout.annotations
    )
    assert "No completed lattice levels" in annotation_text
    assert "GINSU_MAX_FEATURE_CARDINALITY" in annotation_text


def test_search_profile_checks_bounds_before_loading_plotly(
    monkeypatch, search_report
):
    import ginsu.plotting as plotting

    monkeypatch.setattr(
        plotting,
        "_plotly_graph_objects",
        lambda: pytest.fail("Plotly loaded before the plot-data bound check"),
    )

    with pytest.raises(AnalysisLimitError) as raised:
        plotting.plot_search_report(search_report, max_levels=1)

    assert raised.value.code == "GINSU_MAX_SEARCH_PLOT_LEVELS"
