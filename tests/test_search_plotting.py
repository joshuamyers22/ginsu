"""Semantic tests for bounded search-profile visualizations."""

from dataclasses import replace

import polars as pl
import pytest

from ginsu import (
    AnalysisLimitError,
    SearchLimitError,
    SearchLimits,
    SearchStageReport,
    Slicefinder,
)
from ginsu._search_plot_data import (
    SEARCH_CARDINALITY_SCHEMA,
    SEARCH_FUNNEL_SCHEMA,
    SEARCH_SUMMARY_SCHEMA,
    SEARCH_TIMING_SCHEMA,
    search_cardinality_data,
    search_funnel_data,
    search_summary_data,
    search_timing_data,
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


def test_level_one_can_prune_after_evaluating_literals(search_report):
    level_one = replace(
        search_report.levels[0],
        source_slices=10,
        candidates_after_pruning=7,
        evaluated_candidates=10,
        valid_candidates=3,
    )
    report = replace(
        search_report,
        levels=(level_one, *search_report.levels[1:]),
    )

    data = search_funnel_data(report)

    level_one_data = data.filter(pl.col("level") == 1)
    assert (
        level_one_data.filter(pl.col("stage") == "After pruning")["count"][0]
        == 7
    )
    assert (
        level_one_data.filter(pl.col("stage") == "Evaluated candidates")[
            "count"
        ][0]
        == 10
    )


def test_search_cardinality_and_summary_are_typed(search_report):
    cardinality = search_cardinality_data(search_report)
    timing = search_timing_data(search_report)
    summary = search_summary_data(search_report)

    assert cardinality.schema == SEARCH_CARDINALITY_SCHEMA
    assert cardinality["feature"].to_list() == ["segment", "region"]
    assert cardinality["cardinality"].to_list() == [2, 2]
    assert cardinality["dtype"].to_list() == ["String", "Int64"]
    assert not cardinality["over_limit"].any()
    assert timing.schema == SEARCH_TIMING_SCHEMA
    assert timing["stage"].to_list() == [
        stage.stage for stage in search_report.stages
    ]
    assert timing["stage_order"].to_list() == list(
        range(len(search_report.stages))
    )
    assert timing["observed_peak_memory_bytes"].null_count() == timing.height
    assert summary.schema == SEARCH_SUMMARY_SCHEMA
    assert summary.height == 1
    assert summary["status"][0] == "complete"
    assert summary["exhaustive"][0]
    assert summary["completed_level_count"][0] == 2
    assert summary["last_completed_level"][0] == 2
    assert summary["recorded_stage_count"][0] == len(search_report.stages)
    assert summary["memory_measurement"][0] is None
    assert summary["observed_peak_memory_bytes"][0] is None
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

    with pytest.raises(AnalysisLimitError) as raised:
        search_timing_data(search_report, max_stages=1)
    assert raised.value.code == "GINSU_MAX_SEARCH_PLOT_STAGES"


def test_search_plot_data_rejects_invalid_inputs(search_report):
    with pytest.raises(TypeError, match="SearchReport"):
        search_summary_data(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive integer"):
        search_funnel_data(search_report, max_levels=0)
    with pytest.raises(ValueError, match="positive integer"):
        search_cardinality_data(search_report, max_features=True)
    with pytest.raises(ValueError, match="positive integer"):
        search_timing_data(search_report, max_stages=0)
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
            replace(search_report, elapsed_seconds=float("nan"), stages=()),
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
    assert figure.layout.xaxis2.title.text == "Search stage"
    assert figure.layout.xaxis3.title.text == "Distinct values"
    assert {trace.name for trace in figure.data} >= {
        "Source slices",
        "Potential pairs",
        "Compatible pairs",
        "After pruning",
        "Evaluated candidates",
        "Valid candidates",
        "Stage duration",
        "Cardinality",
    }
    annotation_text = " ".join(
        annotation.text for annotation in figure.layout.annotations
    )
    assert "Memory: not enabled" in annotation_text
    assert "boundary observations" in annotation_text
    assert "not model-quality evidence" in annotation_text


def test_search_profile_plots_declared_boundary_memory(search_report):
    pytest.importorskip("plotly")
    from ginsu.plotting import plot_search_report

    stages = tuple(
        SearchStageReport(
            stage=stage.stage,
            status=stage.status,
            elapsed_seconds=stage.elapsed_seconds,
            memory_start_bytes=100 + index,
            memory_end_bytes=101 + index,
            observed_peak_memory_bytes=101 + index,
        )
        for index, stage in enumerate(search_report.stages)
    )
    report = replace(
        search_report,
        stages=stages,
        memory_measurement="test_boundary_bytes",
        observed_peak_memory_bytes=100 + len(stages),
    )

    figure = plot_search_report(report)

    memory_trace = next(
        trace
        for trace in figure.data
        if trace.name == "Observed boundary memory"
    )
    assert list(memory_trace.y) == [101 + i for i in range(len(stages))]
    assert figure.layout.meta["memory_measurement"] == "test_boundary_bytes"
    annotation_text = " ".join(
        annotation.text for annotation in figure.layout.annotations
    )
    assert "test_boundary_bytes" in annotation_text


def test_search_profile_labels_legacy_report_without_stage_evidence(
    search_report,
):
    pytest.importorskip("plotly")
    from ginsu.plotting import plot_search_report

    figure = plot_search_report(replace(search_report, stages=()))

    assert "No per-stage timing evidence" in " ".join(
        annotation.text for annotation in figure.layout.annotations
    )


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
