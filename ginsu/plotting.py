"""Optional Plotly visualizations for Ginsu analyses."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import polars as pl

from ginsu._comparison_plot_data import (
    ComparisonPlotMetric,
    comparison_dumbbell_data,
    comparison_migration_data,
)
from ginsu._plot_data import (
    error_dependence_data,
    error_dependence_summary,
    impact_plot_data,
    lattice_edges_data,
    overlap_data,
    predicate_matrix_data,
)
from ginsu._stability_plot_data import (
    SensitivityMetric,
    StabilityPlotMetric,
    sensitivity_plot_data,
    stability_plot_data,
)
from ginsu.comparison import AnalysisComparison


def plot_impact(finder: Any):
    """Plot error lift against support for each discovered slice."""
    go = _plotly_graph_objects()
    data = impact_plot_data(finder)
    marker_sizes = 10 + 35 * np.sqrt(
        data.get_column("support_fraction").to_numpy()
    )
    figure = go.Figure(
        go.Scatter(
            x=data.get_column("error_lift").to_list(),
            y=data.get_column("__ginsu_rule").to_list(),
            mode="markers",
            marker={
                "color": data.get_column("slice_score").to_list(),
                "colorbar": {"title": "Slice score"},
                "colorscale": "Viridis",
                "size": marker_sizes.tolist(),
                "showscale": True,
            },
            customdata=data.select(
                "__ginsu_id",
                "support_count",
                "support_fraction",
                "excess_error",
            ).to_numpy(),
            hovertemplate=(
                "%{y}<br>Error lift=%{x:.3f}<br>Support=%{customdata[1]} "
                "(%{customdata[2]:.1%})<br>Excess error=%{customdata[3]:.3f}"
                "<extra></extra>"
            ),
        )
    )
    figure.add_vline(x=1.0, line_dash="dash", line_color="gray")
    figure.update_layout(
        title="Ginsu slice impact",
        xaxis_title="Observed error lift vs. population",
        yaxis_title="Slice rule",
        template="plotly_white",
    )
    return figure


def plot_predicate_matrix(
    finder: Any,
    *,
    max_slices: int = 100,
    max_cells: int = 10_000,
):
    """Plot slice composition as a bounded feature-by-rule matrix."""
    go = _plotly_graph_objects()
    data = predicate_matrix_data(
        finder, max_slices=max_slices, max_cells=max_cells
    )
    slice_count = finder.slices_.height
    features = [name for name, _dtype in finder._feature_schema]
    rules = finder.slices_.get_column("__ginsu_rule").to_list()
    labels = [f"#{rank}: {rule}" for rank, rule in enumerate(rules, start=1)]
    if slice_count and features:
        used = (
            data.get_column("is_used")
            .cast(int)
            .to_numpy()
            .reshape(slice_count, len(features))
        )
        text = (
            data.get_column("display_value")
            .fill_null("")
            .to_numpy()
            .reshape(slice_count, len(features))
        )
        custom = np.stack(
            (
                data.get_column("support_count").to_numpy(),
                data.get_column("error_lift").to_numpy(),
                data.get_column("slice_score").to_numpy(),
            ),
            axis=1,
        ).reshape(slice_count, len(features), 3)
    else:
        used = np.empty((slice_count, len(features)))
        text = np.empty((slice_count, len(features)))
        custom = np.empty((slice_count, len(features), 3))

    figure = go.Figure(
        go.Heatmap(
            x=features,
            y=labels,
            z=used,
            zmin=0,
            zmax=1,
            colorscale=((0, "#f7f7f7"), (1, "#4c78a8")),
            showscale=False,
            text=text,
            texttemplate="%{text}",
            customdata=custom,
            hovertemplate=(
                "%{y}<br>Feature=%{x}<br>Value=%{text}"
                "<br>Support=%{customdata[0]}"
                "<br>Error lift=%{customdata[1]:.3f}"
                "<br>Slice score=%{customdata[2]:.3f}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title="Ginsu predicate matrix",
        xaxis_title="Feature",
        yaxis_title="Slice rule",
        template="plotly_white",
    )
    return figure


def plot_error_dependence(
    finder: Any,
    X: Any,
    errors: Any,
    *,
    feature: str,
    slice_id: str | None = None,
    max_points: int = 5_000,
    seed: int = 0,
):
    """Plot observed feature/loss dependence inside and outside one slice.

    This is descriptive model-error dependence, not causal or conventional
    model partial dependence. Summaries use all rows; only rendered points are
    deterministically sampled.
    """
    if max_points <= 0:
        raise ValueError("max_points must be positive.")
    go = _plotly_graph_objects()
    data = error_dependence_data(
        finder, X, errors, feature=feature, slice_id=slice_id
    )
    summary = error_dependence_summary(data)
    rendered = _sample_rows(data, max_points=max_points, seed=seed)
    numeric = data.schema["feature_value"].is_numeric()
    figure = go.Figure()

    for in_slice, label, color in (
        (False, "Outside slice", "#7f8c8d"),
        (True, "Inside slice", "#d62728"),
    ):
        group = rendered.filter(rendered["in_slice"] == in_slice)
        if numeric:
            figure.add_trace(
                go.Scattergl(
                    x=group.get_column("feature_value").to_list(),
                    y=group.get_column("error").to_list(),
                    mode="markers",
                    name=label,
                    marker={"color": color, "opacity": 0.45, "size": 6},
                    customdata=group.get_column("__ginsu_row").to_list(),
                    hovertemplate=(
                        f"{feature}=%{{x}}<br>Error=%{{y:.3f}}"
                        "<br>Row=%{customdata}<extra></extra>"
                    ),
                )
            )
            summary_group = summary.filter(summary["in_slice"] == in_slice)
            figure.add_trace(
                go.Scatter(
                    x=summary_group.get_column("feature_mean").to_list(),
                    y=summary_group.get_column("error_mean").to_list(),
                    mode="lines+markers",
                    name=f"{label} mean",
                    line={"color": color, "width": 3},
                )
            )
        else:
            figure.add_trace(
                go.Box(
                    x=group.get_column("feature_value").to_list(),
                    y=group.get_column("error").to_list(),
                    name=label,
                    marker_color=color,
                    boxpoints="outliers",
                )
            )

    figure.update_layout(
        title=f"Observed error dependence: {feature}",
        xaxis_title=feature,
        yaxis_title="Observed model loss",
        template="plotly_white",
        annotations=[
            {
                "text": "Descriptive association; not causal or partial dependence.",
                "showarrow": False,
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": 1.08,
                "xanchor": "left",
            }
        ],
    )
    return figure


def plot_overlap(
    finder: Any,
    X: Any,
    *,
    metric: str = "jaccard",
    max_slices: int = 100,
    max_cells: int = 10_000,
    max_membership_cells: int = 10_000_000,
):
    """Plot a bounded pairwise Jaccard heatmap of returned slice rules."""
    go = _plotly_graph_objects()
    data = overlap_data(
        finder,
        X,
        metric=metric,
        max_slices=max_slices,
        max_cells=max_cells,
        max_membership_cells=max_membership_cells,
    )
    identifiers = finder.slices_.get_column("__ginsu_id").to_list()
    rules = finder.slices_.get_column("__ginsu_rule").to_list()
    labels = [f"#{rank}: {rule}" for rank, rule in enumerate(rules, start=1)]
    count = len(identifiers)
    if count:
        values = data.get_column("jaccard").to_numpy().reshape(count, count)
        custom = np.stack(
            (
                data.get_column("intersection_count").to_numpy(),
                data.get_column("union_count").to_numpy(),
                data.get_column("equivalent").to_numpy(),
            ),
            axis=1,
        ).reshape(count, count, 3)
    else:
        values = np.empty((0, 0))
        custom = np.empty((0, 0, 3))

    figure = go.Figure(
        go.Heatmap(
            x=labels,
            y=labels,
            z=values,
            zmin=0,
            zmax=1,
            colorscale="Blues",
            colorbar={"title": "Jaccard"},
            customdata=custom,
            hovertemplate=(
                "%{y}<br>%{x}<br>Jaccard=%{z:.3f}"
                "<br>Intersection=%{customdata[0]}"
                "<br>Union=%{customdata[1]}"
                "<br>Equivalent=%{customdata[2]}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title="Ginsu slice overlap",
        xaxis_title="Slice rule",
        yaxis_title="Slice rule",
        template="plotly_white",
    )
    return figure


def plot_lattice(finder: Any, *, max_nodes: int = 100):
    """Plot deterministic rule-refinement edges among returned slices."""
    go = _plotly_graph_objects()
    edges = lattice_edges_data(finder, max_nodes=max_nodes)
    nodes = finder.slices_.join(
        finder.slice_statistics_.select(
            "__ginsu_id",
            "support_count",
            "error_lift",
            "slice_score",
            "predicate_count",
        ),
        on="__ginsu_id",
        how="left",
        validate="1:1",
    )

    positions: dict[str, tuple[float, float]] = {}
    predicate_counts = nodes.get_column("predicate_count").to_list()
    identifiers = nodes.get_column("__ginsu_id").to_list()
    for level in sorted(set(predicate_counts)):
        level_ids = [
            identifier
            for identifier, count in zip(
                identifiers, predicate_counts, strict=True
            )
            if count == level
        ]
        offset = (len(level_ids) - 1) / 2
        for index, identifier in enumerate(level_ids):
            positions[identifier] = (index - offset, -float(level))

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for parent_id, child_id in edges.select(
        "parent_id", "child_id"
    ).iter_rows():
        parent_x, parent_y = positions[parent_id]
        child_x, child_y = positions[child_id]
        edge_x.extend((parent_x, child_x, None))
        edge_y.extend((parent_y, child_y, None))

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line={"color": "#aab2bd", "width": 1.5},
            hoverinfo="skip",
            name="Adds one predicate",
        )
    )
    support = nodes.get_column("support_count").to_numpy()
    max_support = float(support.max()) if support.size else 1.0
    sizes = 14 + 28 * np.sqrt(support / max_support)
    node_x = [positions[identifier][0] for identifier in identifiers]
    node_y = [positions[identifier][1] for identifier in identifiers]
    figure.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            name="Slice",
            marker={
                "size": sizes.tolist(),
                "color": nodes.get_column("error_lift").to_list(),
                "colorscale": "Viridis",
                "showscale": True,
                "colorbar": {"title": "Error lift"},
                "line": {"color": "white", "width": 1},
            },
            customdata=nodes.select(
                "__ginsu_rule", "support_count", "slice_score"
            ).to_numpy(),
            hovertemplate=(
                "%{customdata[0]}<br>Support=%{customdata[1]}"
                "<br>Slice score=%{customdata[2]:.3f}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title="Ginsu slice lattice",
        xaxis={"visible": False},
        yaxis={
            "title": "Predicate count (rule refinement)",
            "tickvals": sorted({-float(value) for value in predicate_counts}),
            "ticktext": [
                str(value)
                for value in sorted(set(predicate_counts), reverse=True)
            ],
        },
        template="plotly_white",
        annotations=[
            {
                "text": "Edges mean rule refinement, not causality or transitions.",
                "showarrow": False,
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": 1.08,
                "xanchor": "left",
            }
        ],
    )
    return figure


def plot_stability(
    report: Any,
    *,
    metric: StabilityPlotMetric = "selection_frequency",
    max_slices: int = 100,
    max_cells: int = 100_000,
):
    """Plot recurrence or per-run metric distributions with availability."""
    go = _plotly_graph_objects()
    make_subplots = _plotly_make_subplots()
    data = stability_plot_data(
        report,
        metric=metric,
        max_slices=max_slices,
        max_cells=max_cells,
    )
    if not data.height:
        figure = go.Figure()
        figure.add_annotation(text="No discovered slices", showarrow=False)
        figure.update_layout(title="Ginsu stability", template="plotly_white")
        return figure

    rules = data["anchor_rule"].unique(maintain_order=True).to_list()
    run_ids = data["run_id"].unique(maintain_order=True).to_list()
    figure = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        column_widths=(0.55, 0.45),
        horizontal_spacing=0.03,
        subplot_titles=("Summary / observed distribution", "Run evidence"),
    )
    if metric == "selection_frequency":
        aggregate = data.unique("anchor_id", maintain_order=True)
        figure.add_trace(
            go.Bar(
                x=aggregate["frequency_successful"].to_list(),
                y=aggregate["anchor_rule"].to_list(),
                orientation="h",
                marker_color=[
                    _stability_color(status)
                    for status in aggregate["stability_status"].to_list()
                ],
                customdata=aggregate.select(
                    "present_run_count",
                    "successful_run_count",
                    "unavailable_run_count",
                    "stability_status",
                ).to_numpy(),
                hovertemplate=(
                    "%{y}<br>Frequency=%{x:.1%}"
                    "<br>Present=%{customdata[0]}/%{customdata[1]}"
                    "<br>Unavailable=%{customdata[2]}"
                    "<br>Status=%{customdata[3]}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    else:
        for rule in rules:
            observed = data.filter(
                (pl.col("anchor_rule") == rule)
                & pl.col("observation_value").is_not_null()
            )
            figure.add_trace(
                go.Box(
                    x=observed["observation_value"].to_list(),
                    y=[rule] * observed.height,
                    orientation="h",
                    boxpoints="all",
                    jitter=0.25,
                    pointpos=0,
                    customdata=observed.select(
                        "run_id", "run_status"
                    ).to_numpy(),
                    hovertemplate=(
                        "%{y}<br>Value=%{x:.4g}<br>Run=%{customdata[0]}"
                        "<br>Status=%{customdata[1]}<extra></extra>"
                    ),
                    showlegend=False,
                    marker_color="#4c78a8",
                ),
                row=1,
                col=1,
            )
    z, statuses = _run_evidence_matrix(data, rules=rules, run_ids=run_ids)
    figure.add_trace(
        go.Heatmap(
            x=run_ids,
            y=rules,
            z=z,
            zmin=-1,
            zmax=1,
            colorscale=(
                (0.0, "#8c8c8c"),
                (0.499, "#8c8c8c"),
                (0.5, "#f2f2f2"),
                (0.749, "#f2f2f2"),
                (0.75, "#4c78a8"),
                (1.0, "#4c78a8"),
            ),
            colorbar={
                "title": "Run evidence",
                "tickvals": (-1, 0, 1),
                "ticktext": ("Unavailable", "Absent", "Present"),
            },
            customdata=statuses,
            hovertemplate=(
                "%{y}<br>Run=%{x}<br>Evidence=%{customdata}<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )
    report_kind = data["report_kind"][0]
    similarity_method = data["similarity_method"][0]
    title = (
        "Ginsu exact-rule stability"
        if report_kind == "exact"
        else f"Ginsu {similarity_method}-similarity stability"
    )
    note = "Descriptive recurrence; gray cells are unavailable runs."
    reference_id = data["reference_id"][0]
    if reference_id is not None:
        note += f" Reference: {reference_id}."
    figure.update_layout(
        title=title,
        template="plotly_white",
        height=max(420, 34 * len(rules) + 180),
        annotations=[
            *list(figure.layout.annotations),
            {
                "text": note,
                "showarrow": False,
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": 1.12,
                "xanchor": "left",
            },
        ],
    )
    figure.update_xaxes(
        title_text=_stability_metric_label(metric), row=1, col=1
    )
    if metric == "selection_frequency":
        figure.update_xaxes(range=(0, 1), tickformat=".0%", row=1, col=1)
    elif metric == "rank":
        figure.update_xaxes(autorange="reversed", row=1, col=1)
    figure.update_xaxes(title_text="Run", row=1, col=2)
    figure.update_yaxes(title_text="Slice rule", row=1, col=1)
    return figure


def plot_sensitivity(
    report: Any,
    *,
    parameter: str,
    metric: SensitivityMetric = "rank",
    max_slices: int = 20,
    max_cells: int = 100_000,
):
    """Plot run-level parameter response without implying causal effects."""
    go = _plotly_graph_objects()
    make_subplots = _plotly_make_subplots()
    data = sensitivity_plot_data(
        report,
        parameter=parameter,
        metric=metric,
        max_slices=max_slices,
        max_cells=max_cells,
    )
    if not data.height:
        figure = go.Figure()
        figure.add_annotation(text="No discovered slices", showarrow=False)
        figure.update_layout(
            title="Ginsu sensitivity", template="plotly_white"
        )
        return figure
    figure = make_subplots(
        rows=2,
        cols=1,
        row_heights=(0.72, 0.28),
        vertical_spacing=0.16,
        subplot_titles=("Observed parameter response", "Run evidence"),
    )
    numeric = data["parameter_is_numeric"][0]
    rules = data["anchor_rule"].unique(maintain_order=True).to_list()
    for rule in rules:
        observed = data.filter(
            (pl.col("anchor_rule") == rule)
            & pl.col("observation_value").is_not_null()
        )
        x_column = "parameter_numeric" if numeric else "parameter_value"
        figure.add_trace(
            go.Scatter(
                x=observed[x_column].to_list(),
                y=observed["observation_value"].to_list(),
                mode="markers",
                name=rule,
                customdata=observed.select(
                    "run_id", "run_status", "parameter_value"
                ).to_numpy(),
                hovertemplate=(
                    "%{fullData.name}<br>Parameter=%{customdata[2]}"
                    "<br>Value=%{y:.4g}<br>Run=%{customdata[0]}"
                    "<br>Status=%{customdata[1]}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
    run_ids = data["run_id"].unique(maintain_order=True).to_list()
    labels = []
    for run_id in run_ids:
        value = data.filter(pl.col("run_id") == run_id)["parameter_value"][0]
        labels.append(f"{run_id}<br>{parameter}={value}")
    z, statuses = _run_evidence_matrix(data, rules=rules, run_ids=run_ids)
    figure.add_trace(
        go.Heatmap(
            x=labels,
            y=rules,
            z=z,
            zmin=-1,
            zmax=1,
            colorscale=(
                (0.0, "#8c8c8c"),
                (0.499, "#8c8c8c"),
                (0.5, "#f2f2f2"),
                (0.749, "#f2f2f2"),
                (0.75, "#4c78a8"),
                (1.0, "#4c78a8"),
            ),
            showscale=False,
            customdata=statuses,
            hovertemplate=(
                "%{y}<br>%{x}<br>Evidence=%{customdata}<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )
    figure.update_layout(
        title=f"Ginsu sensitivity: {parameter} vs. {metric}",
        template="plotly_white",
        height=max(620, 30 * len(rules) + 440),
        annotations=[
            *list(figure.layout.annotations),
            {
                "text": (
                    "Descriptive across declared runs; markers are not a "
                    "causal or significance claim."
                ),
                "showarrow": False,
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": 1.10,
                "xanchor": "left",
            },
        ],
    )
    figure.update_xaxes(title_text=parameter, row=1, col=1)
    figure.update_yaxes(
        title_text=_sensitivity_metric_label(metric), row=1, col=1
    )
    if metric == "rank":
        figure.update_yaxes(autorange="reversed", row=1, col=1)
    figure.update_xaxes(
        title_text="Run and parameter configuration", row=2, col=1
    )
    figure.update_yaxes(title_text="Slice rule", row=2, col=1)
    return figure


def plot_comparison(
    comparison: AnalysisComparison,
    *,
    kind: Literal["dumbbell", "migration"] = "dumbbell",
    metric: ComparisonPlotMetric = "error_lift",
    max_changes: int = 100,
):
    """Plot bounded metric movement or common-reference membership migration."""
    if kind not in ("dumbbell", "migration"):
        raise ValueError("kind must be 'dumbbell' or 'migration'.")
    if not isinstance(comparison, AnalysisComparison):
        raise TypeError("comparison must be an AnalysisComparison.")
    if kind == "migration":
        data = comparison_migration_data(comparison, max_changes=max_changes)
    else:
        data = comparison_dumbbell_data(
            comparison, metric=metric, max_changes=max_changes
        )
    go = _plotly_graph_objects()
    if not comparison.comparable:
        summary = comparison.summary.row(0, named=True)
        figure = go.Figure()
        figure.add_annotation(
            text=(
                f"{summary['compatibility_status']}: "
                f"{summary['compatibility_reason']}"
            ),
            showarrow=False,
        )
        figure.update_layout(
            title="Ginsu analysis comparison: not comparable",
            template="plotly_white",
        )
        return figure
    if kind == "migration":
        return _plot_comparison_migration(comparison, data=data, go=go)
    return _plot_comparison_dumbbell(data, go=go, metric=metric)


def _plot_comparison_dumbbell(
    data: pl.DataFrame,
    *,
    go: Any,
    metric: ComparisonPlotMetric,
):
    figure = go.Figure()
    if not data.height:
        figure.add_annotation(text="No comparison changes", showarrow=False)
    else:
        paired = data.filter(
            pl.col("baseline_value").is_not_null()
            & pl.col("candidate_value").is_not_null()
        )
        line_x: list[float | None] = []
        line_y: list[str | None] = []
        for row in paired.iter_rows(named=True):
            line_x.extend(
                [row["baseline_value"], row["candidate_value"], None]
            )
            line_y.extend([row["label"], row["label"], None])
        if line_x:
            figure.add_trace(
                go.Scatter(
                    x=line_x,
                    y=line_y,
                    mode="lines",
                    name="Change",
                    line={"color": "#b8b8b8", "width": 2},
                    hoverinfo="skip",
                )
            )
        for side, color, symbol in (
            ("baseline", "#4c78a8", "circle"),
            ("candidate", "#f58518", "diamond"),
        ):
            side_data = data.filter(pl.col(f"{side}_value").is_not_null())
            figure.add_trace(
                go.Scatter(
                    x=side_data[f"{side}_value"].to_list(),
                    y=side_data["label"].to_list(),
                    mode="markers",
                    name=side.title(),
                    marker={"color": color, "size": 10, "symbol": symbol},
                    customdata=side_data.select(
                        "comparison_status",
                        "match_type",
                        "similarity",
                        "delta",
                    ).to_numpy(),
                    hovertemplate=(
                        "%{y}<br>Value=%{x:.4g}"
                        "<br>Status=%{customdata[0]}"
                        "<br>Match=%{customdata[1]}"
                        "<br>Similarity=%{customdata[2]}"
                        "<br>Candidate − baseline=%{customdata[3]}"
                        "<extra>%{fullData.name}</extra>"
                    ),
                )
            )
    figure.update_layout(
        title=f"Ginsu analysis comparison: {metric}",
        xaxis_title=_comparison_metric_label(metric),
        yaxis_title="Rule change",
        template="plotly_white",
        height=max(420, 34 * data.height + 220),
        annotations=[
            *list(figure.layout.annotations),
            {
                "text": (
                    "Descriptive discovery metrics; emerged and resolved "
                    "rules remain explicit."
                ),
                "showarrow": False,
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": 1.10,
                "xanchor": "left",
            },
        ],
    )
    figure.update_yaxes(autorange="reversed")
    if metric == "rank":
        figure.update_xaxes(autorange="reversed")
    return figure


def _plot_comparison_migration(
    comparison: AnalysisComparison, *, data: pl.DataFrame, go: Any
):
    figure = go.Figure()
    for segment, label, color in (
        ("baseline_only", "Baseline only", "#4c78a8"),
        ("both", "Shared", "#54a24b"),
        ("candidate_only", "Candidate only", "#f58518"),
    ):
        group = data.filter(pl.col("membership_segment") == segment)
        figure.add_trace(
            go.Bar(
                x=group["membership_count"].to_list(),
                y=group["label"].to_list(),
                orientation="h",
                name=label,
                marker_color=color,
                customdata=group.select(
                    "comparison_status",
                    "reference_neither_count",
                    "reference_union_count",
                    "reference_jaccard",
                ).to_numpy(),
                hovertemplate=(
                    "%{y}<br>Rows=%{x}<br>Status=%{customdata[0]}"
                    "<br>Neither=%{customdata[1]}"
                    "<br>Union=%{customdata[2]}"
                    "<br>Jaccard=%{customdata[3]}<extra>%{fullData.name}</extra>"
                ),
            )
        )
    if not data.height:
        figure.add_annotation(text="No comparison changes", showarrow=False)
    figure.update_layout(
        title="Ginsu reference membership migration",
        xaxis_title="Reference rows",
        yaxis_title="Rule change",
        barmode="stack",
        template="plotly_white",
        height=max(420, 34 * comparison.changes.height + 220),
        annotations=[
            *list(figure.layout.annotations),
            {
                "text": (
                    f"Reference: {comparison.reference_id}. Each bar is one "
                    "rule pair, not an additive attribution."
                ),
                "showarrow": False,
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": 1.10,
                "xanchor": "left",
            },
        ],
    )
    figure.update_yaxes(autorange="reversed")
    return figure


def _comparison_metric_label(metric: ComparisonPlotMetric) -> str:
    return {
        "rank": "Discovery rank",
        "slice_score": "Slice score",
        "support_count": "Support count",
        "support_fraction": "Support fraction",
        "error_lift": "Observed error lift",
        "excess_error": "Observed excess error",
    }[metric]


def _run_evidence_matrix(data, *, rules, run_ids):
    by_key = {
        (row["anchor_rule"], row["run_id"]): row
        for row in data.iter_rows(named=True)
    }
    values = []
    statuses = []
    for rule in rules:
        value_row = []
        status_row = []
        for run_id in run_ids:
            row = by_key[(rule, run_id)]
            if not row["evidence_available"]:
                value_row.append(-1)
                status_row.append(row["run_status"])
            elif row["present"]:
                value_row.append(1)
                status_row.append("present")
            else:
                value_row.append(0)
                status_row.append("absent")
        values.append(value_row)
        statuses.append(status_row)
    return values, statuses


def _stability_color(status: str) -> str:
    return {
        "stable": "#2e8b57",
        "fragile": "#e07b39",
        "insufficient_successful_runs": "#8c8c8c",
    }[status]


def _stability_metric_label(metric: str) -> str:
    return {
        "selection_frequency": "Selection / match frequency",
        "rank": "Discovery rank",
        "slice_score": "Slice score",
        "support_fraction": "Support fraction",
        "error_lift": "Observed error lift",
        "similarity": "Jaccard similarity",
    }[metric]


def _sensitivity_metric_label(metric: str) -> str:
    return {
        "selected": "Selected / matched",
        "rank": "Discovery rank",
        "slice_score": "Slice score",
        "support_fraction": "Support fraction",
        "error_lift": "Observed error lift",
    }[metric]


def _sample_rows(data, *, max_points: int, seed: int):
    if data.height <= max_points:
        return data
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(data.height, max_points, replace=False))
    return data[indices]


def _plotly_graph_objects():
    try:
        import plotly.graph_objects as go
    except ImportError as error:
        raise ImportError(
            "Plotting requires the optional dependency: pip install 'ginsu[plot]'."
        ) from error
    return go


def _plotly_make_subplots():
    try:
        from plotly.subplots import make_subplots
    except ImportError as error:
        raise ImportError(
            "Plotting requires the optional dependency: pip install 'ginsu[plot]'."
        ) from error
    return make_subplots
