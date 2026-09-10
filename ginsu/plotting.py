"""Optional Plotly visualizations for Ginsu analyses."""

from __future__ import annotations

from typing import Any

import numpy as np

from ginsu._plot_data import (
    error_dependence_data,
    error_dependence_summary,
    impact_plot_data,
    lattice_edges_data,
    overlap_data,
    predicate_matrix_data,
)


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
