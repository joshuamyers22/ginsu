"""Pure-Polars data contracts for stability and sensitivity plots."""

from __future__ import annotations

import json
import math
from typing import Any, Literal

import polars as pl

from ginsu.diagnostics import AnalysisLimitError
from ginsu.stability import SimilarityStabilityReport, StabilityReport

StabilityPlotMetric = Literal[
    "selection_frequency",
    "rank",
    "slice_score",
    "support_fraction",
    "error_lift",
    "similarity",
]
SensitivityMetric = Literal[
    "selected", "rank", "slice_score", "support_fraction", "error_lift"
]

STABILITY_PLOT_SCHEMA = {
    "anchor_id": pl.String,
    "anchor_rule": pl.String,
    "run_id": pl.String,
    "run_status": pl.String,
    "evidence_available": pl.Boolean,
    "present": pl.Boolean,
    "observation_value": pl.Float64,
    "frequency_successful": pl.Float64,
    "present_run_count": pl.UInt64,
    "successful_run_count": pl.UInt64,
    "unavailable_run_count": pl.UInt64,
    "stability_status": pl.String,
    "report_kind": pl.String,
    "similarity_method": pl.String,
    "reference_id": pl.String,
}

SENSITIVITY_PLOT_SCHEMA = {
    **STABILITY_PLOT_SCHEMA,
    "parameter": pl.String,
    "parameter_value": pl.String,
    "parameter_numeric": pl.Float64,
    "parameter_kind": pl.String,
    "parameter_is_numeric": pl.Boolean,
}


def stability_plot_data(
    report: StabilityReport | SimilarityStabilityReport,
    *,
    metric: StabilityPlotMetric = "selection_frequency",
    max_slices: int = 100,
    max_cells: int = 100_000,
) -> pl.DataFrame:
    """Normalize exact or related-rule stability into bounded plot rows."""
    _validate_plot_bounds(max_slices=max_slices, max_cells=max_cells)
    allowed = {
        "selection_frequency",
        "rank",
        "slice_score",
        "support_fraction",
        "error_lift",
        "similarity",
    }
    if metric not in allowed:
        raise ValueError(f"metric must be one of {sorted(allowed)!r}.")
    if not isinstance(report, (StabilityReport, SimilarityStabilityReport)):
        raise TypeError("report must be a Ginsu stability report.")
    similarity_report = isinstance(report, SimilarityStabilityReport)
    if metric == "similarity" and not similarity_report:
        raise ValueError(
            "metric='similarity' requires a SimilarityStabilityReport."
        )

    summary = report.summary
    run_frame = (
        report.exact.runs
        if isinstance(report, SimilarityStabilityReport)
        else report.runs
    )
    slice_count = summary.height
    cells = slice_count * run_frame.height
    if slice_count > max_slices:
        raise AnalysisLimitError(
            "GINSU_MAX_STABILITY_PLOT_SLICES",
            observed=slice_count,
            limit=max_slices,
            stage="stability plot data",
        )
    if cells > max_cells:
        raise AnalysisLimitError(
            "GINSU_MAX_STABILITY_PLOT_CELLS",
            observed=cells,
            limit=max_cells,
            stage="stability plot data",
        )
    if not slice_count:
        return pl.DataFrame(schema=STABILITY_PLOT_SCHEMA)

    if isinstance(report, SimilarityStabilityReport):
        source = report.matches
        identifiers = summary["anchor_id"].to_list()
        summaries = {
            row["anchor_id"]: row for row in summary.iter_rows(named=True)
        }
        source_by_key = {
            (row["anchor_id"], row["run_id"]): row
            for row in source.iter_rows(named=True)
        }
        report_kind = "similarity"
        similarity_method = report.similarity_method
        reference_id = report.reference_id
    else:
        source = report.slice_runs
        identifiers = summary["__ginsu_id"].to_list()
        summaries = {
            row["__ginsu_id"]: row for row in summary.iter_rows(named=True)
        }
        source_by_key = {
            (row["__ginsu_id"], row["run_id"]): row
            for row in source.iter_rows(named=True)
        }
        report_kind = "exact"
        similarity_method = None
        reference_id = None

    rows: list[dict[str, Any]] = []
    run_ids = run_frame["run_id"].to_list()
    for identifier in identifiers:
        aggregate = summaries[identifier]
        rule = aggregate[
            "anchor_rule" if similarity_report else "__ginsu_rule"
        ]
        frequency = aggregate[
            "match_frequency_successful"
            if similarity_report
            else "selection_frequency_successful"
        ]
        present_count = aggregate[
            "matched_run_count" if similarity_report else "selected_run_count"
        ]
        for run_id in run_ids:
            observation = source_by_key[(identifier, run_id)]
            available = observation[
                "evidence_available" if similarity_report else "selected"
            ]
            if not similarity_report:
                available = observation["selected"] is not None
            present = observation[
                "matched" if similarity_report else "selected"
            ]
            value = _metric_value(
                observation,
                metric=metric,
                present=present,
                similarity_report=similarity_report,
            )
            rows.append(
                {
                    "anchor_id": identifier,
                    "anchor_rule": rule,
                    "run_id": run_id,
                    "run_status": observation["run_status"],
                    "evidence_available": available,
                    "present": present,
                    "observation_value": value,
                    "frequency_successful": frequency,
                    "present_run_count": present_count,
                    "successful_run_count": aggregate["successful_run_count"],
                    "unavailable_run_count": aggregate[
                        "unavailable_run_count"
                    ],
                    "stability_status": aggregate["stability_status"],
                    "report_kind": report_kind,
                    "similarity_method": similarity_method,
                    "reference_id": reference_id,
                }
            )
    return pl.DataFrame(rows, schema=STABILITY_PLOT_SCHEMA)


def sensitivity_plot_data(
    report: StabilityReport | SimilarityStabilityReport,
    *,
    parameter: str,
    metric: SensitivityMetric = "rank",
    max_slices: int = 20,
    max_cells: int = 100_000,
) -> pl.DataFrame:
    """Attach one declared run parameter to bounded stability observations."""
    if not isinstance(parameter, str) or not parameter:
        raise ValueError("parameter must be a nonempty string.")
    allowed = {
        "selected",
        "rank",
        "slice_score",
        "support_fraction",
        "error_lift",
    }
    if metric not in allowed:
        raise ValueError(f"metric must be one of {sorted(allowed)!r}.")
    source_metric: StabilityPlotMetric = (
        "selection_frequency" if metric == "selected" else metric
    )
    data = stability_plot_data(
        report,
        metric=source_metric,
        max_slices=max_slices,
        max_cells=max_cells,
    )
    run_frame = (
        report.exact.runs
        if isinstance(report, SimilarityStabilityReport)
        else report.runs
    )
    parameter_rows = []
    kinds = []
    for row in run_frame.select("run_id", "parameters_json").iter_rows(
        named=True
    ):
        parameters = json.loads(row["parameters_json"])
        if parameter not in parameters:
            raise ValueError(
                f"Parameter {parameter!r} is missing from run "
                f"{row['run_id']!r}."
            )
        value = parameters[parameter]
        kind, display, numeric = _parameter_value(value)
        kinds.append(kind)
        parameter_rows.append(
            {
                "run_id": row["run_id"],
                "parameter": parameter,
                "parameter_value": display,
                "parameter_numeric": numeric,
                "parameter_kind": kind,
            }
        )
    numeric_parameter = bool(kinds) and set(kinds) <= {"int", "float"}
    parameters = pl.DataFrame(
        parameter_rows,
        schema={
            "run_id": pl.String,
            "parameter": pl.String,
            "parameter_value": pl.String,
            "parameter_numeric": pl.Float64,
            "parameter_kind": pl.String,
        },
    ).with_columns(pl.lit(numeric_parameter).alias("parameter_is_numeric"))
    if not data.height:
        return pl.DataFrame(schema=SENSITIVITY_PLOT_SCHEMA)
    if metric == "selected":
        data = data.with_columns(
            pl.col("present").cast(pl.Float64).alias("observation_value")
        )
    return data.join(
        parameters, on="run_id", how="left", validate="m:1"
    ).select(*SENSITIVITY_PLOT_SCHEMA)


def _metric_value(
    observation: dict[str, Any],
    *,
    metric: StabilityPlotMetric,
    present: bool | None,
    similarity_report: bool,
) -> float | None:
    if metric == "selection_frequency":
        return float(present) if present is not None else None
    if not present:
        return None
    if metric == "similarity":
        return observation["similarity"] if similarity_report else None
    return observation[metric]


def _parameter_value(value: Any) -> tuple[str, str, float | None]:
    if value is None:
        return "null", "null", None
    if isinstance(value, bool):
        return "bool", str(value).lower(), None
    if isinstance(value, int):
        try:
            numeric = float(value)
        except OverflowError as error:
            raise ValueError(
                "Integer sensitivity parameters must fit Float64."
            ) from error
        if not math.isfinite(numeric):
            raise ValueError(
                "Integer sensitivity parameters must fit Float64."
            )
        return "int", str(value), numeric
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Sensitivity parameters must be finite.")
        return "float", str(value), value
    if isinstance(value, str):
        return "string", value, None
    raise TypeError("Sensitivity parameters must be JSON scalar values.")


def _validate_plot_bounds(*, max_slices: int, max_cells: int) -> None:
    for name, value in (("max_slices", max_slices), ("max_cells", max_cells)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
