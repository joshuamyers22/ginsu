# ADR 0003: Keep Plot Data Separate from the Optional Renderer

- Status: Accepted
- Date: 2026-09-09

## Context

Ginsu needs reproducible impact and dependence visualizations without making a
browser-oriented plotting stack part of its core import path. Dependence plots
can also be mistaken for causal or model partial-dependence results.

## Decision

- Build every plotting dataset with pure Polars functions in `_plot_data.py`.
- Load Plotly only inside renderer calls and distribute it through the `plot`
  extra.
- Define the initial dependence view as observed feature/loss association,
  stratified by slice membership.
- Compute summaries from all eligible rows. Deterministic sampling may reduce
  only the raw points sent to the renderer.
- Put an explicit non-causal, non-partial-dependence annotation on the figure.

## Consequences

Core users do not install or import Plotly. Plot-data contracts can be tested
without a renderer, while semantic figure tests inspect traces, labels, and
annotations rather than brittle full-figure snapshots.
