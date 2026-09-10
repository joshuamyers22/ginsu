# ADR 0010: Stability and sensitivity visualization

- Status: Accepted
- Date: 2026-09-10

## Context

Aggregate stability averages can hide run failures, absences, and wide metric
variation. Parameter-response lines can imply continuity, interpolation, or a
causal effect that a small caller-defined configuration set does not support.
Plotting must also remain optional and must not introduce pandas into the core
data path.

## Decision

Build stability and sensitivity plot data as bounded Polars tables before
loading Plotly. Normalize exact and similarity-aware reports into one row per
anchor and attempted run, retaining availability, presence, run status,
observed metric, aggregate frequency, support counts, and stability status.

The stability renderer pairs a frequency bar or run-level metric distribution
with a complete run-evidence heatmap. Present, absent, and unavailable evidence
use distinct states. The sensitivity renderer uses unconnected run markers and
a separate evidence heatmap. It does not draw a response line for numeric or
categorical parameters. Membership-similarity plots display the declared
reference identifier.

Sensitivity parameters come only from each run's canonical scalar
``parameters_json``. All attempted runs must record the requested parameter;
missing values fail rather than being dropped. Homogeneous integer/float
parameters use a numeric axis. Boolean, string, null, and mixed parameter sets
use categorical labels.

## Consequences

- Failed and resource-limited runs remain visible in every plot.
- Rank, score, support, error lift, and similarity can be inspected as observed
  run-level distributions rather than averages alone.
- Plotly remains a lazy optional dependency and plot-data construction remains
  pandas-free.
- Slice and anchor-by-run cell limits fail before plot rows are expanded.
- The plots are descriptive diagnostics. They do not claim significance,
  causality, interpolation, or robustness outside the supplied runs.
