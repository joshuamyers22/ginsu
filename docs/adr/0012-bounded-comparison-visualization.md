# ADR 0012: Bounded analysis-comparison visualization

- Status: Accepted
- Date: 2026-09-10

## Context

Analysis comparisons contain matched, emerged, resolved, and incompatible
outcomes. A chart that keeps only matched rules hides churn. A conventional
aggregate waterfall can also double count people or events because Ginsu
slices overlap, while a membership-migration chart is meaningful only on the
same declared reference population.

Plotting must preserve the Polars-first, optional-Plotly boundary and fail
before an unbounded number of browser marks is created.

## Decision

Normalize comparison plots into bounded Polars tables. Check the number of
comparison change rows before plot expansion and raise
``GINSU_MAX_COMPARISON_PLOT_CHANGES`` when the caller's limit is exceeded.

Provide a dumbbell view for rank, SliceLine score, support count, support
fraction, error lift, or excess error. Connect only rule pairs with both
endpoints. Retain emerged and resolved rules as labeled one-sided endpoints.
Render incompatible comparisons as an explicit status and reason.

Provide a stacked membership-migration view only when the comparison records a
caller-declared common reference. For each rule pair, show baseline-only,
shared, and candidate-only row counts; retain neither, union, and Jaccard in
the plot data and hover evidence. Load Plotly only inside rendering functions.

Do not provide an aggregate excess-error waterfall until Ginsu defines and
tests a non-overlapping allocation policy. Do not infer one by summing
overlapping slice totals.

## Consequences

- Matched movement and rule churn remain visible in the same bounded view.
- Migration claims are tied to explicit common-reference evidence.
- Empty and incompatible comparisons are visibly distinguishable.
- Plot data remains Polars-native and usable without Plotly or pandas.
- The migration bars are per-rule diagnostics, not additive attribution.
- Validation/stability deltas, structured time metadata, timelines, and an
  overlap-allocation contract remain separate work.
