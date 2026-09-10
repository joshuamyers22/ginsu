# ADR 0013: Bounded search-profile visualization

- Status: Accepted
- Date: 2026-09-10

## Context

Search cost can grow sharply with feature cardinality and lattice depth. A
single elapsed-time number cannot show where candidates multiplied or were
discarded, and successful-only charts hide searches stopped by resource
limits. At the same time, the current ``SearchReport`` records total elapsed
time but does not contain stage timing or peak-memory observations.

The visualization must stay within the Polars-first, optional-Plotly boundary
and must not imply that computational cost measures model quality.

## Decision

Normalize search-profile evidence into three typed Polars tables: one row per
completed level and funnel stage, one row per ordered source feature, and one
summary row. Bound lattice levels, expanded funnel cells, and source features
before loading Plotly. Fail with stable
``GINSU_MAX_SEARCH_PLOT_LEVELS``, ``GINSU_MAX_SEARCH_PLOT_CELLS``, or
``GINSU_MAX_SEARCH_PLOT_FEATURES`` codes.

Plot source slices, potential pairs, compatible pairs, post-pruning candidates,
evaluated candidates, and valid candidates across completed lattice levels.
Pair generation does not apply to the first literal level, so those values are
null rather than zero. Use both color and marker/line style to distinguish
stages.

Plot ordered source cardinalities separately and mark values that exceed the
active cardinality limit with color and pattern. Retain status, exhaustiveness,
backend, Numba use, input size, encoded width, copy boundaries, warnings,
termination reason, total elapsed time, and every active search limit in typed
data and figure metadata. Early termination before any completed level remains
explicit.

Do not fabricate per-stage timing, peak memory, or an exact incomplete lattice
level from the termination message. Display the last completed level and state
that stage timing and peak memory were not recorded.

## Consequences

- Candidate expansion and attrition can be inspected without raw search data.
- Limit-terminated and failed searches remain visible, including pre-lattice
  failures.
- Level-one pair counts cannot be mistaken for observed zero pair generation.
- Plotly remains lazy, while custom reporting can consume the Polars tables.
- Per-stage timing and optional peak-memory instrumentation remain a future
  ``SearchReport`` contract change.
- The profile is execution evidence, not model-performance evidence.
