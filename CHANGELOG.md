# Changelog

All notable Ginsu changes will be documented here.

The format follows Keep a Changelog, and releases will use semantic versioning
after the pre-1.0 API is established.

## Unreleased

### Added

- Independent Ginsu project identity and production implementation plan.
- Regression validation for named-column order, loss domain, and fractional
  minimum support.
- Canonical Polars/Arrow input normalization with pandas interoperability.
- Immutable predicates and slices with stable versioned identifiers.
- Canonical Polars slice, statistic, and predicate result frames.
- Stable-ID membership frames with explicit positional or caller-supplied row
  identity.
- Polars-native impact and observed error-dependence plot data, with optional
  Plotly renderers.
- Frozen dependency lock and isolated performance-test target.
- Static type checking and offline warnings-as-errors documentation gates.
- Fitted Polars-native fixed, equal-width, quantile, and categorical
  discretization with stable rare/unknown handling.
- Immutable search reports with per-level candidate funnels, actual backend
  use, copy boundaries, stable termination codes, and conservative resource
  limits.
- Bounded exact-equivalence, Jaccard-overlap, and rule-lattice Polars tables,
  plus optional Plotly heatmap and deterministic lattice renderers.
- Bounded predicate-matrix data and an optional Plotly composition heatmap.
- Versioned, deterministic ``SliceAnalysis`` artifacts using closed-schema JSON
  and uncompressed Arrow IPC, with hashes, read limits, relational validation,
  and fitted discretization round trips.
- Fixed-rule descriptive holdout validation with preserved discovery order,
  explicit support status, zero-baseline handling, and membership limits.
- Optional deterministic percentile-bootstrap intervals and one-sided
  fixed-rule permutation tests with Holm correction, explicit testability
  status, and bounded resampling work.
- Auditable score, exact-membership, and greedy Jaccard-diversity selection
  views with representative, exclusion, and incremental-coverage evidence.
- Failure-aware exact-rule stability aggregation across caller-controlled runs,
  including recurrence, rank, score, support, and error-lift distributions.
- Deterministic anchor-based predicate and common-reference membership
  similarity reports with audited candidates, row-identity fingerprints,
  undefined-empty handling, and fail-fast work limits.
- Pure-Polars stability and parameter-sensitivity plot data with bounded
  optional Plotly frequency, distribution, parameter-response, and run-evidence
  panels.
- Bounded one-to-one analysis comparison with exact, predicate, or
  common-reference membership matching; metric and predicate deltas;
  migration counts; and explicit emerged, resolved, or non-comparable results.
- Bounded Polars-native comparison dumbbell and common-reference membership
  migration data, with optional Plotly renderers that retain unmatched and
  incompatible outcomes.
- Bounded Polars-native search funnel, feature-cardinality, and execution
  summary data with an optional Plotly profile that retains early termination,
  resource limits, backend, and copy-boundary evidence.
- Deterministic per-stage search timing and opt-in labeled boundary-memory
  observations, carried by artifact format v2 and rendered in the bounded
  search profile; legacy artifact v1 remains readable without invented data.
- Deterministic offline Polars notebook tutorials with disjoint model-training,
  slice-discovery, and fixed-rule-validation partitions, plus a pandas-free
  notebook execution gate and dedicated compatibility dependencies.
- Sliceline, pandas, and NumPy migration guidance; an explicit limitations
  reference; canonical fitted-output documentation; and executable documentation
  examples in the quality gate.
- Coverage measurement for the owned validation and optional Numba modules.

### Changed

- Renamed the distribution and import package from `sliceline` to `ginsu`.
- Started independent versioning at `0.1.0`.
- Made Polars the core named-table interface; Arrow and pandas table inputs
  return Polars outputs while NumPy inputs preserve ndarray outputs.
- Disabled inherited automatic PyPI/TestPyPI publication pending independent
  release authority and trusted-publisher configuration.
- Made the performance target execute both calibrated benchmarks and
  performance-marked resource/parity checks.
- Corrected inherited estimator documentation that implied minimum support or
  discovery ranking established statistical significance.

### Fixed

- Reject negative and all-zero error vectors before scoring.
- Round fractional minimum support upward with a minimum of one row.
- Reject reordered or missing named columns before membership evaluation.
- Permit level-one search reports where literal evaluation precedes pruning
  for the next lattice join.
