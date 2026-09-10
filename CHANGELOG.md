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

### Changed

- Renamed the distribution and import package from `sliceline` to `ginsu`.
- Started independent versioning at `0.1.0`.
- Made Polars the core named-table interface; Arrow and pandas table inputs
  return Polars outputs while NumPy inputs preserve ndarray outputs.
- Disabled inherited automatic PyPI/TestPyPI publication pending independent
  release authority and trusted-publisher configuration.

### Fixed

- Reject negative and all-zero error vectors before scoring.
- Round fractional minimum support upward with a minimum of one row.
- Reject reordered or missing named columns before membership evaluation.
