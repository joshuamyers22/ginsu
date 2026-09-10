# ADR 0002: Use Polars as the Canonical Table Boundary

- Status: Accepted for implementation
- Date: 2026-09-09

## Context

Ginsu needs named, typed result tables and plotting data. It must be Polars
native while remaining usable from PyArrow and pandas without producer-specific
implementation branches. The inherited search engine relies on efficient NumPy
and SciPy sparse operations.

## Decision

- Normalize supported named-table inputs to Polars through public Arrow
  PyCapsule or dataframe interchange.
- Support pandas through that protocol; Ginsu production code does not import
  pandas or inspect its private state.
- Preserve NumPy/SciPy sparse matrices behind one measured conversion boundary.
- Treat conversion allocation, dtype changes, null semantics, column order, and
  schema changes as explicit contracts.
- Keep Plotly optional and build plot data with Polars independently of the
  renderer.

## Consequences

Polars becomes a core dependency. Not every Arrow-compatible conversion is
zero-copy, so compatibility tests and diagnostics must expose copies and reject
unsupported schema changes. The sparse algorithm is not rewritten merely to
claim dataframe purity.
