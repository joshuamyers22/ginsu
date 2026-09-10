# ADR 0014: Stage diagnostics and analysis artifact version 2

- Status: Accepted
- Date: 2026-09-10

## Context

The search profile records candidate attrition and total elapsed time, but a
single duration cannot locate expensive normalization, encoding, lattice join,
evaluation, or result-materialization work. Process-memory APIs also have
different semantics and portability costs, so the core library cannot label an
implicit reading as a precise allocation or continuous peak.

Adding evidence to ``SearchReport`` changes the closed analysis-manifest schema.
Existing version 1 artifacts must remain readable without inventing data that
was never observed.

## Decision

Record ordered ``SearchStageReport`` values using the estimator's injected
monotonic clock. Each stage has a stable name, a ``complete`` or ``terminated``
status, and a finite nonnegative elapsed duration. A failure or resource limit
records the active stage as terminated when doing so cannot mask the original
error.

Memory evidence is opt-in. Callers provide both a zero-argument sampler that
returns nonnegative integer bytes and a bounded measurement label describing
its semantics. Ginsu samples at stage start and end, carries the largest of
those boundary values for the stage and report, and stores the declared label.
It does not claim allocation attribution or a continuous peak unless the
caller-provided sampler itself has peak-to-date semantics. Sampler callables are
never serialized.

Add a bounded, typed Polars timing table and a search-profile timing panel.
``max_stages`` is checked before Plotly loads and fails with
``GINSU_MAX_SEARCH_PLOT_STAGES``. Terminated stages use color and pattern, and
optional memory uses a separately labeled byte axis. Reports without stage
evidence say so explicitly.

The artifact writer now emits ``ginsu.slice-analysis`` version ``2.0.0`` with
closed stage and memory fields. The reader accepts both version ``2.0.0`` and
legacy ``1.0.0``. Version 1 is migrated in memory by assigning an empty stage
sequence and unavailable memory fields. Unknown versions and invalid
cross-field memory evidence continue to fail closed.

## Consequences

- Search-cost regressions can be localized without retaining raw observations.
- Tests can inject deterministic clocks and samplers without coupling the core
  dependency set to a process-monitoring package.
- Memory meaning remains explicit and caller-owned.
- Version 1 artifacts remain usable, while missing historical evidence cannot
  be confused with a measured zero.
- Artifact version 2 remains declarative: no raw rows, callables, or executable
  payloads are added.
