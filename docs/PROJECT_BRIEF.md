# Ginsu Project Brief

## Outcome

Ginsu helps ML practitioners discover, validate, compare, and visualize
subpopulations where observed model loss is elevated. Polars is the canonical
table interface; compatible Arrow, PyArrow, pandas-interchange, and NumPy inputs
are supported at explicit boundaries.

Success criteria and the staged acceptance evidence are defined in
`POLARS_VISUALIZATION_PRODUCTION_PLAN.md`.

## Constraints

- Preserve correct SliceLine enumeration and scoring unless a separate ADR and
  exhaustive parity evidence approve an algorithm change.
- Reject inputs that violate named-schema, finite-loss, nonnegative-loss, or
  bounded-resource contracts.
- Keep plotting optional and free of import-time side effects.
- Do not present discovery scores as statistical significance.
- Do not change external repository settings, publish a package, or release an
  artifact without explicit authority and completed release evidence.
- Preserve BSD attribution to the inherited implementation.

## Initial critical journeys

1. Fit on a validated categorical/discretized dataset and loss vector.
2. Transform schema-compatible observations into slice membership.
3. Inspect canonical slice rules and impact metrics.
4. Validate fixed discovered rules on untouched observations.
5. Produce bounded, interpretable plots without pandas-specific code.

## Explicit non-goals for the first release

- Hosted service or dashboard.
- Distributed/streaming search.
- Causal or conventional partial-dependence claims.
- Executable pickle as a trusted artifact format.

## Open decisions

| Decision | Owner | Required before |
|---|---|---|
| PyPI publication authority and trusted-publisher environment | Project owner | First release candidate publication |
| Supported Python/OS matrix | Maintainer | Locking the first release gate |
| Exact Polars/Arrow and plotting dependency ranges | Maintainer | Phase 1 merge |
