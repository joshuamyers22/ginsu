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

## Resolved release inputs

| Input | Decision | Evidence |
|---|---|---|
| Python support | CPython 3.10, 3.11, and 3.12 | `pyproject.toml`; `.github/workflows/release.yml` |
| Operating-system smoke coverage | Core wheel on Linux, macOS, and Windows; extras and sdist on Linux | `.github/workflows/release.yml` |
| Dependency ranges | Core and optional ranges are declared in `pyproject.toml`; the candidate gate uses the committed `uv.lock` | `pyproject.toml`; `uv.lock` |
| Publication posture | Build-only candidates; package-index and GitHub Release publication disabled | `docs/adr/0016-build-only-release-candidates.md` |
