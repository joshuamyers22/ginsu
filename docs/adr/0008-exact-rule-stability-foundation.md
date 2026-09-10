# ADR 0008: Exact-rule stability across caller-controlled runs

- Status: Accepted
- Date: 2026-09-10

## Context

A high discovery score or one successful holdout evaluation does not establish
that a slice is reproducible across folds, resamples, or time windows. Ginsu
needs stability evidence without silently choosing resampling units, refitting
preprocessing on the wrong partition, or hiding runs that fail or reach a
resource limit.

Canonical slice IDs provide a strict first recurrence definition: the same
typed predicate conjunction has the same ID. They do not capture semantically
similar rules or membership similarity on a reference population; those are
separate, more permissive comparisons.

## Decision

Callers execute discovery runs under their approved partitioning design and
capture each result as a ``StabilityRun``. Every run records:

- a unique run and partition identifier;
- the declared resampling unit;
- an optional random seed;
- canonical JSON scalar parameters;
- complete or no-valid-slices observations; or
- an explicit failed or limit-reached status and reason.

``evaluate_stability`` aggregates exact canonical IDs across the supplied runs.
It returns run evidence, a slice-by-run table, and per-slice selection frequency
plus rank, score, support-fraction, and error-lift distributions.

Failed and limit-reached runs are unavailable observations. Their presence and
metrics are null, not false. Reports expose selection frequency among successful
runs and separately among all attempted runs. No-valid-slices runs are
successful observations and contribute true absences.

A rule is labeled ``stable`` when its successful-run selection frequency meets
the caller-declared threshold and the minimum successful-run count is met.
Other adequately observed rules are ``fragile``. When too few runs succeed, the
status is ``insufficient_successful_runs`` regardless of observed frequency.

Run count, union slice count, and the slice-by-run cross-product are bounded
before materialization.

## Consequences

The initial stability contract is deterministic, failure-aware, and agnostic to
how folds, resamples, or windows were constructed. It cannot validate the
caller's partitioning design and does not run searches automatically.

This foundation measures exact recurrence only. Predicate-set similarity,
membership similarity on a fingerprinted common reference frame,
hyperparameter sensitivity, clustered/block resampling, and stability plots
remain planned extensions and must not be inferred from these fields.
