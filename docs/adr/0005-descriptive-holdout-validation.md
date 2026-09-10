# ADR 0005: Fixed-rule descriptive holdout validation

- Status: Accepted
- Date: 2026-09-09

## Context

Slice discovery selects rules because their observed loss is unusual on the
discovery partition. Reusing those same observations as evidence of
generalization produces selection bias. Ginsu therefore needs an explicit
validation boundary before adding confidence intervals, multiple-comparison
procedures, stability claims, or validation-aware plots.

The core loss contract supports general finite, nonnegative realized losses,
not only Bernoulli errors. A first validation API must not silently choose an
inferential procedure whose assumptions are unsuitable for some supported
losses.

## Decision

Ginsu provides ``validate_slices`` as a fixed-rule, descriptive holdout
operation:

- It evaluates every discovered rule in original discovery-rank order.
- It does not search, select, filter, rerank, refit, or modify the estimator.
- Validation observations pass through the same Polars-native schema boundary
  as transformation.
- Results retain discovery metrics beside validation support, mean loss, lift,
  excess loss, and discovery-to-validation deltas.
- ``sufficient_support``, ``insufficient_support``, and ``no_members`` are
  explicit statuses. Low-support rules remain visible.
- A zero validation baseline is valid; ratio metrics are null rather than
  infinite or undefined.
- Integer and fractional minimum support use an explicit validation-partition
  threshold, with fractional thresholds rounded upward.
- Slice count and row-by-slice membership cells are bounded before membership
  materialization.
- The result declares ``holdout_descriptive`` evidence with interval and
  multiplicity methods set to ``none``. It makes no significance claim.

The caller owns partition construction and must supply loss values produced
without training or preprocessing leakage from the validation observations.

## Consequences

The initial API supports honest out-of-sample descriptive comparison without
pretending that selected slices have inferential guarantees. Downstream plots
can distinguish discovery metrics from holdout metrics and visibly retain
failed support checks.

Intervals, resampling, permutation tests, multiplicity correction, and
stability analysis require separate reviewed methods and simulation evidence.
When added, they will extend the validation result rather than reinterpret
descriptive fields.
