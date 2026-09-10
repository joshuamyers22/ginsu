# ADR 0006: Fixed-rule holdout inference

- Status: Accepted
- Date: 2026-09-09

## Context

ADR 0005 established descriptive evaluation of unchanged discovery rules on an
untouched partition. Users also need uncertainty and a controlled test of
whether validation loss is higher inside a fixed slice than outside it. Ginsu
supports general finite, nonnegative losses, so a Bernoulli-only interval or
test would not cover the public loss contract.

Inference after slice discovery is especially easy to overstate. Testing on
the discovery rows would be selection-biased, and correction across returned
rules cannot correct reuse of the validation partition, model-selection
leakage, or the much larger set of rules considered during discovery.

## Decision

Inference is opt-in through ``ValidationInference``. Descriptive validation
remains the default.

For each fixed rule with at least ``max(2, min_support)`` observations both
inside and outside the slice, Ginsu computes:

- a percentile bootstrap interval for the conditional mean loss inside the
  slice;
- the observed difference between mean loss inside and outside the slice;
- a one-sided Monte Carlo permutation p-value for a greater inside-slice mean,
  using the plus-one correction; and
- a Holm-adjusted p-value across all testable returned rules.

The significance flag compares the Holm-adjusted p-value with
``1 - confidence_level``. Percentile intervals are marginal intervals; they are
not simultaneous or multiplicity-adjusted. Rules without adequate slice or
complement support remain in discovery order with an explicit ``not_tested``
status and null inferential values.

Bootstrap and permutation random streams are deterministically derived from a
recorded seed. All testable rules share each loss permutation. Work,
membership, slice count, and bootstrap batch size are bounded before the
corresponding allocation or resampling work begins.

## Assumptions and interpretation

The validation rows must be independent under the sampling design. The
permutation null requires losses to be exchangeable with respect to fixed slice
membership. The caller must account for clusters, repeated entities, temporal
dependence, survey weights, delayed labels, and any training or preprocessing
leakage; the current procedure does not model them.

Holm correction controls the family consisting only of testable returned rules
on this one untouched validation use. It does not repair discovery-set
selection bias, correct across all enumerated candidates, or preserve nominal
error rates after repeated inspection and tuning on the holdout partition.
Results describe predictive error association, not causality or a fairness or
legal conclusion.

## Consequences

Ginsu can now attach reproducible uncertainty and corrected fixed-rule tests to
holdout metrics while keeping unsupported claims mechanically distinguishable.
The default remains descriptive because valid inference depends on a data
partition and sampling design the library cannot infer.

Cluster/block bootstrap, time-aware tests, full-search permutation, stability
analysis, and alternate multiple-testing procedures require separate reviewed
contracts and simulation evidence.
