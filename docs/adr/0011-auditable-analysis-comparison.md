# ADR 0011: Auditable one-to-one analysis comparison

- Status: Accepted
- Date: 2026-09-10

## Context

Comparing model versions or time windows requires more than joining ranked
rules by display text. Canonical rules can disappear, related rules can emerge,
and one broad rule can resemble several narrower rules. Many-to-many matching
would double count changes. Comparing memberships on different populations or
silently coercing different discretization semantics would be misleading.

## Decision

Compare only validated exhaustive ``SliceAnalysis`` artifacts. Require equal
ordered feature schemas and either no discretization on both sides or identical
complete discretization specifications. Return an explicit non-comparable
report rather than coercing incompatible analyses.

Match exact canonical IDs first. For predicate or reference-membership modes,
score remaining baseline/candidate pairs with Jaccard similarity, discard pairs
below the declared threshold, and greedily accept unused endpoints in this
order: similarity descending, baseline rank, candidate rank, baseline ID, and
candidate ID. Preserve every unmatched baseline and candidate rule.

Classify matched direction using candidate minus baseline error lift and a
caller-declared nonnegative tolerance. Preserve rank, score, support, lift, and
excess-error values and deltas without treating them as significance. Preserve
added, removed, and shared canonical predicates.

When one raw reference population and caller-declared reference identifier are
supplied, apply each artifact's recorded discretization plan, evaluate its
rules, and report baseline-only, candidate-only, both, neither, union, and
Jaccard membership counts. Zero union is undefined and remains null.

## Consequences

- Matches are deterministic and one-to-one, preventing direct double counting.
- Greedy matching is explainable but is not a maximum-weight assignment.
- Error lift—not score, count, or excess error—defines improved/regressed labels.
- Unmatched rules are explicit ``emerged`` and ``resolved`` rows.
- Reference migration is empirical and specific to the declared population.
- Counts and excess-error deltas are not automatically comparable across
  different exposures, and overlapping slices make their sum non-attributive.
- Validation/stability-status changes, overlap-adjusted attribution, comparison
  plots, and structured time-window metadata remain future contracts.
