# ADR 0009: Anchor-based stability similarity

- Status: Accepted
- Date: 2026-09-10

## Context

Exact canonical-ID recurrence is intentionally strict. Resampling, changed
search parameters, or different discretization plans can produce a related
rule without reproducing the identical conjunction. Treating those rules as
unrelated hides useful robustness evidence, while transitively clustering
pairwise-similar rules can merge endpoints that are not similar to each other.

Membership comparisons also become invalid if rows from different reference
populations, or different row orders, are aligned positionally without proof.
An empty membership union is not evidence of perfect similarity.

## Decision

Keep exact stability unchanged and primary. Provide a separate
``evaluate_similarity_stability`` report that treats every canonical rule as
an anchor. In every successful run, select its best candidate by:

1. highest Jaccard similarity;
2. lowest discovery rank; and
3. canonical slice ID.

Predicate similarity uses sets of versioned canonical equality-predicate
tokens. Membership similarity uses Boolean selections on a caller-declared
common reference population. Each successful run records the same nonempty
reference identifier, ordered row count, and Ginsu fingerprint of typed,
unique, non-null row identities. The reference feature view may be transformed
differently for each fitted finder, but row identity and order may not differ.

The report retains the best candidate even below the threshold, whether the
exact anchor was selected, candidate metrics, failures, and limit terminations.
Failures remain unavailable rather than false absences. A zero membership union
has null similarity and cannot match. Resource limits apply before reference
membership and anchor/candidate work.

## Consequences

- Exact and related-rule recurrence can be compared without changing stable
  IDs or fitted discovery results.
- The same candidate can answer more than one anchor question. Summary rows are
  not disjoint clusters and must not be summed as unique slice families.
- Predicate similarity measures shared rule syntax, not population overlap.
- Membership similarity measures empirical overlap only on the declared
  reference population and may change with that population.
- Similarity thresholds and stability labels remain descriptive diagnostics,
  not significance, causal equivalence, or correction for discovery search.
- Callers own reference-population semantics and must supply stable row IDs;
  Ginsu verifies their ordered typed fingerprint.
