Comparing analyses
==================

``compare_analyses`` compares two exhaustive ``SliceAnalysis`` artifacts. It
never mutates or rewrites either artifact.

.. code:: python

   from ginsu import compare_analyses

   comparison = compare_analyses(
       baseline_analysis,
       candidate_analysis,
       method="predicate",
       similarity_threshold=0.75,
       direction_tolerance=0.05,
   )

   print(comparison.summary)
   print(comparison.changes)

Matching
--------

Canonical IDs are matched first. With ``method="predicate"``, remaining rules
are compared using Jaccard similarity over their canonical predicate sets.
With ``method="membership"``, remaining rules are compared by Jaccard overlap
on one caller-supplied reference frame. Related matches are one-to-one: Ginsu
orders eligible pairs by highest similarity, baseline rank, candidate rank,
and canonical IDs, then greedily accepts pairs whose endpoints are unused.

This deterministic greedy policy is an audit rule, not a claim that it
maximizes global similarity. Exact, related, and unmatched rules remain
distinguishable. Unmatched baseline rules are ``resolved``; unmatched candidate
rules are ``emerged``.

Direction and deltas
--------------------

Matched rules are ``regressed`` when candidate error lift minus baseline error
lift exceeds ``direction_tolerance``. A negative delta beyond the tolerance is
``improved``; a delta inside the tolerance is ``unchanged``. The summary names
``error_lift`` as the direction metric so that the classification is explicit.

Every change row retains baseline, candidate, and delta values for discovery
rank, SliceLine score, support count, support fraction, error lift, and excess
error. Added, removed, and shared canonical predicates describe rule drift.
These remain discovery metrics rather than statistical significance.

Support counts and excess-error totals may be incomparable when dataset sizes,
exposure, or baselines differ. Slices can overlap, so excess-error deltas must
not be summed as an attribution waterfall without an explicit non-overlapping
allocation policy.

Reference membership migration
------------------------------

Pass the same raw reference population once to calculate empirical migration:

.. code:: python

   comparison = compare_analyses(
       baseline_analysis,
       candidate_analysis,
       method="membership",
       reference_data=X_reference,
       reference_id="sha256:caller-versioned-reference-v1",
   )

Each artifact's recorded discretization plan transforms the reference data
before its rules are evaluated. Without recorded discretization, the reference
schema must exactly match the analysis feature schema. The change table reports
baseline-only, candidate-only, both, neither, union, and Jaccard counts. A zero
union has null Jaccard rather than zero or one.

``reference_id`` is caller-declared provenance. Ginsu does not infer which
columns, ordering, or privacy treatment belong in the identifier.

Compatibility and limits
------------------------

Ordered feature names and dtypes must match. Both artifacts must either omit
discretization or contain identical complete discretization specifications.
Incompatible inputs produce a non-comparable report with an empty, typed change
table, a stable compatibility status and reason, and
``GINSU_COMPARISON_NOT_COMPARABLE``. They are never coerced.

``ComparisonLimits`` bounds slices per analysis, pair comparisons, reference
rows, and reference membership cells. Bounds fail before pairwise or membership
work. The first comparison contract does not yet attach validation/stability
status deltas, aggregate overlap-adjusted attribution, or structured event-time
window metadata; those remain separate work rather than inferred semantics.
