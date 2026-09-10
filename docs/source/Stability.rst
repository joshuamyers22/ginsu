Stability across discovery runs
===============================

Stability analysis aggregates discovery runs produced by caller-controlled
folds, resamples, or time windows. Ginsu does not create those partitions or
refit preprocessing implicitly because only the caller knows the correct
resampling unit and leakage boundary.

.. code:: python

   from ginsu import StabilityRun, evaluate_stability

   runs = []
   for fold in folds:
       finder = make_finder().fit(fold.X_discovery, fold.errors)
       runs.append(
           StabilityRun.from_finder(
               finder,
               run_id=fold.name,
               partition_id=fold.partition_id,
               resampling_unit="customer",
               seed=fold.seed,
           )
       )

   report = evaluate_stability(
       runs,
       minimum_successful_runs=5,
       stable_frequency=0.80,
   )

   print(report.runs)
   print(report.slice_runs)
   print(report.summary)

Run evidence
------------

``StabilityRun.from_finder`` accepts only exhaustive ``complete`` or
``no_valid_slices`` fits. It captures stable IDs, rules, ranks, scores, support
fractions, and error lifts with the run ID, partition ID, resampling unit, seed,
and canonical scalar parameter JSON.

Failures and resource-limit termination must remain visible:

.. code:: python

   failed = StabilityRun.unavailable(
       run_id="fold-7",
       status="limit_reached",
       partition_id="fold-7-train",
       resampling_unit="customer",
       seed=7,
       failure_reason="candidate limit reached",
   )

An unavailable run produces null presence and metrics in ``slice_runs``. It is
not treated as evidence that a slice was absent. A successful
``no_valid_slices`` run does provide true absence evidence.

Exact stability metrics
-----------------------

The initial implementation matches only identical canonical slice IDs. For
each union rule, ``summary`` reports:

- attempted, successful, unavailable, and selected run counts;
- selection frequency among successful runs and among attempted runs;
- rank mean, population standard deviation, minimum, and maximum;
- score, support-fraction, and error-lift means and population standard
  deviations; and
- ``stable``, ``fragile``, or ``insufficient_successful_runs`` status.

The attempted-run frequency is intentionally conservative but mixes missing
results with absence; use the successful-run frequency for exact recurrence and
inspect unavailable runs separately. Stability status never removes a rule.

Similarity-aware stability
--------------------------

Exact ID recurrence remains the primary, unambiguous result. A separate
anchor-based analysis can show whether each canonical rule has a related rule
in another run without relabeling or merging the exact rules:

.. code:: python

   from ginsu import evaluate_similarity_stability

   related = evaluate_similarity_stability(
       runs,
       method="predicate",
       similarity_threshold=0.75,
       minimum_successful_runs=5,
       stable_frequency=0.80,
   )

   print(related.exact.summary)
   print(related.matches)
   print(related.summary)

``predicate`` similarity is Jaccard similarity over canonical equality
predicate sets. For each anchor and run, Ginsu chooses the candidate with the
highest similarity, then the lowest discovery rank, then the canonical ID.
The complete match table retains below-threshold candidates, exact-presence
flags, unavailable evidence, and candidate metrics. The same candidate may be
the best match for more than one anchor: these are independent anchor
questions, not transitive fuzzy clusters.

Membership similarity
---------------------

Membership Jaccard must use the same ordered reference population in every
successful run. Capture it when the run is created:

.. code:: python

   import polars as pl

   reference_rows = pl.Series("account_id", reference_account_ids)
   run = StabilityRun.from_finder(
       finder,
       run_id="fold-1",
       partition_id="fold-1-train",
       resampling_unit="account",
       reference_frame=X_reference_for_this_finder,
       reference_id="sha256:caller-versioned-reference-v1",
       reference_row_id=reference_rows,
   )

   membership = evaluate_similarity_stability(
       runs,
       method="membership",
       similarity_threshold=0.80,
   )

The caller-supplied ``reference_id`` identifies the population and its feature
semantics. Ginsu separately fingerprints the ordered, typed, unique row IDs
and rejects cross-run mismatches. This permits run-specific transformed views
of the same raw population while preventing silent positional misalignment.
Every successful run must carry the same reference metadata; failed and
limit-terminated runs remain unknown.

A membership union of zero has undefined Jaccard similarity. Ginsu records a
null similarity, does not call it a match, and emits
``GINSU_STABILITY_EMPTY_REFERENCE_MEMBERSHIP``. Predicate and membership
similarity are descriptive recurrence diagnostics, not statistical
significance or causal equivalence.

Resource limits
---------------

``StabilityLimits`` bounds run count, union slice count, slice-by-run cells,
and anchor/candidate comparisons before cross-run tables are materialized.
``StabilityReferenceLimits`` bounds reference rows and membership cells before
membership is evaluated. Exceeding a bound raises ``AnalysisLimitError`` with
a stable code.

The caller remains responsible for independent sampling units, leakage-safe
preprocessing, temporal ordering, label availability, and whether the number
of successful runs supports a stability claim.

Visualization
-------------

``ginsu.plotting.plot_stability`` renders exact or similarity-aware recurrence
with a run-evidence matrix. ``ginsu.plotting.plot_sensitivity`` renders a
declared scalar run parameter against observed selection, rank, score, support,
or error lift. Both retain unavailable runs visually and are documented in the
visualization guide.
