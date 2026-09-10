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

Exact ID recurrence is stricter than semantic or membership similarity. The
current report does not claim that related predicates are matches, and it does
not compare memberships on a common reference population. Those analyses
remain separate planned work.

Resource limits
---------------

``StabilityLimits`` bounds run count, union slice count, and slice-by-run cells
before cross-run tables are materialized. Exceeding a bound raises
``AnalysisLimitError`` with a stable code.

The caller remains responsible for independent sampling units, leakage-safe
preprocessing, temporal ordering, label availability, and whether the number
of successful runs supports a stability claim.
