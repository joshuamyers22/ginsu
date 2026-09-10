Holdout validation
==================

Ginsu evaluates discovered rules on untouched observations without searching
again or changing their discovery order. This separates exploratory discovery
metrics from descriptive evidence about whether the same fixed rules recur.

.. code:: python

   validation = finder.validate_slices(
       X_validation,
       validation_errors,
       min_support=0.02,
   )

   print(validation.statistics)
   print(validation.warning_codes)

``X_validation`` must have the exact fitted feature names, order, and dtypes.
Polars is canonical; supported Arrow and pandas-interchange producers pass
through the same boundary. ``validation_errors`` contains one finite,
nonnegative realized loss per row.

Fixed-rule contract
-------------------

``validate_slices`` evaluates every discovered slice exactly once and returns
rows in the original discovery-rank order. It does not refit preprocessing,
discover new rules, remove low-support rules, or rerank results. The caller is
responsible for producing the validation partition and its loss values without
training or preprocessing leakage.

The result places discovery and validation values side by side:

- stable slice ID, display rule, discovery rank, score, support, and error;
- validation support count and fraction;
- validation error sum, maximum, mean, baseline, lift, and excess error;
- mean-error and lift changes from discovery to validation; and
- the predicate count and validation support status.

Support statuses
----------------

``sufficient_support`` means the fixed rule selects at least the declared
validation threshold. ``insufficient_support`` means it selects some rows but
not enough. ``no_members`` means it selects no validation rows. Rules in the
latter two states remain in the table; Ginsu never silently drops them.

An integer ``min_support`` is a row count. A float in ``(0, 1]`` is multiplied
by the validation row count and rounded upward. ``None`` reuses the estimator's
configured discovery threshold under the same rule. Every resolved threshold
is at least one row.

Interpretation boundary
-----------------------

The current result declares ``evidence_kind="holdout_descriptive"``,
``interval_method="none"``, and ``multiplicity_method="none"``. It does not
provide confidence intervals, p-values, corrected significance, or causal
claims. An all-zero validation loss vector is permitted; in that case lift and
lift-delta values are null because the population baseline is zero.

Optional fixed-rule inference
-----------------------------

Inference is opt-in. The default remains descriptive because Ginsu cannot infer
whether a caller's validation partition satisfies the required sampling and
leakage assumptions.

.. code:: python

   from ginsu import ValidationInference

   validation = finder.validate_slices(
       X_validation,
       validation_errors,
       min_support=0.02,
       inference=ValidationInference(
           confidence_level=0.95,
           bootstrap_resamples=2_000,
           permutation_resamples=2_000,
           random_seed=42,
       ),
   )

For rules with at least ``max(2, min_support)`` rows both inside and outside the
slice, this adds:

- a percentile bootstrap interval for mean loss inside the slice;
- the inside-minus-outside mean-loss difference;
- a one-sided, plus-one-corrected permutation p-value for higher loss inside
  the fixed slice;
- a Holm-adjusted p-value across testable returned rules; and
- a significance flag using ``1 - confidence_level``.

``inference_status`` identifies tested rules and rules lacking slice or
complement support. Untestable rules remain visible with null inferential
fields. The bootstrap interval is marginal, not simultaneous or
multiplicity-adjusted.

These procedures assume independent validation rows and, for the permutation
test, exchangeable losses under the null. They do not account for clusters,
repeated entities, time dependence, sample weights, or repeated reuse of the
validation partition. Holm correction covers only testable returned rules. It
does not correct discovery over all searched candidates or make a result causal
or legally dispositive.

Resource limits
---------------

``ValidationLimits`` bounds both the number of fixed slices and the total
row-by-slice membership cells before membership is materialized:

.. code:: python

   from ginsu import ValidationLimits

   validation = finder.validate_slices(
       X_validation,
       validation_errors,
       limits=ValidationLimits(
           max_slices=500,
           max_membership_cells=5_000_000,
           max_resample_work=50_000_000,
           max_resample_batch_cells=500_000,
       ),
   )

The resampling work bound conservatively covers requested rows, slices,
bootstrap draws, and permutations. The batch bound ensures that one bootstrap
draw and every generated index batch remain bounded. Exceeding a limit raises
``AnalysisLimitError`` with a stable diagnostic code.
