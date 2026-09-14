Getting started
===============

This example starts with already-computed per-row model losses, discovers
high-loss subpopulations, inspects the result tables, and evaluates the same
rules on a holdout partition. It uses only synthetic data and runs as part of
the documentation check.

Prepare features and losses
---------------------------

Ginsu expects one row of categorical or discretized features and one finite,
nonnegative loss value for each observation. The loss can be any caller-chosen
row-level model error, such as squared error or log loss.

.. testcode:: getting-started

   import polars as pl

   discovery = pl.DataFrame(
       {
           "region": [
               "east", "east", "east", "east",
               "west", "west", "west", "west",
           ],
           "tier": [
               "free", "free", "pro", "pro",
               "free", "free", "pro", "pro",
           ],
       }
   )
   discovery_loss = [4.0, 3.0, 3.0, 2.0, 1.0, 1.0, 0.5, 0.5]

   assert discovery.height == len(discovery_loss)

Train the predictive model before this step. Do not pass the target or the
prediction itself where Ginsu expects realized loss. For continuous inputs,
fit a :class:`~ginsu.DiscretizationPlan` on discovery features and reuse it on
later partitions; see :doc:`Discretization`.

Fit a slice finder
------------------

.. testcode:: getting-started

   from ginsu import Slicefinder

   finder = Slicefinder(
       alpha=0.90,
       k=5,
       max_l=2,
       min_sup=2,
       verbose=False,
   ).fit(discovery, discovery_loss)

   assert finder.search_report_.status == "complete"
   assert finder.slices_.height == finder.slice_statistics_.height

``alpha`` controls the balance between average slice error and support,
``max_l`` limits predicates per rule, and ``min_sup`` excludes small candidates.
``k`` is a score cutoff rather than an exact row count: tied rules at the cutoff
are retained.

Inspect the results
-------------------

The fitted estimator exposes three canonical Polars tables:

.. testcode:: getting-started

   rules = finder.slices_.select(
       "__ginsu_id", "__ginsu_rank", "__ginsu_rule", "region", "tier"
   )
   statistics = finder.slice_statistics_.select(
       "__ginsu_id",
       "rank",
       "slice_score",
       "support_count",
       "error_lift",
       "excess_error",
   )
   predicates = finder.predicates_

   assert rules.get_column("__ginsu_id").to_list() == statistics.get_column(
       "__ginsu_id"
   ).to_list()
   assert predicates.get_column("__ginsu_id").is_in(
       rules.get_column("__ginsu_id")
   ).all()

``slices_`` is the readable rule table, ``slice_statistics_`` contains discovery
metrics, and ``predicates_`` is a typed long-form representation. Stable IDs
identify canonical predicate sets; display-rule strings are labels, not query
programs.

To evaluate membership while preserving row identity:

.. testcode:: getting-started

   row_ids = pl.Series(
       "case_id",
       ["d-01", "d-02", "d-03", "d-04", "d-05", "d-06", "d-07", "d-08"],
   )
   membership = finder.membership_frame(discovery, row_id=row_ids)

   assert membership.columns[0] == "__ginsu_row"
   assert membership.height == discovery.height

Validate unchanged rules
------------------------

Discovery describes the data that was searched. Use a separately prepared
partition to check whether the unchanged rules retain support and elevated loss:

.. testcode:: getting-started

   validation_features = pl.DataFrame(
       {
           "region": ["east", "east", "west", "west"],
           "tier": ["free", "pro", "free", "pro"],
       }
   )
   validation_loss = [3.0, 2.0, 1.0, 0.5]

   validation = finder.validate_slices(
       validation_features,
       validation_loss,
       min_support=1,
   )

   assert validation.statistics.height == finder.slices_.height
   assert validation.evidence_kind == "holdout_descriptive"

Every discovered rule remains in discovery-rank order, including rules with no
or insufficient holdout members. By default the result is descriptive: it has
no p-values, confidence intervals, or multiplicity claim. Optional fixed-rule
inference and its assumptions are explained in :doc:`Validation`.

Choose the next operation
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Goal
     - Next step
   * - Create a short, less-overlapping result
     - Use :meth:`~ginsu.Slicefinder.select_slices` and read :doc:`Selection`.
   * - Check recurrence across folds or time windows
     - Capture caller-created runs with :class:`~ginsu.StabilityRun`; see
       :doc:`Stability`.
   * - Compare a baseline and candidate model analysis
     - Save exhaustive :class:`~ginsu.SliceAnalysis` objects and read
       :doc:`Comparison`.
   * - Build an impact, overlap, or diagnostics figure
     - Install the ``plot`` extra and read :doc:`Visualization`.
   * - Persist a safe analysis record
     - Use the non-executable format in :doc:`Artifacts`.

Before treating any result as evidence beyond the supplied observations, read
:doc:`Limitations` and define the sampling unit, leakage boundary, validation
population, and loss semantics for the application.
