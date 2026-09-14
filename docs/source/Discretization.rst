Preparing features for slice search
===================================

Ginsu searches equality predicates, so continuous features usually need a
small, meaningful set of bins. Fit numeric boundaries and category policies on
the discovery partition, then reuse the fitted plan unchanged on validation or
reference data.

Choose a strategy
-----------------

.. list-table::
   :header-rows: 1
   :widths: 26 36 38

   * - Strategy
     - What it learns
     - Typical use
   * - ``FixedBins``
     - Nothing; uses caller-supplied boundaries
     - Domain thresholds that must stay fixed across datasets
   * - ``EqualWidthBins``
     - Evenly spaced boundaries over the discovery range
     - Features where distance on the original scale matters
   * - ``QuantileBins``
     - Discovery quantiles, with duplicate boundaries removed
     - Skewed numeric features where balanced support is useful
   * - ``CategoryPolicy``
     - Frequent levels and explicit rare/unseen mappings
     - High-cardinality categorical features

Fit once, transform later
-------------------------

.. code:: python

   from ginsu import CategoryPolicy, DiscretizationPlan, QuantileBins

   plan = DiscretizationPlan(
       numeric={"age": QuantileBins(n_bins=8)},
       categorical={"city": CategoryPolicy(max_categories=20)},
   )
   X_discovery_binned = plan.fit_transform(X_discovery)
   X_validation_binned = plan.transform(X_validation)

The plan never accepts errors or targets. This prevents target-aware binning in
the transformation itself, but callers must still fit it only on the intended
discovery partition. It does not make a leaky partition safe. ``metadata_``
records the learned strategies, boundaries, and levels in a Polars frame.

Numeric output is a zero-based ``UInt32`` bin identifier. Equality with a
boundary remains in the lower bin. Duplicate quantile boundaries are removed
deterministically; a constant feature produces one realized bin even when more
were requested.

Category policies retain levels by descending discovery count with lexical
tie-breaking. Observed rare levels map to ``__ginsu_other__`` and values never
seen during fitting map to ``__ginsu_unknown__``. Input values that collide with
either configured sentinel are rejected.

``to_spec()`` emits the fitted plan as closed JSON-compatible data and
``DiscretizationPlan.from_spec()`` reconstructs it with strict validation.
``SliceAnalysis`` uses this representation so discovery and validation
processes apply identical learned transforms after a restart.
