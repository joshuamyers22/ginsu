Reproducible discretization
===========================

Ginsu searches categorical predicates. Fit numeric boundaries and category
policies on the discovery partition, then reuse the fitted plan unchanged on
validation data.

.. code:: python

   from ginsu import CategoryPolicy, DiscretizationPlan, QuantileBins

   plan = DiscretizationPlan(
       numeric={"age": QuantileBins(n_bins=8)},
       categorical={"city": CategoryPolicy(max_categories=20)},
   )
   X_discovery_binned = plan.fit_transform(X_discovery)
   X_validation_binned = plan.transform(X_validation)

Numeric output is a zero-based UInt32 bin identifier. Boundary equality remains
in the lower bin. Duplicate quantile boundaries are removed deterministically;
a constant feature has one bin.

Category policies retain levels by descending discovery count with lexical
tie-breaking. Observed rare levels map to ``__ginsu_other__`` and values never
observed during fitting map to ``__ginsu_unknown__``. Sentinel collisions are
rejected.

The plan never accepts errors or targets. This prevents target-aware binning in
the transformation itself, but callers must still fit it only on the intended
discovery/training partition. ``metadata_`` records the learned strategy and
boundaries/levels in a Polars frame.

``to_spec()`` emits the fitted plan as closed JSON-compatible data and
``DiscretizationPlan.from_spec()`` reconstructs it with strict validation.
``SliceAnalysis`` uses this representation so discovery and validation
processes apply identical learned transforms after a restart.
