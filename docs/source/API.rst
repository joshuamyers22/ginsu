API reference
=============

Fitted estimator outputs
------------------------

``Slicefinder.fit`` exposes three canonical Polars result tables:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Attribute
     - Contract
   * - ``slices_``
     - One row per ranked rule, keyed by stable ``__ginsu_id``, with the
       display rule and nullable predicate-value columns.
   * - ``slice_statistics_``
     - Discovery score, support, observed loss summaries, lift, excess error,
       and predicate count in matching rank order.
   * - ``predicates_``
     - Long-form, typed predicate records keyed by stable slice ID.

``top_slices_`` and ``top_slices_statistics_`` remain legacy positional
representations. New integrations should use the canonical tables and
``membership_frame``. Table-producing APIs return Polars even when their input
originated in pandas or PyArrow; NumPy input preserves ndarray output for
``transform`` and ``get_slice``.

Stable IDs identify canonical rules, not rows or empirical memberships. Supply
an explicit unique row identifier to ``membership_frame`` when positional row
identity is insufficient.

Core estimator
--------------

.. autoclass:: ginsu.Slicefinder
   :members: fit, transform, get_slice, membership_frame, validate_slices, select_slices, equivalence_groups, overlap_frame, lattice_edges, get_feature_names_out

Holdout validation
------------------

.. autofunction:: ginsu.validate_slices

.. autoclass:: ginsu.SliceValidation
   :members:

.. autoclass:: ginsu.ValidationLimits
   :members:

.. autoclass:: ginsu.ValidationInference
   :members:

Post-selection
--------------

.. autofunction:: ginsu.select_slices

.. autoclass:: ginsu.SliceSelection
   :members:

.. autoclass:: ginsu.SelectionLimits
   :members:

Stability
---------

.. autofunction:: ginsu.evaluate_stability

.. autofunction:: ginsu.evaluate_similarity_stability

.. autoclass:: ginsu.StabilityRun
   :members: from_finder, unavailable

.. autoclass:: ginsu.StabilityReport
   :members:

.. autoclass:: ginsu.SimilarityStabilityReport
   :members:

.. autoclass:: ginsu.StabilityLimits
   :members:

.. autoclass:: ginsu.StabilityReferenceLimits
   :members:

Analysis comparison
-------------------

.. autofunction:: ginsu.compare_analyses

.. autoclass:: ginsu.AnalysisComparison
   :members:

.. autoclass:: ginsu.ComparisonLimits
   :members:

Search diagnostics
------------------

.. autoclass:: ginsu.SearchLimits
   :members:

.. autoclass:: ginsu.SearchReport
   :members:

.. autoclass:: ginsu.SearchLevelReport
   :members:

.. autoclass:: ginsu.SearchStageReport
   :members:

.. autoclass:: ginsu.SearchLimitError

.. autoclass:: ginsu.AnalysisLimitError

Analysis artifacts
------------------

.. autoclass:: ginsu.SliceAnalysis
   :members: from_finder, write, read

.. autoclass:: ginsu.ArtifactLimits
   :members:

.. autoclass:: ginsu.ArtifactError

Slice domain values
-------------------

.. autoclass:: ginsu.Predicate
   :members:

.. autoclass:: ginsu.Slice
   :members:

Discretization
--------------

.. autoclass:: ginsu.DiscretizationPlan
   :members:

.. autoclass:: ginsu.FixedBins

.. autoclass:: ginsu.EqualWidthBins

.. autoclass:: ginsu.QuantileBins

.. autoclass:: ginsu.CategoryPolicy

Plotting
--------

.. autofunction:: ginsu.plotting.plot_impact

.. autofunction:: ginsu.plotting.plot_predicate_matrix

.. autofunction:: ginsu.plotting.plot_error_dependence

.. autofunction:: ginsu.plotting.plot_overlap

.. autofunction:: ginsu.plotting.plot_lattice

.. autofunction:: ginsu.plotting.plot_stability

.. autofunction:: ginsu.plotting.plot_sensitivity

.. autofunction:: ginsu.plotting.plot_comparison

.. autofunction:: ginsu.plotting.plot_search_report
