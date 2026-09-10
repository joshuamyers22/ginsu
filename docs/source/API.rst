API reference
=============

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

Search diagnostics
------------------

.. autoclass:: ginsu.SearchLimits
   :members:

.. autoclass:: ginsu.SearchReport
   :members:

.. autoclass:: ginsu.SearchLevelReport
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
