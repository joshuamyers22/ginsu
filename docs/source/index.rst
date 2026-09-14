Ginsu documentation
===================

.. container:: ginsu-meta

   **Version:** |release|  |  **Python:** 3.10--3.12  |
   **Install a checkout:** ``uv sync --frozen --all-extras``

.. container:: ginsu-summary

   Ginsu finds interpretable subpopulations where a machine-learning model has
   elevated observed loss. It provides a Polars-native workflow for slice
   discovery, fixed-rule validation, stability analysis, comparison, and
   bounded visualization.

.. raw:: html

   <div class="ginsu-grid" role="navigation" aria-label="Documentation sections">
     <a class="ginsu-card" href="GettingStarted.html">
       <span class="ginsu-card-label">Start here</span>
       <strong>Getting started</strong>
       <span>Fit a first finder, inspect its results, and validate fixed rules.</span>
     </a>
     <a class="ginsu-card" href="UserGuide.html">
       <span class="ginsu-card-label">Learn</span>
       <strong>User guide</strong>
       <span>Prepare features and work through validation, selection, stability, and comparison.</span>
     </a>
     <a class="ginsu-card" href="Notebooks.html">
       <span class="ginsu-card-label">Practice</span>
       <strong>Examples</strong>
       <span>Run the offline classification and regression tutorials.</span>
     </a>
     <a class="ginsu-card" href="API.html">
       <span class="ginsu-card-label">Reference</span>
       <strong>API reference</strong>
       <span>Find estimators, result objects, diagnostics, limits, and plotting functions.</span>
     </a>
   </div>

What Ginsu does
---------------

A *slice* is a conjunction of equality predicates, such as
``region == "east" AND tier == "free"``. Given categorical or discretized
features and one nonnegative loss per row, :class:`~ginsu.Slicefinder` searches
for sufficiently supported slices with unusually high observed loss. The
SliceLine score balances slice size and average error, then ranks the returned
rules.

The usual workflow is:

1. Train and evaluate a predictive model outside Ginsu.
2. Prepare one categorical or discretized feature row and one realized loss
   value per observation.
3. Discover and inspect slices on a designated discovery partition.
4. Evaluate the unchanged rules on a genuinely untouched validation partition.
5. Use selection, stability, comparison, artifacts, and plots as auditable
   descriptive evidence.

Ginsu does not train the predictive model or decide how observations should be
split. Discovery scores, minimum support, error lift, stability, and plots are
not statistical significance or causal effects. Optional holdout inference is
for fixed returned rules and does not correct the adaptive search. Read
:doc:`Limitations` before using results in consequential decisions.

Find the right page
-------------------

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Need
     - Documentation
   * - Install Ginsu or choose optional dependencies
     - :doc:`Installing`
   * - Run a complete first analysis
     - :doc:`GettingStarted`
   * - Prepare numeric and categorical features
     - :doc:`Discretization`
   * - Understand accepted input and output containers
     - :doc:`Interoperability`
   * - Check fixed rules on new observations
     - :doc:`Validation`
   * - Reduce a long result to a diverse shortlist
     - :doc:`Selection`
   * - Compare rules across folds, resamples, or time windows
     - :doc:`Stability`
   * - Compare two saved analyses
     - :doc:`Comparison`
   * - Save safe, non-executable analysis evidence
     - :doc:`Artifacts`
   * - Plot impact, overlap, stability, or search diagnostics
     - :doc:`Visualization`
   * - Check signatures, attributes, and defaults
     - :doc:`API`

.. toctree::
   :maxdepth: 2
   :hidden:

   Installing
   GettingStarted
   UserGuide
   Notebooks
   API
