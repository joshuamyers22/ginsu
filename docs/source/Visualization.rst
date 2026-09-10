Visualization
=============

Install the optional Plotly renderer with ``ginsu[plot]`` once the package is
published. In a source checkout, ``uv sync --frozen --all-extras`` installs it.
Plot-data construction remains Polars-only and does not require Plotly.

Impact plot
-----------

.. code:: python

   from ginsu.plotting import plot_impact

   figure = plot_impact(finder)
   figure.show()

Marker position shows observed error lift relative to the population baseline,
marker size shows support, and color shows the SliceLine score. The dashed
reference at lift 1 indicates population-average error.

Predicate matrix
----------------

.. code:: python

   from ginsu.plotting import plot_predicate_matrix

   figure = plot_predicate_matrix(finder, max_slices=100)
   figure.show()

Each row is a ranked slice, each column is an input feature, and filled cells
show the predicate value used by that rule. Support, error lift, and score stay
available in the hover detail. Cell and slice bounds are checked before the
rectangular Polars table is materialized.

Observed error-dependence plot
------------------------------

.. code:: python

   from ginsu.plotting import plot_error_dependence

   figure = plot_error_dependence(
       finder,
       X_validation,
       validation_errors,
       feature="age",
       max_points=5_000,
       seed=42,
   )
   figure.show()

The plot compares observed per-row model loss against a feature, inside and
outside a fixed discovered slice. Numeric features use raw points plus binned
means; categorical features use grouped box plots. Summaries use every row,
while ``max_points`` only limits deterministically sampled rendered points.

This is descriptive error dependence. It is not a causal estimate and is not
conventional model partial dependence. Use untouched validation errors when
making generalization claims; discovery-set lift is exploratory.

Overlap and equivalence
-----------------------

Use the stable-ID Polars tables directly when building a report or a custom
renderer:

.. code:: python

   groups = finder.equivalence_groups(X_validation)
   overlap = finder.overlap_frame(X_validation, max_slices=100)

``groups`` combines rules that select exactly the same validation rows.
``overlap`` contains the full symmetric pair matrix, intersection and union
counts, exact-equivalence flags, and Jaccard similarity. A zero union is
represented by a null Jaccard value rather than zero or one.

.. code:: python

   from ginsu.plotting import plot_overlap

   figure = plot_overlap(finder, X_validation, max_slices=100)
   figure.show()

The default renderer is a heatmap because pairwise values remain directly
comparable. Slice, cell, and membership limits are checked before pairwise or
row-by-slice materialization. Raising those limits is an explicit resource
decision, not an automatic fallback.

Rule-refinement lattice
-----------------------

.. code:: python

   edges = finder.lattice_edges(max_nodes=100)

   from ginsu.plotting import plot_lattice

   figure = plot_lattice(finder, max_nodes=100)
   figure.show()

An edge exists only when the child contains every parent predicate and adds
exactly one more. Vertical levels are predicate counts; node size is support;
node color is observed error lift. These edges describe logical rule
refinement, not causality, dependence, or observed transitions.
