Ginsu
=====

**Fast, interpretable slice finding for machine-learning model debugging.**

Ginsu finds subpopulations where a model has elevated observed loss. A slice
is an interpretable conjunction of equality predicates, such as
``region == "east" AND tier == "free"``. Results are Polars tables with stable
rule identities, explicit resource diagnostics, and workflows for holdout
validation, selection, stability, comparison, artifacts, and visualization.

Ginsu is an independent evolution of DataDome's BSD-licensed Sliceline
implementation of `SliceLine: Fast, Linear-Algebra-based Slice Finding for ML
Model Debugging
<https://mboehm7.github.io/resources/sigmod2021b_sliceline.pdf>`__, by Svetlana
Sagadeeva and Matthias Boehm.

Quick start
-----------

Ginsu takes categorical or discretized features and one finite, nonnegative
realized model loss per row:

.. code-block:: python

   import polars as pl

   from ginsu import Slicefinder

   discovery = pl.DataFrame(
       {
           "region": ["east", "east", "east", "east",
                      "west", "west", "west", "west"],
           "tier": ["free", "free", "pro", "pro",
                    "free", "free", "pro", "pro"],
       }
   )
   discovery_loss = [4.0, 3.0, 3.0, 2.0, 1.0, 1.0, 0.5, 0.5]

   finder = Slicefinder(
       alpha=0.90,
       k=5,
       max_l=2,
       min_sup=2,
       verbose=False,
   ).fit(discovery, discovery_loss)

   finder.slices_             # ranked rules with stable IDs
   finder.slice_statistics_   # score, support, lift, and excess error
   finder.predicates_         # typed long-form predicates
   finder.search_report_      # execution status, timing, and active limits

Train and evaluate the predictive model before calling Ginsu. Pass per-row
loss—not labels or predictions—and fit any discretization only on the intended
discovery partition.

Validate fixed rules
--------------------

Discovery metrics describe the observations that were searched. Evaluate the
unchanged rules on a separately prepared holdout partition before making a
generalization claim:

.. code-block:: python

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
   validation.statistics

Validation preserves discovery order and retains rules with insufficient or no
holdout support. The default result is descriptive. Optional percentile
bootstrap intervals and one-sided permutation tests apply only to fixed returned
rules under their documented assumptions; they do not correct the adaptive
search or repeated holdout use.

Choose a workflow
-----------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Goal
     - Operation
   * - Prepare continuous or high-cardinality inputs
     - Fit ``DiscretizationPlan`` on discovery features, then reuse
       ``transform`` on validation features.
   * - Preserve row identity while evaluating rules
     - Use ``finder.membership_frame(data, row_id=...)``.
   * - Create a short, auditable result
     - Use ``finder.select_slices(..., method="diverse")`` without changing
       the raw ranking.
   * - Check recurrence across folds, resamples, or time windows
     - Capture caller-controlled ``StabilityRun`` objects and aggregate them
       with ``evaluate_stability`` or ``evaluate_similarity_stability``.
   * - Compare a baseline and candidate model
     - Create exhaustive ``SliceAnalysis`` artifacts and call
       ``compare_analyses``.
   * - Plot impact, overlap, stability, comparison, or search cost
     - Install the ``plot`` extra and use ``ginsu.plotting``.
   * - Save non-executable analysis evidence
     - Write a ``SliceAnalysis`` directory containing canonical JSON and Arrow
       IPC tables; raw observations are excluded.

Interpretation
--------------

Discovery score, minimum support, error lift, excess error, selection,
stability, comparison, and plots are descriptive diagnostics. They are not
statistical significance, causal effects, fairness certification, or deployment
approval. Slices may overlap, so per-rule error totals cannot be added into an
attribution waterfall without a separate allocation rule.

See the `limitations guide <docs/source/Limitations.rst>`__ for the complete
statistical, data, resource, artifact, and reporting boundaries.

Inputs and outputs
------------------

Polars is the canonical named-table interface. Compatible pandas and PyArrow
tables enter through public Arrow or dataframe-interchange protocols; Ginsu
does not import pandas in production. Named-table operations return Polars.
Two-dimensional NumPy input remains supported, with ndarray output from
``transform`` and ``get_slice``.

Fitted feature names, order, and dtypes are enforced on later calls. Nulls,
floating NaNs, implicit LazyFrame collection, and several nested or ambiguous
dtypes are rejected rather than silently transformed. See the
`interoperability guide <docs/source/Interoperability.rst>`__.

Installation
------------

Ginsu is not yet published on PyPI. From a checked-out repository, install the
locked core environment with:

.. code-block:: console

   $ uv sync --frozen

Use ``uv sync --frozen --all-extras`` for the complete development environment.
The optional extras are ``plot`` for Plotly, ``optimized`` for Numba,
``compat`` for pandas/PyArrow compatibility checks, and ``notebooks`` for the
maintained tutorials. Ginsu supports CPython 3.10 through 3.12.

Documentation
-------------

The documentation follows a task-oriented guide/reference split:

- `Getting started <docs/source/GettingStarted.rst>`__
- `User guide <docs/source/UserGuide.rst>`__
- `Examples and notebooks <docs/source/Notebooks.rst>`__
- `API reference <docs/source/API.rst>`__
- `Migration guide <docs/source/Migration.rst>`__

Build the warning-free site and execute its examples with:

.. code-block:: console

   $ make doc
   $ make doctest

The maintained notebooks are deterministic, synthetic, and network-independent.
Run them with ``make execute-notebooks``.

Development
-----------

Read `CONTRIBUTING.md <CONTRIBUTING.md>`__ before proposing a change.
``make check`` runs formatting, linting, type checking, tests, the documentation
build, and documentation examples. Security reports should follow
`SECURITY.md <SECURITY.md>`__.

Useful links
------------

- `Source repository <https://github.com/joshuamyers22/ginsu>`__
- `Issue tracker <https://github.com/joshuamyers22/ginsu/issues>`__
- `SliceLine paper <https://mboehm7.github.io/resources/sigmod2021b_sliceline.pdf>`__
- `Upstream Sliceline project <https://github.com/DataDome/sliceline>`__
- `Release process <docs/RELEASE_PROCESS.md>`__
- `Release readiness <docs/RELEASE_READINESS.md>`__

License
-------

Ginsu is free and open-source software licensed under the 3-clause BSD license.
The retained license notice records the upstream copyright.
