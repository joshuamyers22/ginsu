Ginsu
=====

Ginsu is a Python library for fast slice finding for machine-learning model
debugging.

It is an independent evolution of DataDome's BSD-licensed Sliceline
implementation of `SliceLine: Fast, Linear-Algebra-based Slice
Finding for ML Model
Debugging <https://mboehm7.github.io/resources/sigmod2021b_sliceline.pdf>`__,
from Svetlana Sagadeeva and Matthias Boehm of Graz University of
Technology.

👉 Getting started
------------------

Given an input dataset ``X`` and a model error vector ``errors``,
Ginsu finds the top slices in ``X`` that identify where an ML model
performs significantly worse.

You can use Ginsu as follows:

.. code:: python

   import polars as pl

   from ginsu import Slicefinder
   from ginsu.plotting import plot_error_dependence, plot_impact

   X = pl.DataFrame(
       {
           "region": ["east", "east", "west", "west"],
           "tier": [1, 2, 1, 2],
       }
   )
   errors = [4.0, 3.0, 1.0, 1.0]

   slice_finder = Slicefinder(alpha=0.95, min_sup=1, verbose=False)

   slice_finder.fit(X, errors)

   print(slice_finder.slices_)
   print(slice_finder.slice_statistics_)

   membership = slice_finder.transform(X)
   stable_membership = slice_finder.membership_frame(X)
   impact_figure = plot_impact(slice_finder)
   dependence_figure = plot_error_dependence(
       slice_finder, X, errors, feature="tier"
   )

Evaluate the unchanged discovered rules on a separately prepared holdout
partition before making generalization claims:

.. code:: python

   X_validation = pl.DataFrame(
       {
           "region": ["east", "west"],
           "tier": [1, 2],
       }
   )
   validation_errors = [2.0, 1.0]

   validation = slice_finder.validate_slices(
       X_validation,
       validation_errors,
       min_support=0.02,
   )

   print(validation.statistics)

This first validation contract is descriptive: it preserves discovery order,
labels insufficient support, and does not claim confidence intervals,
significance, or multiplicity correction.

Optional ``ValidationInference`` adds deterministic percentile-bootstrap
intervals and one-sided permutation tests with Holm correction for fixed rules
on a genuinely untouched validation partition. Its assumptions and the limits
of that correction are documented in the holdout-validation guide.

Build a compact view without overwriting raw discoveries:

.. code:: python

   diverse = slice_finder.select_slices(
       X_validation,
       method="diverse",
       k=20,
       max_jaccard=0.80,
   )

The complete decision table retains overlap blockers, capacity exclusions,
empty reference memberships, and incremental coverage for auditability.

Caller-controlled fold, resample, or time-window searches can be captured with
``StabilityRun`` and aggregated with ``evaluate_stability``. Failed and
limit-terminated attempts remain visible rather than being counted as false
slice absences. ``evaluate_similarity_stability`` separately matches related
rules by canonical predicate-set Jaccard or by membership Jaccard on a
verified common reference population; exact identities are never overwritten.
Optional ``plot_stability`` and ``plot_sensitivity`` render recurrence,
run-level metric distributions, parameter response, and unavailable-run
evidence without converting the Polars data contracts to pandas.

Two exhaustive ``SliceAnalysis`` artifacts can be compared with
``compare_analyses``. Exact IDs are matched first, optional related matches are
one-to-one, incompatible feature/discretization semantics remain explicitly
non-comparable, and every emerged or resolved rule stays in the Polars result.
Optional ``plot_comparison`` dumbbell and reference-migration views preserve
those unmatched outcomes and refuse to imply additive attribution across
overlapping slices.

Polars is the canonical table interface. NumPy inputs preserve NumPy outputs;
compatible pandas and PyArrow tables enter through public Arrow or dataframe
interchange protocols and produce Polars outputs. Ginsu production code does
not import pandas.

The ``notebooks/`` directory contains more thorough tutorials:

1. Implementing Ginsu on Titanic dataset
2. Implementing Ginsu on California housing dataset

🛠 Installation
---------------

The ``ginsu`` distribution is not yet published. Install the working checkout
while the first independent release is prepared:

.. code:: sh

   uv sync --frozen --all-extras

Once published, plotting will remain optional and installable with
``ginsu[plot]``.

⚡ Performance Optimization
---------------------------

Ginsu includes optional Numba JIT compilation for scoring operations.

**Quick Installation:**

.. code:: sh

   # With optimization support
   pip install ginsu[optimized]

**Benefits:**

- 5-6x faster scoring operations
- 1.4-4.5x faster overall fit() performance
- Up to 17% memory reduction on large datasets
- Automatic fallback to pure NumPy if Numba not available

**System Requirements:**

Numba requires LLVM to be installed:

.. code:: sh

   # macOS
   brew install llvm

   # Linux (Ubuntu/Debian)
   sudo apt-get install llvm

**Disabling Numba:**

If you need to disable Numba JIT (e.g., in restricted environments), set the environment variable:

.. code:: sh

   export NUMBA_DISABLE_JIT=1

**Docker / Read-only Filesystems:**

Numba requires a writable cache directory. In Docker containers or read-only filesystems,
set ``NUMBA_CACHE_DIR`` to a writable path:

.. code:: dockerfile

   ENV NUMBA_CACHE_DIR=/tmp/numba_cache

If the cache directory is not writable, Ginsu will automatically fall back to pure NumPy.

**Verify Optimization:**

.. code:: python

   from ginsu import is_numba_available

   print("Numba available:", is_numba_available())

See ``benchmarks/`` and ``NUMBA_OPTIMIZATION.md`` for current evidence and its
limitations.

🔗 Useful links
---------------

-  `Ginsu repository <https://github.com/joshuamyers22/ginsu>`__
-  `Issue tracker <https://github.com/joshuamyers22/ginsu/issues>`__
-  `SliceLine paper <https://mboehm7.github.io/resources/sigmod2021b_sliceline.pdf>`__
-  `Upstream Sliceline project <https://github.com/DataDome/sliceline>`__

👐 Contributing
---------------

Feel free to contribute in any way you like, we’re always open to new
ideas and approaches.

Read ``CONTRIBUTING.md`` before proposing or implementing a change.

📝 License
----------

Ginsu is free and open-source software licensed under the 3-clause BSD license.
The retained license notice records the upstream copyright.
