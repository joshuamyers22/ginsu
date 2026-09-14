Installing Ginsu
================

Ginsu is not yet published on PyPI. Install the checked-out source with the
locked development environment:

.. code-block:: console

   $ git clone https://github.com/joshuamyers22/ginsu.git
   $ cd ginsu
   $ uv sync --frozen

This installs the core library. Ginsu supports CPython 3.10, 3.11, and 3.12.

Optional dependencies
---------------------

Choose extras at the boundary where they are needed:

.. list-table::
   :header-rows: 1
   :widths: 22 35 43

   * - Extra
     - Install in a checkout
     - Purpose
   * - ``plot``
     - ``uv sync --frozen --extra plot``
     - Plotly renderers. Polars plot-data builders do not require it.
   * - ``compat``
     - ``uv sync --frozen --extra compat``
     - Compatibility checks and caller-side pandas/PyArrow objects.
   * - ``optimized``
     - ``uv sync --frozen --extra optimized``
     - Optional Numba acceleration with the same search semantics.
   * - ``notebooks``
     - ``uv sync --frozen --extra notebooks --extra plot``
     - Execute the maintained offline tutorials.
   * - all extras
     - ``uv sync --frozen --all-extras``
     - Reproduce the complete development and documentation environment.

Polars, NumPy, SciPy, and scikit-learn are core dependencies. Ginsu does not
import pandas in production code. Compatible pandas and PyArrow tables enter
through public interchange protocols and still produce Polars result tables;
see :doc:`Interoperability`.

Verify the checkout
-------------------

.. code-block:: console

   $ uv run --frozen python -c "import ginsu; print(ginsu.Slicefinder)"

Install every optional dependency before running the complete repository gate:

.. code-block:: console

   $ uv sync --frozen --all-extras
   $ make check

``make check`` runs formatting, linting, type checking, tests, the warning-free
documentation build, and documentation examples. To build only the HTML guide,
run ``make doc``; output is written to ``docs/build``.

The release workflow currently builds and smoke-tests candidates without
publishing them. The exact matrix and current publication decision are recorded
in ``docs/RELEASE_PROCESS.md`` and ``docs/RELEASE_READINESS.md`` in the source
repository.
