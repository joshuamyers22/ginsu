Migration guide
===============

Ginsu is an independent package, not a drop-in distribution replacement for
Sliceline. Migration is deliberately explicit: install ``ginsu`` and change
imports to ``ginsu``. There is no ``sliceline`` compatibility package or import
alias.

Minimal Polars migration
------------------------

The estimator parameters ``alpha``, ``k``, ``max_l``, ``min_sup``, and
``verbose`` retain their familiar search roles. The primary change is to use a
named Polars frame and canonical result tables:

.. testcode:: migration

   import polars as pl

   from ginsu import Slicefinder

   discovery = pl.DataFrame(
       {
           "region": ["east", "east", "west", "west"],
           "tier": [1, 2, 1, 2],
       }
   )
   discovery_loss = [4.0, 3.0, 1.0, 1.0]

   finder = Slicefinder(
       alpha=0.95,
       k=3,
       max_l=2,
       min_sup=1,
       verbose=False,
   ).fit(discovery, discovery_loss)

   assert isinstance(finder.slices_, pl.DataFrame)
   assert isinstance(finder.slice_statistics_, pl.DataFrame)
   assert isinstance(finder.predicates_, pl.DataFrame)
   assert finder.slices_.height > 0

The loss vector must contain one finite, nonnegative realized model loss per
row, and at least one loss must be positive. Train the predictive model before
slice discovery; do not pass labels or predictions where per-row loss is
required.

Imports and result attributes
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 32 38

   * - Previous idiom
     - Ginsu idiom
     - Reason
   * - ``from sliceline import Slicefinder``
     - ``from ginsu import Slicefinder``
     - Ginsu owns an independent package and version history.
   * - ``finder.top_slices_``
     - ``finder.slices_``
     - Typed Polars rows include stable IDs, rank, display rules, and named
       predicate columns.
   * - ``finder.top_slices_statistics_``
     - ``finder.slice_statistics_``
     - The canonical schema has typed support, loss, lift, excess-error, and
       predicate-count columns.
   * - positional ``slice_<n>`` membership
     - ``finder.membership_frame(X, row_id=...)``
     - Stable rule IDs and explicit row identity survive reranking and joins.
   * - manual Boolean filtering
     - ``finder.get_slice(X, index)``
     - Membership uses the fitted schema and rule representation.
   * - treating top ``k`` as exactly ``k`` rows
     - inspect ``finder.slices_.height``
     - Score ties may return more than ``k`` rules.

The two legacy result attributes currently remain available for inherited
algorithm parity and NumPy-oriented inspection. They are positional and should
not be persisted as Ginsu's application contract. Use ``SliceAnalysis`` for a
validated, non-executable artifact.

.. testcode:: migration

   membership = finder.membership_frame(discovery)
   stable_ids = finder.slices_.get_column("__ginsu_id").to_list()

   assert membership.columns == ["__ginsu_row", *stable_ids]
   assert finder.get_slice(discovery, 0).height == 1

Migrating pandas-oriented code
------------------------------

Prefer constructing Polars at the application boundary:

.. code:: python

   import polars as pl

   discovery = pl.read_parquet("governed-discovery.parquet")
   discovery = discovery.select("region", "tier")

If upstream code must produce pandas, pass a dataframe-interchange-compatible
DataFrame directly. PyArrow tables are accepted through Arrow protocols. Ginsu
does not import either producer, and named-table outputs are still Polars:

.. code:: python

   pandas_finder = Slicefinder(verbose=False).fit(pandas_frame, losses)
   assert isinstance(pandas_finder.slices_, pl.DataFrame)
   assert isinstance(pandas_finder.transform(pandas_frame), pl.DataFrame)

In a source checkout, install the producer dependencies with
``uv sync --frozen --extra compat``. After publication, the equivalent will be
``pip install "ginsu[compat]"``. Use this extra only in environments that create
pandas or PyArrow objects. It supports producer integration tests; it does not
activate a different Ginsu execution path.

Common dataframe operations map as follows:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - pandas-oriented operation
     - Polars/Ginsu operation
   * - ``frame.loc[mask]``
     - ``frame.filter(mask)`` or ``finder.get_slice(frame, index)``
   * - joining positional membership columns
     - join ``membership_frame`` using ``__ginsu_row`` or a caller-supplied
       unique row ID
   * - ``pandas.cut`` fitted independently per partition
     - fit one ``DiscretizationPlan`` on discovery and reuse ``transform`` on
       validation
   * - ``frame.query`` with rendered rule strings
     - use ``get_slice`` or stable-ID membership; display rules are not query
       programs
   * - converting results with ``to_pandas`` immediately
     - keep Polars through analysis and convert only at the consumer boundary

Schema behavior is stricter than many pandas workflows. Feature names, order,
and dtypes captured by ``fit`` must match later calls. Collect a Polars
``LazyFrame`` explicitly. Resolve nulls and floating NaNs under an application
policy before calling Ginsu; implicit imputation is not performed.

Numeric and category preprocessing
----------------------------------

Ginsu searches equality predicates over categorical or discretized features.
Do not let near-unique continuous values become accidental categories. Fit an
error-independent plan on the discovery features, then apply it unchanged to
validation features:

.. code:: python

   from ginsu import CategoryPolicy, DiscretizationPlan, QuantileBins

   plan = DiscretizationPlan(
       numeric={"age": QuantileBins(n_bins=8)},
       categorical={"region": CategoryPolicy(max_categories=20)},
   )
   discovery_binned = plan.fit_transform(discovery_raw)
   validation_binned = plan.transform(validation_raw)

The plan never accepts targets or losses. This prevents error-supervised
boundaries inside the plan, but the caller must still preserve partition and
preprocessing isolation.

NumPy migration
---------------

Two-dimensional NumPy-like input remains supported. ``transform`` and
``get_slice`` then return ndarrays, and result rules use generated feature names
such as ``column_0``. Canonical result tables and ``membership_frame`` are still
Polars. Migrate to a named Polars frame when schema diagnostics, durable IDs, or
cross-run artifacts matter.

Interpretation changes
----------------------

Minimum support removes small candidates but does not establish statistical
significance. Discovery score, error lift, plots, overlap, stability, and model
comparison are descriptive. Evaluate unchanged rules on a genuinely untouched
partition with ``validate_slices`` before making generalization claims. Its
optional inference applies only to fixed returned rules under the documented
sampling assumptions; it does not correct the original search.

Review the :doc:`limitations reference <Limitations>` before using slice
results for deployment, fairness, policy, or other consequential decisions.
