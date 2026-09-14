Input data and interoperability
===============================

Polars is Ginsu's canonical named-table container. Use a
``polars.DataFrame`` for the most direct path and for named, typed result
tables. Two-dimensional NumPy arrays and compatible Arrow/dataframe producers
are also accepted at explicit boundaries.

.. list-table::
   :header-rows: 1
   :widths: 28 32 40

   * - Input
     - Normalized as
     - Membership/filter output
   * - ``polars.DataFrame``
     - Polars
     - Polars
   * - two-dimensional NumPy-like array
     - generated ``column_<n>`` names
     - NumPy from ``transform`` and ``get_slice``
   * - compatible PyArrow table
     - Polars through Arrow protocols
     - Polars
   * - compatible pandas DataFrame
     - Polars through public interchange protocols
     - Polars

PyArrow tables and compatible pandas DataFrames enter through their public
Arrow PyCapsule or dataframe-interchange protocols. Ginsu does not import
pandas. These external table producers return Polars outputs by default, so
the behavior is independent of the producer library.

Use external producers at the boundary
--------------------------------------

Pandas and PyArrow belong at the caller boundary, not in the primary notebooks
or Ginsu implementation. The same finder accepts both producers:

.. code:: python

   import pandas as pd
   import pyarrow as pa

   from ginsu import Slicefinder

   arrow_frame = pa.table(
       {"region": ["east", "east", "west", "west"]}
   )
   pandas_frame = arrow_frame.to_pandas(types_mapper=pd.ArrowDtype)
   errors = [4.0, 3.0, 1.0, 1.0]

   arrow_finder = Slicefinder(min_sup=1, verbose=False).fit(
       arrow_frame, errors
   )
   pandas_finder = Slicefinder(min_sup=1, verbose=False).fit(
       pandas_frame, errors
   )

   assert arrow_finder.slices_.equals(pandas_finder.slices_)

This path is exercised in the dedicated compatibility test environment. It
does not introduce a pandas-specific execution branch.

Schema rules
------------

Ginsu records ordered feature names and Polars dtypes during ``fit``. Later
membership and selection calls reject reordered, missing, additional, or
dtype-incompatible columns. Nulls and floating NaNs are rejected. Nested,
Object, Decimal, and timezone-aware columns are not yet supported.

Column names beginning with ``__ginsu_`` are reserved for stable result and
artifact metadata.

Copies and conversion
---------------------

Conversions from external producers may allocate or cast unsupported Arrow
representations. Passing ``allow_copy=False`` to the internal normalization
boundary is therefore deliberately conservative and accepts only an existing
Polars DataFrame. ``SearchReport.copy_boundaries`` records the observed
producer-to-Polars and internal engine-conversion boundaries for each fit; it
does not claim that every boundary necessarily allocated.

Preserve row identity
---------------------

``transform`` retains scikit-learn-style ``slice_0`` columns. For analysis,
``membership_frame`` is the durable interface: it returns ``__ginsu_row`` and
Boolean columns named by stable slice IDs. Supply a unique input column or a
Polars Series as ``row_id`` when positional identity is not sufficient.

Stable IDs identify predicate sets, not datasets or rows. A rule can therefore
have different empirical membership on two populations. Record the reference
population separately when comparing membership across analyses or runs.
