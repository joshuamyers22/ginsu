Polars and Arrow interoperability
=================================

Polars is Ginsu's canonical named-table container. Pass a
``polars.DataFrame`` for native operation. A two-dimensional NumPy array is
also supported and retains ndarray outputs from ``transform`` and
``get_slice``.

PyArrow tables and compatible pandas DataFrames enter through their public
Arrow PyCapsule or dataframe-interchange protocols. Ginsu does not import
pandas. These external table producers return Polars outputs by default, so
the behavior is independent of the producer library.

Schema contract
---------------

Ginsu records ordered feature names and Polars dtypes during ``fit``. Later
membership and selection calls reject reordered, missing, additional, or
dtype-incompatible columns. Nulls and floating NaNs are rejected. Nested,
Object, Decimal, and timezone-aware columns are not yet supported.

Column names beginning with ``__ginsu_`` are reserved for stable result and
artifact metadata.

Copy policy
-----------

Conversions from external producers may allocate or cast unsupported Arrow
representations. Passing ``allow_copy=False`` to the internal normalization
boundary is therefore deliberately conservative and accepts only an existing
Polars DataFrame. Public copy diagnostics will be added with ``SearchReport``.

Membership
----------

``transform`` retains scikit-learn-style ``slice_0`` columns. For analysis,
``membership_frame`` is the durable interface: it returns ``__ginsu_row`` and
Boolean columns named by stable slice IDs. Supply a unique input column or a
Polars Series as ``row_id`` when positional identity is not sufficient.
