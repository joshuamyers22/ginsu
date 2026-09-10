Limitations and interpretation boundaries
=========================================

Ginsu discovers and ranks subpopulations with elevated observed model loss. It
is a diagnostic component, not an autonomous decision system. The following
boundaries are part of the current ``0.1`` contract.

Statistical and causal limits
-----------------------------

- Discovery scores and minimum support are not statistical significance.
- Error lift and excess error are descriptive associations with the supplied
  loss vector. They do not identify causes or intervention effects.
- Fixed-rule holdout inference assumes independent rows and exchangeable losses
  under its permutation null. Its Holm correction covers only testable returned
  rules, not the original adaptive search or repeated holdout use.
- Stability summarizes caller-supplied runs. Ginsu does not create independent
  folds, resamples, clusters, or time windows on the caller's behalf.
- Model-comparison and membership-migration values are per rule. Overlapping
  slices make sums across rules non-attributive.
- Results do not establish fairness, discrimination, safety, privacy, legal
  compliance, or fitness for deployment.

Data and preprocessing limits
-----------------------------

- Each loss must be finite and nonnegative, and discovery requires at least one
  positive value. Sample weights and alternate baseline definitions are not
  supported.
- Ginsu searches equality predicates. Continuous features should normally be
  discretized with boundaries learned without target or loss supervision.
- Input must be nonempty and two-dimensional. Feature names must be unique and
  cannot start with the reserved ``__ginsu_`` prefix.
- Nulls and floating NaNs are rejected. Ginsu does not choose an imputation or
  missingness policy.
- Nested, Polars Object, Decimal, and timezone-aware datetime columns are not
  supported for named-table inputs.
- Later transforms require exactly the fitted feature names, order, and dtypes.
  Unknown categorical values are ignored by fitted membership; a
  ``DiscretizationPlan`` can instead map them to its explicit unknown sentinel.
- ``LazyFrame`` input is not accepted because implicit collection would hide an
  execution and memory boundary.

Producer and output limits
--------------------------

Polars is the canonical named-table model. Compatible pandas and PyArrow inputs
enter through public dataframe-interchange or Arrow protocols and may require a
copy. Ginsu does not promise zero-copy conversion for external producers.
Named-table operations return Polars rather than preserving the producer type.
NumPy-like input preserves ndarray output only for ``transform`` and
``get_slice``.

The ordered schema is intentional. Reordering compatible-looking columns is an
error rather than an implicit correction, because positional engine conversion
would otherwise change rule meaning.

Search and resource limits
--------------------------

Search cost can grow rapidly with feature cardinality, lattice depth, and
candidate ties. Conservative ``SearchLimits`` bound encoded width, pair-matrix
bytes, per-level and total candidates, elapsed time, and returned ties. A limit
raises ``SearchLimitError`` and leaves search diagnostics, but not a fitted
estimator.

The current engine converts validated Polars data through NumPy and SciPy sparse
structures. It is neither distributed nor streaming. Optional Numba accelerates
scoring but does not change semantics. Boundary memory samples are caller
provided; they are not allocation attribution or a continuous peak unless the
sampler itself has those semantics.

Validation, selection, overlap, stability, comparison, and plotting each have
separate bounds on rows, slices, cells, pair comparisons, or resampling work.
Increasing a bound is an explicit resource decision, not a guarantee that a
workload will fit available memory or time.

Identity and artifact limits
----------------------------

Stable slice IDs identify canonical predicate sets. They do not identify a
dataset, model, partition, or empirical membership. Membership comparison needs
a caller-identified common reference population with stable, unique row IDs.

``SliceAnalysis`` contains result tables, fitted discretization metadata, and
execution evidence. It deliberately excludes raw observations, per-row
membership, trained models, arbitrary Python objects, and executable pickle.
Dataset and partition fingerprints are caller supplied because Ginsu cannot
infer governance or privacy scope. Unknown artifact versions fail closed.

Plotting and reporting limits
-----------------------------

Plotly is optional. Renderers cap slices, points, nodes, stages, features, and
cells before materialization. Deterministic point sampling affects rendered raw
points, not grouped summaries. Display rule strings and hover labels are for
people; they are not executable filters or durable serialization.

Observed error-dependence plots are not conventional partial-dependence plots:
they do not intervene on a feature or query a predictive model. Rule lattices
show logical predicate refinement, not temporal transitions. Search profiles
describe computation and resource pressure, not model quality.

Versioning limits
-----------------

Ginsu is establishing a pre-1.0 API. Canonical schemas and serialized formats
are versioned and tested, but future pre-1.0 releases may require a documented
migration. The project does not provide a ``sliceline`` import alias. Python
3.13, sample weighting, alternate baselines, distributed execution, and a
hosted dashboard are not currently supported claims.
