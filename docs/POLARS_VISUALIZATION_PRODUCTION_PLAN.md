# Ginsu Polars and Visualization Production Plan

**Status:** In progress
**Prepared:** 2026-09-09
**Reviewed repository:** `DataDome/sliceline` at commit
`356f71a9ecaa6b97e299e341ebc9424cf7b88c4a`
**Reference standard:** `production-project-template`, especially
`standards/PRODUCTION_REPOSITORY_STANDARD.md`,
`docs/PYTHON_ENGINEERING_GUIDE.md`, and
`checklists/RELEASE_READINESS.md`
**Decision owner:** `@joshuamyers22`
**Implementation owner:** Ginsu contributors

### Implementation snapshot

As of 2026-09-09, the independent `ginsu` package name, frozen lock, core
correctness fixes, Polars/Arrow boundary, immutable slice domain values,
canonical result frames, impact plot, and observed error-dependence plot are
implemented locally with focused tests. Cross-producer parity, stable-ID
membership frames, a mypy gate, and offline warnings-as-errors documentation
are also implemented. The fitted, error-independent Polars discretization
foundation is implemented for fixed, equal-width, quantile, and categorical
policies. The search-diagnostics/resource-limit foundation, predicate-matrix,
exact-equivalence and Jaccard-overlap data, lattice edges, and bounded Plotly
composition/overlap/lattice views are also implemented. Automatic publication
remains disabled. The initial versioned, non-executable ``SliceAnalysis``
artifact with fitted-discretizer round trips and bounded fail-closed loading is
implemented. Later-phase validation, comparison, notebook migration, and
release hardening remain open.

## 1. Executive summary

Build Ginsu as a Polars-native model-debugging library and add a
production-quality visualization layer without replacing the sparse
linear-algebra algorithm that makes the package useful.

The implementation should:

1. Make `polars.DataFrame` the canonical named-table interface, with supported
   Arrow ingestion for PyArrow and pandas inputs.
2. Preserve NumPy, SciPy, scikit-learn, and optional Numba as the internal
   compute engine unless benchmarks justify a different engine.
3. Remove pandas from the implementation, maintained examples, and required
   dependencies while supporting pandas through Arrow/dataframe interchange
   rather than a pandas-specific code path.
4. Expose typed Polars result tables for slices, predicates, statistics,
   membership, equivalence groups, and overlap.
5. Add impact, predicate-matrix, overlap, lattice, and observational
   error-dependence plots through an optional Plotly adapter.
6. Correct current input-contract and scoring edge cases before changing the
   public data interface.
7. Bring applicable repository, CI, release, security, typing, packaging, and
   reproducibility controls up to the local production-project-template
   standard.
8. Add reproducible feature discretization, out-of-sample validation,
   uncertainty/stability analysis, and diversity-aware selection so plots do
   not overstate in-sample discoveries.
9. Add bounded resource controls, structured search diagnostics, comparative
   model/time analysis, and a versioned analysis artifact.

This is not a rewrite of SliceLine in Polars. Polars is the canonical table and
analysis layer; Arrow-compatible producers feed that layer; NumPy/SciPy sparse
arrays remain the bounded compute boundary. That division avoids turning a
dataframe preference into a regression in the algorithm's sparse-memory
behavior.

## 2. Outcomes and success measures

### 2.1 User outcomes

- A user can fit, transform, inspect, filter, and visualize slices while
  keeping named data in Polars.
- A user does not need pandas to install, run, document, test, or plot
  Ginsu.
- A user with a pandas or PyArrow table can use the same public API through a
  documented Arrow/interchange boundary without Ginsu importing pandas.
- Results explain both slice importance and slice composition.
- Equivalent or heavily overlapping slice rules are visible rather than
  presented as independent discoveries.
- Dependence plots clearly describe observed error relationships and are not
  mislabeled as causal or as partial-dependence estimates.
- Existing NumPy/scikit-learn users receive a documented compatibility path.
- Users can distinguish discovery performance from out-of-sample validated
  performance, uncertainty, and stability.
- Users can compare failure slices across model versions and time windows.

### 2.2 Engineering outcomes

- The current 17 experiment fixtures produce equivalent ranked rules and
  statistics through NumPy, Polars, PyArrow, and pandas-via-interchange inputs.
- Named input columns cannot be silently reordered, omitted, duplicated, or
  added at transform time.
- Score inputs satisfy explicit finite, nonnegative, and positive-baseline
  invariants.
- The built wheel passes a clean, pandas-free smoke test.
- Binning specifications, canonical slices, search diagnostics, validation
  evidence, and provenance can be saved in a versioned analysis artifact.
- The standard quality gate covers lint, format, typing, tests, coverage,
  documentation, build, artifact inspection, dependency audit, and wheel smoke
  tests.
- Release workflows use locked dependencies, immutable action references,
  least-privilege permissions, attestable artifacts, checksums, and an SBOM.

### 2.3 Guardrails

- No change to the SliceLine score formula, lattice enumeration, tie behavior,
  or pruning semantics without a separate algorithm decision record and
  numerical evidence.
- No implicit collection of `polars.LazyFrame` in the initial release.
- No dense row-by-slice or slice-by-slice allocation without a documented
  bound, estimate, and explicit override.
- No plotting dependency in the minimal computational installation.
- No runtime telemetry, network request, notebook execution, or file write from
  importing `ginsu`.
- No silent dataframe-to-array conversion that loses column identity, null
  semantics, or dtype information.
- No claim of statistical significance from the discovery score alone.
- No supervised binning, slice selection, or hyperparameter tuning outside the
  training/discovery partition used to learn it.
- No input adapter may use private dataframe internals or rely on pandas being
  importable.

## 3. Scope

### 3.1 In scope

- Polars input and type-preserving output contracts.
- Arrow C Data/PyCapsule and dataframe-interchange ingestion for compatible
  PyArrow and pandas tables.
- A typed, tabular result model.
- Derived impact and overlap metrics.
- Reproducible Polars-native discretization.
- Statistical validation, multiple-search qualification, stability analysis,
  and diversity-aware result selection.
- Model-version and time-window comparison.
- Immutable slice/predicate domain objects and versioned analysis artifacts.
- Search reports and explicit resource limits.
- Sample weights and alternate baseline research behind a separate score ADR.
- Plot-data builders and Plotly rendering.
- Migration of both example notebooks from pandas to Polars.
- Input validation and public scikit-learn compatibility repairs needed by the
  migration.
- Tests, benchmarks, documentation, packaging, CI, release controls, and
  applicable repository governance.

### 3.2 Out of scope for the first release

- Reimplementing sparse lattice enumeration with Polars expressions.
- Distributed or streaming slice search.
- A hosted dashboard or server.
- Causal inference or conventional model partial-dependence calculations.
- Rendering millions of raw observations in a browser.
- Loading executable estimator pickles across untrusted version boundaries;
  safe declarative analysis artifacts are in scope.
- Runtime service telemetry; Ginsu is an in-process library, not a service.

## 4. Current architecture and key constraint

The current estimator accepts an array-like `X` and an error vector, converts
`X` to an ndarray, one-hot encodes it with scikit-learn, and enumerates candidate
slices with SciPy CSR matrices. Optional Numba accelerates score and identifier
operations. Public results are an object ndarray and a list of dictionaries.

Pandas is not the core compute engine today. It appears in the two notebooks,
the development extra, and pandas-aware branches of the vendored validation
module. Therefore, "replace pandas with Polars" should mean:

- replace pandas in the canonical named-table experience and maintained
  examples while accepting compatible pandas inputs through Arrow/interchange;
- make Polars schemas and expressions the result/analysis contract;
- isolate the conversion to the existing sparse numeric engine;
- remove the direct pandas runtime/development dependency, keeping pandas only
  in a dedicated interoperability test environment; and
- avoid an unnecessary rewrite of tested sparse logic.

## 5. Code review against the production project template

### 5.1 Review metadata and verdict

- **Repository:** `DataDome/sliceline`
- **Branch and commit:** `main` at
  `356f71a9ecaa6b97e299e341ebc9424cf7b88c4a`
- **Review date:** 2026-09-09
- **Product purpose:** Discover categorical data slices where an ML model has
  unusually high observed error.
- **Runtime:** Python 3.10+, NumPy, SciPy, scikit-learn, and optional Numba.
- **First-party source reviewed:** 2,003 lines across `sliceline/*.py`; tests,
  workflows, packaging, notebooks, benchmarks, and documentation were also
  inspected.
- **Excluded:** Git internals, generated notebook HTML/CSS, dependency source,
  and line-by-line review of the vendored validation implementation.
- **Starting worktree:** Clean at the reviewed commit.
- **Rubric:** Applicable controls from the production repository standard plus
  correctness, numerical integrity, compatibility, bounded resource use, and
  release safety.
- **Blocking threshold:** Any credible silent result corruption, invalid score
  domain, or unsafe/irreproducible publication path.
- **Overall grade:** C — useful tested core, but not ready for the proposed
  production-facing expansion.
- **Release recommendation:** Block the Polars/plotting feature release until
  all critical and high findings are corrected or explicitly dispositioned by
  an accountable maintainer.
- **Highest risk:** Named-column reorder can silently produce wrong membership.
- **Strongest property:** The sparse engine has a substantial deterministic
  fixture corpus, optional Numba fallback, and a buildable wheel.
- **First improvement:** Establish the locked, green baseline and repair input
  invariants before changing result containers.

### 5.2 Review evidence

The review used repository source, workflows, tests, notebooks, package build,
and focused behavioral probes. At the reviewed commit:

- `tests/test_slicefinder.py`: **47 passed** on Python 3.12.11 with the declared
  dev and optimized extras.
- The focused test run reported **93% coverage** for `slicefinder.py`, but this
  is not full project coverage because the configuration excludes
  `validation.py` and `_numba_ops.py`.
- `ruff check .`: **failed with 32 findings** under the currently resolved Ruff
  version, including source, tests, validation, and notebook findings.
- `ruff format --check .`: **failed with 3 files requiring formatting**.
- `uv build`: **passed**, producing an sdist and universal wheel.
- Sphinx with warnings-as-errors could not be assessed because the local process
  failed while initializing an unsupported host locale. This is an environment
  limitation, not recorded as a Sliceline defect; CI must supply the clean
  documentation-build evidence.
- The full performance suite was not used as a pass/fail release signal because
  its tests mix behavior checks with benchmark calibration and declare no stable
  regression thresholds.

### 5.3 Findings

| ID | Severity | Finding and evidence | Risk | Required disposition |
|---|---|---|---|---|
| CR-01 | Critical | Named-column identity is recorded during `fit`, but the encoder is fitted on the converted ndarray, and `transform` validates only values. See [`fit`](../ginsu/slicefinder.py), lines 213-224, and [`transform`](../ginsu/slicefinder.py), lines 242-249. A focused probe changed membership when the same dataframe columns were reordered. | Silent model-debugging corruption. | Add schema capture and `reset=False` validation at every inference boundary; test reorder, missing, extra, and duplicate columns. |
| CR-02 | Critical | The score divides by `average_error_`, but `fit` accepts all-zero and negative error vectors. See [`_score`](../ginsu/slicefinder.py), lines 456-483, and `average_error_` assignment around line 785. Focused probes accepted both cases. | Undefined scoring and invalid pruning assumptions can silently return empty or incorrect results. | Require a one-dimensional, finite, nonnegative error vector with strictly positive mean; document the loss contract and test both engines. |
| CR-03 | High | Fractional `min_sup` uses `int(fraction * len(X))`; a positive fraction can become zero. See [`fit`](../ginsu/slicefinder.py), lines 213-217. A four-row input with `min_sup=.01` produced `_min_sup_actual == 0`. | The documented minimum-support guarantee is not enforced on small inputs. | Use `max(1, math.ceil(fraction * n_rows))`, document the rounding rule, and add boundary tests. |
| CR-04 | High | CI listens for pushes to `master`, while the repository default branch is `main`. See [`.github/workflows/push-pull.yml`](../.github/workflows/push-pull.yml), lines 3-6. | Direct changes on the default branch may not execute CI. | Trigger on `main` and pull requests; add explicit least-privilege permissions and concurrency cancellation. |
| CR-05 | High | `uv.lock` is both absent and explicitly ignored, while `make init` resolves mutable dependencies with `uv sync --all-extras`. See [`.gitignore`](../.gitignore) and [`Makefile`](../Makefile), lines 1-4. | Local, CI, docs, and release environments are not reproducible. | Track one generated `uv.lock`; use `uv sync --frozen`; separate runtime, test, docs, plot, and optimization groups. |
| CR-06 | High | The release workflow is triggered by an existing GitHub release and later calls `gh release create` for that same reference after publishing to PyPI. See [`.github/workflows/release.yml`](../.github/workflows/release.yml), lines 3 and 53-90. | The GitHub artifact stage is expected to fail after an irreversible PyPI publication. | Redesign as tag-triggered build/test/sign/publish or upload assets to the triggering release without recreating it; validate version/tag equality before publication. |
| CR-07 | High | Coverage omits the entire custom validation module and optional Numba implementation. See [`pyproject.toml`](../pyproject.toml), lines 86-90. These are correctness-sensitive paths, not generated code. | The highest-risk compatibility and numerical branches can regress without affecting the gate. | Measure both; run dedicated NumPy-fallback and Numba jobs and enforce cross-engine parity. |
| CR-08 | High | Candidate compatibility constructs a dense `n_slices x n_slices` matrix using `.toarray()` at each level. See [`_join_compatible_slices`](../ginsu/slicefinder.py), around lines 562-597. | Cardinality can cause quadratic memory exhaustion before later sparse pruning helps. | Establish a memory estimate and configurable candidate ceiling; test fail-fast behavior; investigate chunked joins separately from the Polars migration. |
| CR-18 | High | Both maintained notebooks fit the model, calculate predictions, perform error-supervised binning, discover slices, and report slice error on the same observations. See the [Titanic notebook](../notebooks/1.%20Implementing%20Ginsu%20on%20Titanic%20dataset.ipynb) and [California Housing notebook](../notebooks/2.%20Implementing%20Ginsu%20on%20California%20housing%20dataset.ipynb). | The examples combine training optimism, target-aware preprocessing, search selection, and evaluation, so the displayed error lift can materially overstate generalization. | Use out-of-fold or held-out model errors, fit every learned preprocessing step only on its discovery/training partition, and report slice metrics on untouched validation data. |
| CR-09 | Medium | The reviewed upstream repository contained an 861-line modified copy of scikit-learn validation internals and imported private `_check_feature_names`. See upstream [`validation.py`](https://github.com/DataDome/sliceline/blob/356f71a9ecaa6b97e299e341ebc9424cf7b88c4a/sliceline/validation.py) and [`slicefinder.py`](https://github.com/DataDome/sliceline/blob/356f71a9ecaa6b97e299e341ebc9424cf7b88c4a/sliceline/slicefinder.py). | Upstream private-API drift and duplicated dataframe semantics make upgrades fragile. | Replace private calls with a small owned boundary adapter plus public scikit-learn validation where possible; characterize string/categorical behavior first. |
| CR-10 | Medium | Public result state is split between an object ndarray and `list[dict[str, float]]`; `get_slice` always returns an ndarray. See [`slicefinder.py`](../ginsu/slicefinder.py), lines 251-275 and 862-892. | Names, dtypes, nulls, and row context are lost; plot consumers reconstruct inconsistent tables. | Introduce canonical Polars result tables and type-preserving selection before Ginsu's first stable release. |
| CR-11 | Medium | Unit tests wrap most correctness calls in `pytest-benchmark`, exact float equality is common, and performance tests lack declared regression thresholds or controlled runner metadata. | Slow test feedback and noisy numbers are presented without a defensible performance decision. | Separate deterministic behavior tests from marked benchmarks; use tolerance-based numeric assertions and benchmark manifests. |
| CR-12 | Medium | Ruff is declared with a lower bound only, there is no lock, and the current gate fails on repository content. `setup.cfg` also retains an unused Flake8 policy that conflicts with the Ruff-centered workflow. | The documented quality gate is neither reproducible nor currently green. | Lock tools, resolve findings, configure intentional notebook/per-file exclusions, and remove obsolete configuration. |
| CR-13 | Medium | No type-checking gate or `py.typed` marker exists; core aliases use `NDArray[Any]`. | Public contract drift is easy and downstream type checking cannot rely on the package. | Add mypy or pyright, precise public protocols/types, and a packaged `py.typed` marker. |
| CR-14 | Medium | Release builds install unpinned tools directly, actions use mutable major/version tags rather than commit SHAs, and artifacts are not smoke-tested before publication. See both release workflows. | Supply-chain drift and publishing broken artifacts. | Pin actions by SHA, build once after the full gate, install the wheel in a clean environment, verify metadata/contents, then publish with trusted publishing. |
| CR-15 | Medium | The repository lacks the template's `SECURITY.md`, `CHANGELOG.md`, `REPRODUCIBILITY.md`, dependency update policy, and explicit release-readiness record. `CONTRIBUTING.md` refers to a nonexistent `docs/releases` directory. | Vulnerability reporting, change communication, and release evidence are unclear. | Add the applicable governance documents and correct contribution guidance. |
| CR-16 | Low | Import-time Numba detection conflates installation with operational availability, and estimator construction emits a performance warning by default. | Noisy interactive use and brittle interpretation of runtime state. | Make backend selection explicit and inspectable; warn once at operation time only when relevant. |
| CR-17 | Low | Package documentation is essentially the README plus one generated class page; plot semantics, error assumptions, limits, and reproducibility are not documented. | Users can misinterpret scores and observational plots. | Add task-oriented guides, API schemas, limitations, and a plotting interpretation guide. |

### 5.4 Template applicability decisions

| Template control | Applicability | Decision |
|---|---|---|
| Canonical checkout under `~/Projects` | Applicable | Satisfied by this working checkout. |
| `AGENTS.md` and bounded `PROJECT_MEMORY.md` | Applicable when agent-assisted maintenance continues | Add using the template's retrieve-before-act and evidence-linked memory model. |
| Lockfile and frozen installs | Applicable | Required before feature implementation. |
| `SECURITY.md`, changelog, reproducibility | Applicable to a published library | Required. Preserve the existing BSD-3-Clause license. |
| `src/` layout | Beneficial but disruptive | Defer until after the compatibility release unless artifact tests reveal package-discovery ambiguity. Record the decision. |
| Docker image | Not applicable | This is a pure library; validate wheels in clean virtual environments instead. |
| `.env.example` | Not applicable | The library has no runtime configuration or external service. |
| Runtime telemetry schema | Not applicable | Do not add library telemetry. Benchmark and release evidence are build artifacts, not runtime events. |
| Threat model | Limited applicability | Add a lightweight library threat model covering untrusted high-cardinality inputs, resource exhaustion, notebook data exposure, HTML export, and artifact supply chain. |
| SBOM, provenance, checksums | Applicable | Required for externally published distributions. |

## 6. Architecture decisions

### 6.1 Dependency layers

Use three explicit layers:

```text
Arrow-compatible inputs
        │
        ▼
Polars canonical boundary and result model
                │
                ▼
Owned validation/conversion adapter
                │
                ▼
NumPy + SciPy sparse SliceLine engine
                │
                └──── optional Numba acceleration

Polars result frames ───► plot-data builders ───► optional Plotly renderer
```

Recommended dependencies:

- Core: Polars, NumPy, SciPy, scikit-learn.
- Arrow interoperability should use Polars' public Arrow C Data/PyCapsule and
  dataframe-interchange support. Add PyArrow as an optional `arrow` extra only
  if direct `pyarrow.Table` tests demonstrate that it is required; do not make
  pandas a transitive design requirement.
- `optimized` extra: Numba.
- `plot` extra: Plotly.
- Development groups: test, type, docs, notebooks, and release tooling.
- Remove pandas and Matplotlib after notebook and plot migration. Install
  pandas only in a dedicated compatibility-test group. No production code may
  import or perform type checks against pandas.

Resolve exact compatible bounds with the lockfile and tested matrix. Do not copy
today's newest resolved versions into the manifest without compatibility
evidence.

### 6.2 Proposed module boundaries

```text
ginsu/
├── __init__.py
├── py.typed
├── slicefinder.py          # stable estimator facade
├── _engine.py              # sparse enumeration orchestration
├── _numba_ops.py           # optional numerical acceleration
├── _validation.py          # owned input/schema/error contracts
├── _frame.py               # Arrow input → Polars → bounded ndarray conversion
├── predicates.py           # immutable Predicate and Slice domain values
├── binning.py              # fitted, reusable Polars discretization plans
├── results.py              # result schemas, metrics, validation, artifacts
├── diagnostics.py          # search report and resource-limit outcomes
├── comparison.py           # model/time-window slice comparisons
└── plotting/
    ├── __init__.py
    ├── _data.py            # pure Polars plot-data builders
    └── _plotly.py          # optional rendering adapter
```

Do not perform this file split mechanically in the first pull request. First
protect behavior with characterization tests; then move one responsibility at a
time.

### 6.3 Polars and Arrow input contract

Initial supported inputs:

- `pl.DataFrame`: first-class, schema-aware input.
- Objects exposing the Arrow C Stream/PyCapsule interface: accepted through the
  public Polars constructor/interchange boundary when conversion preserves the
  supported schema.
- `pyarrow.Table` and `pyarrow.RecordBatch`: supported Arrow inputs, with direct
  compatibility tests.
- `pandas.DataFrame`: supported through its public Arrow/dataframe interchange
  interface. Ginsu must not import pandas, inspect pandas-private state, or
  maintain a separate pandas execution path.
- `np.ndarray` and existing SciPy-compatible inputs: supported compatibility
  path.
- `pl.LazyFrame`: rejected with a clear message directing the caller to an
  explicit `.collect()`. Add lazy execution only with a defined row bound and
  streaming design.

Every tabular input is normalized into a Polars frame before validation. The
adapter records whether conversion was zero-copy or allocating when the public
producer/Polars APIs expose that information. If a conversion cannot preserve
names, order, values, or supported dtypes, it fails rather than silently
coercing. A necessary copy is allowed and benchmarked; "Arrow-native" must not
be advertised as universally zero-copy.

Implementation references:

- Polars [`from_dataframe`](https://docs.pola.rs/api/python/stable/reference/api/polars.from_dataframe.html)
  uses the Arrow PyCapsule interface and falls back to the dataframe interchange
  protocol; its `allow_copy` option makes copy policy testable.
- Polars [`from_arrow`](https://docs.pola.rs/api/python/stable/reference/api/polars.from_arrow.html)
  accepts Arrow tables, record batches, chunked arrays, and Arrow PyCapsule
  producers. The documentation says conversion is zero-copy for the most part
  and warns that unsupported types may be cast, so Ginsu must compare source
  and canonical schemas instead of assuming lossless or zero-copy conversion.
- The canonical interchange mechanism is the Arrow
  [PyCapsule Interface](https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html),
  not producer-specific private attributes.

At `fit`:

- require at least one row and one feature;
- reject duplicate column names;
- capture ordered names and normalized Polars schema;
- reject unsupported nested, object, decimal, timezone, and mixed semantic
  dtypes until their conversion is specified;
- reject nulls and floating NaNs by default;
- identify high-cardinality columns and fail or warn according to a documented
  threshold before one-hot expansion;
- convert once at the engine boundary and record conversion allocation in
  benchmarks.

At `transform` and `get_slice`:

- require the same named schema for Polars input;
- reject reorder, missing, extra, duplicate, or incompatible columns with a
  specific error;
- preserve input row order;
- return Polars when input is Polars and ndarray when input is ndarray, unless
  an explicit scikit-learn `set_output` configuration overrides the container.

Arrow and pandas inputs return Polars outputs by default because Polars is the
canonical container. Document explicit Arrow export methods for callers that
need `pyarrow.Table`; pandas callers can use their chosen public Arrow conversion
without a Ginsu-owned pandas adapter.

Interoperability tests must cover NumPy-backed pandas columns, Arrow-backed
pandas columns, pandas categorical/string/nullable extension dtypes, PyArrow
dictionary arrays, chunked arrays, dates/timestamps, Boolean values, nulls,
Unicode, and conversion paths that require copies. Unsupported schema behavior
must be identical across producers after canonicalization.

### 6.4 Error-vector contract

The algorithm treats `errors` as nonnegative per-observation losses. Enforce:

- one dimension;
- row count equal to `X.height`;
- numeric or Boolean indicator-loss values, finite after normalization;
- every value `>= 0`;
- at least one value `> 0`; and
- strictly positive finite mean.

Accept a Polars `Series`, a named one-column Polars frame, or a numeric ndarray.
Normalize to contiguous `float64` for the engine. A future dtype optimization
requires parity tests for score, ranking, and pruning.

### 6.5 Canonical result model

Add the following fitted properties/methods:

#### `slices_ -> pl.DataFrame`

One row per slice, with feature columns retaining their natural Polars dtype.
Null means the feature is not part of the rule. Prefix metadata to avoid feature
name collisions:

| Column | Type | Meaning |
|---|---|---|
| `__ginsu_id` | `String` | Stable key derived from canonical predicates. |
| `__ginsu_rank` | `UInt32` | Rank in this fit; tied scores use deterministic rule ordering. |
| `__ginsu_rule` | `String` | Escaped, human-readable conjunction. |
| feature columns | original compatible dtype | Predicate value or null. |

Reject input feature names beginning with reserved `__ginsu_`, or define a
reversible escaping scheme before release.

#### `slice_statistics_ -> pl.DataFrame`

One row per slice:

- `__ginsu_id`
- `rank`
- `slice_score`
- `support_count`
- `support_fraction`
- `error_sum`
- `error_max`
- `error_mean`
- `baseline_error_mean`
- `error_lift = error_mean / baseline_error_mean`
- `excess_error = error_sum - support_count * baseline_error_mean`
- `predicate_count`
- `membership_group_id`, when analyzed against a reference dataset

Use integer support counts. Derived metrics must be computed in one owned
function and tested against their formulas.

#### `predicates_ -> pl.DataFrame`

Normalized long form for plotting and interchange:

- `__ginsu_id`
- `rank`
- `predicate_position`
- `feature`
- `operator` (initially `eq`)
- `value_json` (canonical serialization)
- `display_value`
- `source_dtype`

The long form deliberately avoids a Polars `Object` column. Canonical JSON must
cover strings, booleans, integers, floats, dates, datetimes, and categorical
values before those dtypes are advertised.

#### Pre-release transition

The inherited `top_slices_` and `top_slices_statistics_` attributes may remain
while the engine is being characterized, but they are not a compatibility
commitment. Remove or make them private before Ginsu's first stable release once
the canonical result frames cover every use case. Do not ship two public sources
of truth.

### 6.6 Membership and overlap analysis

Add bounded, explicit methods rather than eagerly materializing every view:

```python
finder.transform(X)  # type-preserving membership columns
finder.membership_frame(X)  # row id + Boolean slice columns
finder.equivalence_groups(X)  # exact membership groups
finder.overlap_frame(X, metric="jaccard", max_slices=100)
finder.lattice_edges()  # parent-child rule relationships
```

Rules:

- Boolean membership columns are named by stable slice ID.
- `membership_frame` accepts an optional row identifier; generated row numbers
  are positional and documented as such.
- Exact equivalence groups use packed Boolean masks or a stable digest plus an
  equality check to handle hash collisions.
- Pairwise overlap estimates `O(k²)` work and fails before allocation when
  `k > max_slices`, unless explicitly overridden.
- Jaccard for two empty sets is undefined; it should be null, not silently zero
  or one.
- Lattice edges mean one rule is the exact predicate-subset parent of another;
  they do not imply causality or statistical dependence.

### 6.7 Immutable slice domain model

Add frozen, typed `Predicate` and `Slice` values beneath the Polars result
tables:

```python
Predicate(feature="age_bin", operator="eq", value="30-39")
Slice(
    predicates=(
        Predicate("age_bin", "eq", "30-39"),
        Predicate("travel_class", "eq", 3),
    )
)
```

Requirements:

- Canonical predicate ordering is independent of display order.
- Equality and hashing use canonical typed values, not formatted labels.
- A versioned canonical encoding produces stable slice IDs.
- Subset, parent, child, and equivalence operations are pure and tested.
- Polars expressions can be generated safely without string-evaluated queries.
- Display strings escape values and are never parsed back into predicates.
- The bulk API remains Polars frames; objects are the domain representation for
  serialization, comparison, and graph construction.

### 6.8 Reproducible Polars-native discretization

Introduce a fitted `DiscretizationPlan` rather than asking every notebook to
invent bins:

```python
plan = DiscretizationPlan(
    numeric={"age": QuantileBins(n_bins=10)},
    categorical={"port": CategoryPolicy(max_categories=20)},
)
X_binned = plan.fit_transform(X_discovery)
X_validation_binned = plan.transform(X_validation)
```

Initial strategies:

- fixed caller-supplied boundaries;
- equal-width bins;
- quantile bins with deterministic duplicate-boundary handling;
- categorical rare-level grouping with a stable unknown category; and
- pass-through for already discrete supported columns.

Rules:

- Learn every boundary, level set, and rare-category rule from the discovery or
  training partition only.
- Store closed/open interval semantics, units, null policy, out-of-range policy,
  and source dtype.
- Apply identical boundaries at validation and transform time.
- Preserve the raw-to-binned feature mapping for dependence plots.
- Report constant, all-null, infinite, and excessive-cardinality columns.
- Default strategies are unsupervised. Any error/target-aware binning requires a
  separate strategy, explicit leakage-safe fold contract, and statistical ADR.
- Serialize the plan declaratively; do not require pickle to reuse it.

### 6.9 Discovery versus statistical validation

Separate finding a rule from estimating how it performs:

```python
finder.fit(X_discovery, discovery_errors)
validation = finder.validate_slices(X_validation, validation_errors)
```

`validate_slices` applies the already discovered predicates to untouched data;
it does not rerun search or modify the fitted ranking. Its Polars result includes
discovery and validation support/error metrics, changes, and a validation status.

Required statistical controls:

- Document that `slice_score` is a search/ranking objective, not a p-value or
  proof of significance.
- Label in-sample statistics as discovery statistics everywhere.
- Support caller-declared IID, cluster, block/time, or no-interval modes; do not
  choose an IID bootstrap silently.
- Provide confidence intervals only when the resampling unit, method, seed,
  repetitions, and assumptions are recorded.
- Apply a declared multiple-comparison policy to inference over several
  validated slices, such as Benjamini-Hochberg, or label results descriptive.
- Offer an expensive optional global permutation test that reruns the full
  discovery procedure under the null; a permutation of only the winning slice's
  rows does not correct search selection bias.
- Report effective validation support and return `insufficient_support` rather
  than an unstable estimate when the declared threshold is not met.
- Keep model selection/tuning data separate from the final assessment set.

### 6.10 Diversity-aware selection and stability analysis

Keep exhaustive score ranking as the source result, then provide an explicit
post-selection API:

```python
diverse = finder.select_slices(
    method="diverse",
    k=20,
    X=X_reference,
    max_jaccard=0.80,
)
```

- `method="score"` preserves the current ranking.
- `method="unique_membership"` keeps one representative per exact membership
  group.
- `method="diverse"` uses a documented deterministic objective balancing score
  and overlap; it never overwrites the raw discoveries.
- Return exclusion reasons and the representative chosen for each group.

Add a separate stability evaluator over caller-approved resamples, folds, or
time windows:

- selection frequency;
- rank distribution;
- score/support/error-lift distribution;
- predicate-set similarity;
- membership similarity on a fixed reference frame; and
- sensitivity to `alpha`, `min_sup`, `max_l`, and discretization choices.

The stability report records seeds, resampling units, data partitions,
parameters, and failures. A low-stability slice is labeled fragile rather than
removed silently.

### 6.11 Model-version and time-window comparison

Add a comparison use case over canonical result artifacts:

```python
comparison = compare_analyses(
    baseline,
    candidate,
    reference_data=X_reference,
)
```

Outputs:

- exact and semantically matched rules;
- newly emerged, resolved, improved, and regressed slices;
- support, error-lift, excess-error, and rank deltas;
- membership migration on a common reference population;
- feature/predicate drift;
- validation and stability-status changes; and
- an aggregate regression waterfall suitable for a release review.

Comparison requires compatible feature semantics and discretization-plan
versions. Incompatible schemas produce an explicit non-comparable result; they
are not coerced. Time comparisons record window boundaries, timezone, event-time
semantics, and label-availability cutoffs.

### 6.12 Sample weights and baseline policies

Treat weighting as an algorithm extension, not a dataframe convenience.
Research and specify:

- frequency versus analytic/sample weights;
- weighted support and whether `min_sup` remains a row count;
- exposure-weighted error means and sums;
- per-cohort or caller-supplied baselines;
- time-decay weights; and
- classification decompositions such as false-positive and false-negative
  rates.

Before implementation, approve an ADR containing formulas, admissible weight
domains, pruning-bound proof or counterexample, Numba/NumPy parity requirements,
and backward-compatibility behavior. If the existing upper bound is invalid
under weights, do not reuse it. Ship weighted scoring only after brute-force
small-lattice tests prove the optimized search returns the same top slices.

### 6.13 Structured diagnostics and resource controls

Every fit produces a read-only `SearchReport` with:

- engine/backend and whether Numba acceleration was actually used;
- normalized input schema, row/feature counts, encoded cardinality, and copy
  boundaries;
- candidate counts generated, pruned, and evaluated by lattice level and
  reason;
- timing by stage using an injected monotonic clock;
- estimated and observed peak memory when measurement is enabled;
- warnings with stable codes;
- termination status: `complete`, `no_valid_slices`, `limit_reached`,
  `cancelled`, or `failed`; and
- active resource limits.

Add explicit limits for:

- encoded columns and per-feature cardinality;
- candidates generated/evaluated per level and in total;
- dense compatibility-matrix bytes;
- elapsed search time through cooperative checks;
- maximum returned tied slices; and
- membership/overlap/plot cells and rendered observations.

Defaults must be conservative and documented. Preflight estimates fail before
allocation when possible. A partial result is never presented as exhaustive;
its report and result metadata retain the limit and termination reason.

### 6.14 Versioned `SliceAnalysis` artifact

Provide a safe declarative artifact consisting of JSON metadata plus Parquet or
Arrow IPC tables:

- schema and artifact-format version;
- canonical predicates, stable IDs, raw rankings, and derived selections;
- discovery and validation statistics;
- discretization plan;
- stability, comparison, and multiple-testing metadata when present;
- search parameters, report, resource limits, and termination status;
- package/algorithm version and dependency-lock identifier;
- ordered feature schema and canonicalization version; and
- caller-supplied dataset/partition fingerprints, never raw source data by
  default.

Loading validates every version, schema, hash, bound, and table relationship.
Unknown major versions fail closed. The artifact contains no executable pickle,
arbitrary class import, HTML, or code. Round-trip and malicious/oversized input
tests are required.

### 6.15 Privacy, fairness, and sensitive-slice controls

Add analysis/display policies rather than assuming `min_sup` is a privacy
control:

- a display/export support threshold distinct from search support;
- aggregate-only dependence plots and deterministic point suppression;
- predicate-value redaction or hashing for named sensitive features;
- protected-feature annotations and optional per-group reporting;
- warnings when intersections create very small populations;
- a strict allowlist for tooltip/export fields; and
- documentation that discovered associations do not establish unfair treatment
  or causal discrimination.

These controls reduce accidental disclosure but do not claim formal privacy.
Differential privacy, legal fairness definitions, and compliance decisions need
separate domain review.

## 7. Visualization design

### 7.1 Rendering architecture

All plotting is two-stage:

1. A pure builder creates a documented Polars table from fitted state and
   caller-provided data.
2. The Plotly adapter renders that table and returns a
   `plotly.graph_objects.Figure` without importing or converting through pandas.

This makes plot semantics unit-testable without pixel snapshots and leaves room
for future renderers. Importing `ginsu` or the plot-data builders must not
import Plotly. Calling a renderer without the extra should raise one actionable
installation error.

### 7.2 Impact plot

```python
finder.plot_impact(
    x="support_fraction",
    y="error_lift",
    size="excess_error",
    color="predicate_count",
)
```

- X: support fraction, optionally logarithmic.
- Y: error lift relative to the fitted baseline.
- Size: nonnegative display transform of excess error. Negative excess error is
  not silently discarded; expose it in tooltips and use a signed color/channel
  when allowed by the result set.
- Color: predicate count or membership-equivalence group.
- Tooltip: stable ID, rank, rule, score, counts, error metrics, and group.
- Reference line: `error_lift == 1`.
- Deterministic label selection prevents every point from being annotated.
- When validation results exist, discovery-only points are visibly distinct and
  users can switch axes/tooltips to validation metrics and declared intervals.

### 7.3 Predicate matrix

```python
finder.plot_predicates(sort_by="slice_score", max_slices=50)
```

- Rows: ranked slices or collapsed membership groups.
- Columns: features.
- Filled cells: predicate display value.
- Blank cells: unused features.
- Marginal bars: support and error lift.
- Sorting: score, support, error lift, predicate count, or group.
- Badges: discovery-only, validated, insufficient support, fragile, or
  limit-affected.
- Long values are escaped and truncated visually while full values remain in
  accessible tooltips.

This is the primary replacement for the notebooks' wide dataframe displays.

### 7.4 Overlap and equivalence plot

```python
finder.plot_overlap(X, metric="jaccard", kind="heatmap")
```

Initial renderer: clustered Jaccard heatmap with exact-equivalence annotations.
An UpSet renderer can follow once its dependency and accessible fallback are
selected. Do not default to a force-directed network: it is hard to compare,
unstable, and quickly unreadable.

Required behaviors:

- collapse exact membership duplicates on request;
- show represented-rule count for each group;
- distinguish zero overlap from unavailable/uncomputed cells;
- display intersection and union counts in tooltips; and
- enforce the pairwise computation bound before allocating.

### 7.5 Slice lattice plot

```python
finder.plot_lattice(max_nodes=100, collapse_equivalent=True)
```

- Node size: support count.
- Node color: error lift or score.
- Edge: child adds exactly one predicate to the parent rule.
- Layout: deterministic levels by predicate count.
- Default pruning: top-ranked nodes plus required ancestors.
- Explicitly label that edges encode rule refinement, not observed transitions
  or causality.

### 7.6 Error-dependence plot

Use the term **error-dependence plot**, not "partial dependence plot".
Ginsu receives realized errors and does not intervene on a fitted predictor.

```python
finder.plot_error_dependence(
    X_binned,
    errors,
    feature="age_bin",
    slice_id="...",
    feature_values=raw_age,  # optional pre-binning Series
    color_by="class",  # optional observed interaction
)
```

Semantics:

- X: the supplied raw feature values when present; otherwise the fitted feature
  values or categories.
- Y: observation-level error/loss.
- Highlight: rows belonging to the selected slice.
- Reference: fitted overall mean error.
- Overlay: deterministic binned mean/median and counts. Do not default to a
  smooth curve that implies unsupported structure.
- Validation data, when supplied, is a separate trace/facet and is never pooled
  into the discovery summary without explicit labeling.
- Slice interval: shaded only when interval metadata is supplied explicitly;
  do not parse arbitrary display strings as mathematical intervals.
- Second feature: color or facets, bounded to a documented category count.
- Categorical mode: box/violin plus points or count-aware summaries rather than
  a continuous trend.

Contract requirements:

- `X_binned`, `errors`, and optional raw values must have identical row count and
  order; accept an optional row key for checked alignment.
- Deterministically downsample rendered points while computing summaries over
  all rows.
- Tooltips must not expose extra source columns.
- Any confidence interval is opt-in and labeled with its method and assumptions.
  No default IID bootstrap band is appropriate for every Ginsu use case.
- Documentation must state that the plot is descriptive, observational, and not
  causal.

### 7.7 Stability and sensitivity plots

```python
plot_stability(stability_report, metric="selection_frequency")
plot_sensitivity(stability_report, parameter="alpha", metric="rank")
```

- Show selection frequency with support counts and uncertainty/resample count.
- Show rank/metric distributions rather than only averages.
- Plot `alpha`, `min_sup`, and `max_l` sensitivity as parameter-response panels.
- Use a fixed reference dataset for membership stability and label it.
- Mark failed or resource-limited resamples instead of dropping them.
- Avoid connecting categorical parameter configurations with lines that imply a
  continuous path.

### 7.8 Model/time comparison plots

```python
plot_comparison(comparison, kind="waterfall")
```

- Waterfall: aggregate excess-error changes attributable to matched slices.
- Dumbbell/slope: baseline versus candidate validation error lift and support.
- Migration matrix: membership movement on the fixed reference population.
- Timeline: slice metrics across explicitly bounded event-time windows.
- Always show unmatched and non-comparable slices; do not force a match to make
  the chart complete.

### 7.9 Search-profile plot

```python
plot_search_report(finder.search_report_)
```

- Candidate funnel by lattice level and pruning reason.
- Stage timing and optional peak-memory profile.
- Cardinality contribution by source feature.
- Resource-limit thresholds and the actual termination point.
- Backend/acceleration state and copy boundaries in figure metadata/tooltips.

This plot explains search cost; it must not be presented as model-quality
evidence.

### 7.10 Accessibility and export

- Use a colorblind-safe palette and never rely on color alone.
- Supply figure titles, axis labels, units, legend labels, and accessible hover
  text.
- Verify usable light and dark themes.
- Support HTML and image export only through explicit user calls.
- Document that self-contained HTML can contain data embedded in the figure.
- Keep static image export in a separate optional extra if it introduces a
  renderer such as Kaleido.

## 8. Public API sketch

```python
import polars as pl

from ginsu import DiscretizationPlan, QuantileBins, Slicefinder
from ginsu.results import SliceAnalysis, compare_analyses
from ginsu.plotting import (
    plot_error_dependence,
    plot_impact,
    plot_overlap,
    plot_predicate_matrix,
)

raw_features = pl.DataFrame(
    {
        "age": [24, 34, 38, 47],
        "travel_class": [1, 3, 3, 2],
        "embarked": ["S", "Q", "Q", "C"],
    }
)
errors = pl.Series("log_loss", [0.05, 0.71, 0.84, 0.12])

binning = DiscretizationPlan(numeric={"age": QuantileBins(n_bins=4)})
features = binning.fit_transform(raw_features)
finder = Slicefinder(k=10, max_l=3, min_sup=0.01, verbose=False)
memberships = finder.fit_transform(features, errors)

print(finder.slices_)
print(finder.slice_statistics_)
print(finder.predicates_)

impact_figure = plot_impact(finder)
overlap_figure = plot_overlap(finder, features, max_slices=100)
dependence_figure = plot_error_dependence(
    finder,
    features,
    errors,
    feature="age_bin",
    slice_id=finder.slices_[0, "__ginsu_id"],
)

validation = finder.validate_slices(validation_features, validation_errors)
diverse = finder.select_slices(
    method="diverse",
    k=5,
    X=validation_features,
    max_jaccard=0.80,
)

analysis = SliceAnalysis.from_finder(
    finder,
    validation=validation,
    dataset_fingerprint="caller-supplied-digest",
)
analysis.write("artifacts/model-v2-slices")

comparison = compare_analyses(
    previous_analysis,
    analysis,
    reference_data=validation_features,
)
```

Arrow-compatible producers use the same API and normalize to Polars:

```python
# pandas_frame implements a public dataframe/Arrow interchange protocol.
finder.fit(pandas_frame, errors)

# Arrow tables and record batches are also supported directly.
finder.fit(arrow_table, errors)
```

Ginsu does not import pandas in either case. Compatibility tests, not class
name checks, define which protocol versions and dtypes are supported.

Prefer free plotting functions in the first release. They keep the estimator
focused and avoid making optional dependencies appear to be core estimator
methods. Thin convenience methods can be added later without changing plot-data
contracts.

## 9. Delivery plan

Each milestone is independently reviewable and keeps the main branch releasable.
Do not combine repository hardening, dataframe migration, and all plot types in
one pull request.

### Phase 0 — Baseline, governance, and correctness blockers

Deliverables:

- Add `AGENTS.md`, `PROJECT_MEMORY.md`, `SECURITY.md`, `CHANGELOG.md`, and
  `REPRODUCIBILITY.md`, adapted for a public library.
- Record architecture decisions for the Polars boundary, plotting dependency,
  and compatibility/deprecation policy.
- Track `uv.lock`; remove it from `.gitignore`; make installs frozen.
- Split nonmutating `lint`/`format-check` from formatting commands.
- Repair CI default-branch triggers and add explicit permissions/concurrency.
- Separate correctness tests from benchmarks.
- Add characterization tests for current rule ranking, ties, unknown categories,
  strings, and Numba/NumPy parity.
- Fix CR-01, CR-02, and CR-03 with regression tests.
- Make the existing lint, format, test, documentation, and build checks green.

Acceptance gate:

- Clean checkout passes `make check` from the frozen lock.
- Reordered or schema-mismatched named input fails before encoding.
- Invalid error domains fail with stable exception types/messages.
- Positive fractional support is always at least one observation.
- Current valid experiment fixtures retain equivalent outputs.

### Phase 1 — Polars/Arrow boundary and canonical domain/results

Deliverables:

- Add Polars as a core dependency and implement `_frame.py`/`_validation.py`.
- Normalize supported Polars, Arrow, pandas-interchange, and NumPy inputs through
  the documented boundary.
- Add immutable `Predicate` and `Slice` values with versioned canonical IDs.
- Add canonical `slices_`, `slice_statistics_`, and `predicates_` frames.
- Keep inherited result attributes only as temporary characterization aids;
  remove or privatize them before the first stable release.
- Make `transform` and `get_slice` type-preserving.
- Add pandas-free core wheel smoke tests and a separate pandas interoperability
  job.
- Add type checking and `py.typed`.

Acceptance gate:

- NumPy, Polars, PyArrow, NumPy-backed pandas, and Arrow-backed pandas versions
  of every golden fixture have the same canonical rules, ranks, membership, and
  statistics within declared float tolerances.
- Feature names, input dtypes, null handling, categorical values, Unicode, dates,
  unknown categories, chunking, copy behavior, and empty-result schemas have
  contract tests.
- `python -I` in a clean environment containing only core dependencies imports,
  fits, transforms, and inspects a Polars dataset.
- Import scanning demonstrates no production pandas import; pandas support
  still passes in its compatibility environment through public interchange.

### Phase 2 — Discretization, diagnostics, resource limits, and artifacts

Deliverables:

- Add fitted Polars-native `DiscretizationPlan` strategies and raw-to-binned
  metadata.
- Add `SearchReport`, stable warning/status codes, stage metrics, and copy
  diagnostics.
- Add cardinality, candidate, memory, elapsed-time, tie, and output-size limits.
- Add versioned, non-executable `SliceAnalysis` JSON plus Parquet/Arrow IPC
  artifacts.
- Add fail-closed loading, schema/hash validation, and artifact migration policy.

Acceptance gate:

- Fitted bin plans reproduce boundaries and results across discovery,
  validation, serialization, and process restart.
- No target/error-aware transform occurs outside an explicitly tested training
  fold.
- Resource-limit tests prove preflight rejection or clearly labeled partial
  termination without reporting an exhaustive result.
- Artifact round trips preserve canonical rules, schemas, statistics, report,
  and bin plan; malformed, oversized, and unknown-major artifacts fail closed.

### Phase 3 — Derived metrics, equivalence, overlap, and lattice data

Deliverables:

- Implement support fraction, error lift, excess error, predicate count, and
  stable rule identifiers.
- Implement bounded membership, exact-equivalence, Jaccard-overlap, and lattice
  edge builders.
- Add deterministic sorting, stable serialization, and collision handling.
- Add resource estimates and fail-fast limits to every pairwise operation.

Acceptance gate:

- Formula tests cover zero intersection, full overlap, subset, equivalence, and
  tied ranks.
- Titanic-style multiple rules targeting identical rows collapse into one
  equivalence group with the correct represented-rule count.
- Property tests verify symmetry, Jaccard bounds, diagonal behavior, subset
  relationships, and stable IDs.
- Large-`k` inputs fail before an unbounded pairwise allocation.

### Phase 4 — Statistical validation, diversity, and stability

Deliverables:

- Add discovery/validation separation and `validate_slices`.
- Add declared interval/resampling strategies and multiple-comparison metadata.
- Add optional full-search permutation testing.
- Add score, unique-membership, and diversity-aware post-selection.
- Add resample/fold/time-window stability and hyperparameter sensitivity reports.
- Label discovery-only, insufficient-support, fragile, and limit-affected
  results throughout the result schema.

Acceptance gate:

- Validation never modifies or reselects discovery rules.
- Synthetic-null and injected-signal simulations verify interval/test behavior
  under their stated assumptions; misuse cases fail or remain descriptive.
- Diversity selection is deterministic, reports every exclusion, and leaves raw
  ranking unchanged.
- Stability reports retain failed/limited runs and reproduce with recorded
  seeds, partitions, and parameters.

### Phase 5 — Plot-data contracts and primary plots

Deliverables:

- Create the optional `plot` extra and lazy Plotly import.
- Implement pure Polars data builders.
- Implement impact and predicate-matrix plots with validation/stability status.
- Add semantic figure tests and small deterministic visual baselines only where
  useful.
- Add gallery documentation with classification and regression examples.

Acceptance gate:

- Plot-data schemas and formulas pass without Plotly installed.
- Missing Plotly produces an actionable optional-dependency error.
- Figures contain expected traces, encodings, labels, reference lines, status,
  uncertainty metadata, and hover fields.
- No plot path imports pandas or emits network/file side effects.

### Phase 6 — Dependence, overlap, lattice, stability, and search plots

Deliverables:

- Implement the bounded overlap heatmap and equivalence collapse.
- Implement deterministic, pruned lattice rendering.
- Implement numeric and categorical error-dependence rendering, raw-value
  overlays, deterministic point sampling, and optional interaction coloring.
- Implement stability, sensitivity, and search-profile plots.
- Add interpretation, privacy, and non-causality documentation.

Acceptance gate:

- Numeric, categorical, binned/raw, sparse membership, exact-equivalence, empty,
  tied, and high-cardinality display cases have tests.
- Summaries use every eligible row even when rendered points are downsampled.
- Row-key mismatch and row-order mismatch are rejected.
- Failed/limited resamples and partial searches remain visible.
- Plot limits fail clearly before browser or process memory is exhausted.

### Phase 7 — Model/time comparison and privacy/fairness controls

Deliverables:

- Add `compare_analyses` with exact, semantic, unmatched, and non-comparable
  outcomes.
- Add delta tables, waterfall, slope, membership-migration, and timeline plots.
- Add display thresholds, raw-point suppression, field allowlists, redaction,
  and sensitive-feature annotations.
- Add explicit time-window, timezone, reference-population, and label-availability
  contracts.

Acceptance gate:

- Known model regression/improvement fixtures produce the expected matched and
  unmatched deltas.
- Incompatible feature/binning schemas are reported as non-comparable.
- Export tests prove suppressed columns/rows are absent from figure payloads and
  artifacts.
- Documentation avoids causal, privacy, legal, or fairness guarantees not
  established by evidence.

### Phase 8 — Notebook and documentation migration

Deliverables:

- Rewrite Titanic and California Housing notebooks with canonical Polars flows.
- Add a focused pandas-via-Arrow interoperability example without making pandas
  a documentation dependency for the primary quickstart.
- Replace manual dataframe filtering with `get_slice` or membership APIs.
- Demonstrate primary plots, validation, diversity, stability, comparison, and
  search diagnostics.
- Add migration guides for pandas-oriented idioms and the inherited prototype
  result attributes.
- Expand API and limitations documentation.

Acceptance gate:

- Primary notebooks execute from a clean locked environment without pandas.
- The isolated pandas interoperability example passes through the same canonical
  Arrow/Polars boundary.
- Stored outputs are cleared or deterministically generated according to the
  repository policy.
- Documentation links, code blocks, spelling, and warnings-as-errors build pass.

### Phase 9 — Weighted scoring research gate

Deliverables:

- Specify sample-weight and alternate-baseline semantics in an ADR.
- Prove or replace every pruning upper bound affected by weighting.
- Implement only approved semantics with brute-force and cross-engine parity
  or record the feature as rejected/deferred with evidence.

Acceptance gate:

- Weighted optimized search matches exhaustive enumeration on generated small
  lattices and boundary cases.
- Weight type/domain, weighted support, baseline, and serialization semantics
  are documented and tested.
- Unweighted behavior remains unchanged.

### Phase 10 — Release hardening and staged publication

Deliverables:

- Consolidate CI and release workflows around the frozen lock and full gate.
- Pin actions to full commit SHAs with update comments.
- Build once; inspect, smoke-test, checksum, generate SBOM/provenance, sign, and
  publish the same artifacts.
- Validate tag/version/changelog/artifact-schema consistency.
- Test on supported Python versions, with and without Numba, core-only, Arrow
  interoperability, and plotting extras.
- Publish a release candidate and complete the release-readiness checklist.

Acceptance gate:

- Clean wheel/sdist smoke tests pass on the support matrix.
- TestPyPI installation and documented Polars, PyArrow, and pandas-interchange
  quickstarts pass.
- Release rollback/forward-fix procedure is documented and exercised without
  attempting to overwrite an immutable published version.
- An accountable maintainer approves remaining risks.

## 10. Verification strategy

### 10.1 Test layers

| Layer | Required evidence |
|---|---|
| Unit | Validation, schemas, metrics, canonical serialization, plot-data transforms, bounds, and errors. |
| Property | Membership equivalence, overlap invariants, stable IDs, formula domains, bin transforms, artifact validation, and container round trips. |
| Characterization | Current 17 experiments, tie expansion, ordering, and unknown-category behavior. |
| Cross-engine | NumPy fallback versus Numba for scores, pruning decisions, ranks, and result statistics. |
| Container contract | Polars input/output; PyArrow, pandas-interchange, and ndarray compatibility; named-column schema mismatch; chunk/null/dtype/copy behavior. |
| Leakage | Binning, selection, validation, and sensitivity operations learn only from their declared discovery/training partitions. |
| Statistical simulation | Synthetic null/signal, clustered/time-dependent samples, multiple comparisons, insufficient support, stability, and selection bias. |
| Exhaustive oracle | Optimized unweighted and any future weighted search match brute-force small-lattice enumeration. |
| Integration | Ingest → bin → fit → validate → select → analyze → compare → plot for classification and regression fixtures. |
| Artifact security | Safe round trip plus malformed, oversized, hash-mismatched, path, schema-version, and decompression/resource abuse cases. |
| Package artifact | Install built wheel in isolated core-only, Arrow-interoperability, and plot environments; run documented smoke cases. |
| Documentation | Sphinx warnings-as-errors and clean execution of maintained examples/notebooks. |
| Performance | Controlled fit/transform, conversion allocation, overlap, and plot-data benchmarks with retained environment metadata. |

### 10.2 Numerical assertions

- Use exact equality only for Boolean membership, integer counts, canonical
  predicates, and deterministic identifiers.
- Use documented `rtol`/`atol` for floating scores and derived metrics.
- Assert rank and pruning parity separately from approximate score equality.
- Cover very small positive baseline error, large finite error, ties, and mixed
  integer/float errors.
- Treat warnings, NaN, infinity, and overflow as explicit test outcomes rather
  than allowing `nan_to_num` to hide them.
- Test confidence/adjustment procedures against their declared estimands and
  preserve seeds, resampling units, failures, and effective sample sizes.
- Use time-forward or blocked fixtures whenever random exchangeability is not
  part of the declared simulation.

### 10.3 Performance and resource budgets

Capture a versioned baseline before Phase 1 using representative low-, medium-,
and high-cardinality fixtures. Record CPU, OS, Python, architecture, dependency
lock hash, warmup policy, repetitions, medians, dispersion, and peak resident
memory.

Provisional gates to ratify after baseline measurement:

- The Polars boundary must not worsen median `fit` time by more than 10% or peak
  memory by more than 15% for equivalent already-materialized data.
- Arrow, PyArrow, and pandas-interchange conversions each get separate zero-copy
  eligible and allocating benchmarks; results must not imply every producer is
  zero-copy.
- NumPy/scikit-learn compatibility paths must remain within the existing
  distribution of baseline performance.
- Result-frame construction scales linearly in returned slice/predicate count.
- Pairwise overlap is disabled above the default slice bound and reports its
  estimated pair count.
- Interactive plots render at most a documented point/node/cell budget by
  default; summaries still cover all data.
- Resampling and full-search permutation methods require explicit work/parallel
  limits and report incomplete iterations.
- Analysis artifact reads preflight declared row/byte limits before materializing
  tables.

Shared CI benchmark results are evidence, not a noisy hard threshold. A
controlled runner owns release-blocking performance comparisons.

## 11. CI and release target

The standard `make check` should be nonmutating and include:

```text
lock freshness
ruff lint
ruff format --check
type check
unit/property/integration tests with coverage
NumPy/Numba parity test
Polars/PyArrow/pandas-interchange parity test
leakage and statistical-simulation tests
Sphinx warnings-as-errors
wheel and sdist build
artifact metadata/content check
isolated wheel and SliceAnalysis round-trip smoke tests
dependency and license audit
```

Recommended workflow separation:

- `ci.yml`: fast quality gate on pull requests and `main` pushes.
- `compatibility.yml`: supported Python/OS/backend matrix.
- `docs.yml`: documentation build and optional preview artifact.
- `benchmark.yml`: manual/scheduled retained benchmark evidence.
- `release.yml`: tag-gated, environment-approved, build-once publication.
- `dependabot.yml`: scheduled action and Python dependency updates.

Every job gets explicit minimal permissions. Publication uses PyPI trusted
publishing. A release must never create a GitHub release in response to that same
release already existing.

## 12. Documentation deliverables

- README quickstart using Polars.
- Installation matrix: core, optimized, plotting, and development.
- Data contract: categorical/discretized expectations, dtypes, nulls, errors,
  cardinality, row alignment, and schema preservation.
- Arrow/PyArrow/pandas interoperability and copy-semantics guide.
- Discretization-plan reference covering fit/transform leakage controls,
  intervals, unknowns, nulls, and serialization.
- Result schema and derived metric formulas.
- Discovery-versus-validation guide, interval assumptions, multiplicity,
  stability, fragility, and descriptive-result labels.
- Diversity-selection and model/time comparison guide.
- `SliceAnalysis` artifact format, compatibility, safe-loading limits, and
  provenance fields.
- Visualization guide explaining every encoding.
- Error-dependence interpretation and non-causality warning.
- Performance/resource-limits guide.
- NumPy/pandas-oriented migration guide.
- Reproducible benchmark guide.
- Security and responsible disclosure policy.
- Changelog and release process.

## 13. Risks and mitigations

| Risk | Mitigation | Rollback/stop condition |
|---|---|---|
| Polars conversion changes category or null semantics | Explicit dtype matrix, golden fixtures, canonical serialization tests | Stop support for an unproven dtype; do not coerce silently. |
| Arrow/pandas producer semantics differ or require hidden copies | Normalize once through public interchange, test producer/dtype matrix, expose copy behavior | Reject the dtype/protocol version rather than add pandas-specific internals. |
| Type-preserving `transform` surprises scikit-learn pipelines | Test `set_output`, document container rules, retain ndarray compatibility | Default compatibility path to ndarray until sklearn contract tests pass. |
| Plotly expands install size | Optional extra and lazy import | Ship plot-data builders first if renderer dependency is unacceptable. |
| Overlap/lattice views exhaust memory | Preflight estimates, conservative defaults, deterministic pruning | Reject above bounds; never auto-override. |
| Raw dependence plots leak sensitive row data in HTML | Deterministic sampling, field allowlist, export warning | Disable raw points and ship aggregate-only mode by default if review requires it. |
| Stable IDs change across versions | Version canonical serialization and test fixtures | Keep serialization version in metadata; provide migration mapping. |
| In-sample winners appear statistically reliable | Separate validation, multiplicity/stability metadata, discovery labels | Withhold inferential labels when assumptions or untouched data are absent. |
| Supervised binning or repeated tuning leaks validation outcomes | Fit transforms inside training folds and retain partition provenance | Block publication of affected validation claims and rerun from an untouched assessment set. |
| Diversity filtering hides the highest-score result | Preserve raw ranking and emit exclusion/selection reasons | Disable post-selection by default until deterministic parity and audit outputs pass. |
| Analysis artifacts become an unsafe serialization channel | Declarative JSON/Parquet only, closed schemas, hashes, bounds, no pickle/code | Fail closed on unknown versions, unexpected files, or exceeded limits. |
| Model/time comparison matches incompatible populations | Require schema/bin-plan/reference-population compatibility | Return non-comparable rather than coercing or forcing a match. |
| Weighting invalidates pruning correctness | Separate ADR and exhaustive oracle | Do not ship weights until every bound is proven or replaced. |
| Inherited result attributes create two sources of truth | Keep them temporary, derive from canonical frames during transition, then remove before stable release | Delay the stable release rather than ship divergent public state. |
| Numba and NumPy take different pruning paths near float boundaries | Rank/pruning parity corpus and boundary tolerances | Block release on any unexplained slice-set divergence. |
| Repository hardening overwhelms feature review | Small ordered pull requests with independent acceptance gates | Stop each phase at its gate; do not batch unrelated policy changes. |

## 14. Required decisions before Phase 1 merge

The plan recommends these defaults; maintainers must record acceptance or an
alternative in ADRs:

1. Polars is the required canonical table dependency, not an optional adapter.
   PyArrow and pandas inputs are supported through public Arrow/dataframe
   interchange; Ginsu does not import pandas.
2. Plotly is optional and plot-data builders remain usable without it.
3. `pl.LazyFrame` is explicitly unsupported initially.
4. Polars input produces Polars output; ndarray input preserves ndarray output,
   subject to scikit-learn output configuration.
5. Inherited `top_slices_` and `top_slices_statistics_` are transitional only
   and do not create a long-term compatibility obligation for this new project.
6. Error vectors are finite, nonnegative realized losses with positive mean.
7. Error-dependence plots are observational summaries, not PDPs.
8. Pairwise and browser-facing visual operations have conservative hard
   defaults with explicit overrides.
9. Discovery score is not inferential evidence; validation, uncertainty, and
   multiplicity policies remain explicit.
10. Discretization is fitted only on discovery/training data and is serialized
    declaratively.
11. Diverse selection is an auditable post-processing view and never replaces
    raw score ranking.
12. `SliceAnalysis` uses a closed non-executable format; pickle is not a trusted
    interchange format.
13. Sample weighting and alternate baselines remain gated on formula and
    pruning-bound approval.

## 15. Definition of done

This initiative is complete only when:

- all Phase 0 correctness blockers are fixed and regression-tested;
- Polars is the documented and tested canonical named-table interface;
- maintained code, examples, and plots run without pandas, while supported
  pandas and PyArrow inputs pass through the same Arrow/Polars boundary;
- fitted discretization is reproducible, leakage-safe, and artifact-backed;
- canonical result schemas and derived metric formulas are versioned and tested;
- discovery, validation, multiplicity, diversity, stability, and fragility
  statuses are distinguishable and tested;
- comparisons across models/time reject incompatible semantics and retain their
  reference-population/window contract;
- search completion, limits, backend, candidates, and copy boundaries are
  available through a structured report;
- safe `SliceAnalysis` artifacts round-trip across the supported schema version;
- impact, predicate-matrix, overlap, lattice, error-dependence, stability,
  comparison, and search-profile plots satisfy their semantic, resource,
  privacy, and accessibility contracts;
- the full locked quality gate passes from a clean checkout;
- wheel and sdist are tested as installed artifacts across the support matrix;
- applicable production-template repository, security, dependency, and release
  controls are in place;
- compatibility and migration behavior are documented;
- benchmark evidence shows guardrails are met or an accountable maintainer has
  accepted a quantified exception; and
- release-readiness review records approval and all remaining risks have an
  owner and reconsideration trigger.

## 16. Suggested pull-request sequence

1. `chore: establish reproducible quality and release baseline`
2. `fix: enforce named schema and error/support invariants`
3. `feat: add Polars and Arrow interchange boundary`
4. `feat: add typed Predicate, Slice, and result frames`
5. `test: add producer, container, and compute-engine parity corpus`
6. `feat: add fitted Polars discretization plans`
7. `feat: add bounded search diagnostics and resource controls`
8. `feat: add safe versioned SliceAnalysis artifacts`
9. `feat: add membership, overlap, equivalence, and lattice data builders`
10. `feat: add holdout validation and statistical metadata`
11. `feat: add diversity selection and stability analysis`
12. `feat: add impact and predicate matrix plotting`
13. `feat: add bounded dependence, overlap, lattice, and stability plots`
14. `feat: add model and time comparison`
15. `feat: add privacy-aware export and comparison plots`
16. `research: prove or reject weighted scoring and pruning`
17. `docs: migrate notebooks and publish interpretation guides`
18. `release: validate artifacts and publish release candidate`

Each pull request must name its affected requirements, add or update acceptance
evidence, pass the nonmutating gate, and leave unrelated later phases out of its
diff.
