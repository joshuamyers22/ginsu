# ADR 0015: Deterministic offline Polars notebooks

- Status: Accepted
- Date: 2026-09-10

## Context

The inherited Titanic and California Housing notebooks downloaded data at run
time, used pandas and OptBinning, learned error-supervised bins, and evaluated
discovered rules on observations also used for model fitting and discovery.
That made clean execution network-dependent and combined several sources of
optimism in the displayed results.

Committing full third-party dataset snapshots would add provenance, licensing,
update, and repository-size obligations unrelated to Ginsu's API contract.

## Decision

Keep two domain-shaped tutorials but use clearly labeled, deterministically
generated synthetic passenger and housing fixtures. The notebooks do not claim
to reproduce the historical Titanic or California Housing datasets.

Each notebook uses three disjoint partitions: model training, slice discovery,
and fixed-rule validation. Per-row loss is computed outside model training.
``DiscretizationPlan`` learns error-independent boundaries from discovery
features and transforms validation features without refitting. Validation does
not rediscover or rerank rules.

The primary notebook environment contains the core package plus the
``notebooks`` and ``plot`` extras. It excludes pandas, PyArrow, OptBinning, and
Matplotlib. Pandas/PyArrow producer interoperability remains a separate
documented example and compatibility-test dependency group.

Checked-in notebook outputs stay empty. ``make execute-notebooks`` executes
offline and writes generated notebooks only beneath ignored ``docs/build``.
CI executes both notebooks on Python 3.12 and verifies pandas is absent first.
Static tests enforce the cleared-output, Polars, offline, and critical-journey
contracts.

## Consequences

- A clean checkout can execute the primary tutorials without external data or
  pandas.
- Examples demonstrate leakage boundaries and fixed-rule validation directly.
- Seeded synthetic fixtures make regression failures reproducible.
- The tutorials teach Ginsu workflows rather than make empirical claims about
  the original named datasets.
- Users seeking results on real data must supply governed datasets and their own
  privacy, sampling, provenance, and validation decisions.
