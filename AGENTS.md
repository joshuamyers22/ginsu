# Agent Working Agreement

Human instructions in the current task take precedence. Repository source,
tests, approved ADRs, and the project brief are authoritative.

## Retrieve before acting

1. Read `README.rst`, `docs/PROJECT_BRIEF.md`, and `PROJECT_MEMORY.md` before a
   substantial change.
2. Inspect the relevant implementation and tests before relying on a memory
   entry or plan statement.
3. Treat `PROJECT_MEMORY.md` as an evidence index, not as evidence by itself.

## Preserve project boundaries

- Ginsu is an independent project derived from the BSD-licensed Sliceline code.
- The import and distribution name is `ginsu`; do not add a `sliceline`
  compatibility package.
- Keep the upstream remote fetch-only. Creating or changing an external remote,
  repository, release, or package requires explicit user authority.
- Keep Polars as the planned canonical table model, with pandas and PyArrow
  supported through public Arrow/dataframe interchange rather than
  producer-specific imports.

## Work safely

- Preserve unrelated user changes and inspect the diff before completion.
- Use small behavior-preserving slices and add regression tests for defects.
- Do not weaken validation, numerical parity, resource bounds, coverage, or
  release controls to make a check pass.
- Keep generated data, environments, caches, secrets, and raw client data out of
  Git.
- Record consequential choices in `docs/adr/` and durable verified state in
  `PROJECT_MEMORY.md`.
- A check that was skipped, deselected, or blocked is not passing evidence.

## Verification

Run focused tests after each coherent change. Before handoff, run the applicable
nonmutating repository gate, build the distribution, inspect the final diff, and
report every unverified or failing check.
