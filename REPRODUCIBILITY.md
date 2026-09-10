# Reproducibility

## Supported workflow

Ginsu uses `uv` as its dependency resolver and environment runner. The committed
`uv.lock` is authoritative once generated for the renamed project.

```sh
uv sync --frozen --all-extras
make check
```

Do not hand-edit `uv.lock`. Update dependencies deliberately with `uv lock`,
review the manifest and lock diff together, then rerun the complete gate.

## Numerical reproducibility

- Deterministic fixtures define expected slice rules, order, membership, and
  statistics.
- NumPy and optional Numba paths require parity evidence.
- Floating comparisons use explicit tolerances where exact representation is
  not part of the contract.
- Performance results must record code revision, lock hash, Python, OS,
  architecture, CPU, warmup, repetitions, and workload.

## Data and notebooks

Maintained examples must use public datasets or synthetic fixtures, pin dataset
identity where possible, and keep learned preprocessing inside the declared
training/discovery partition. Do not commit private data or environment-specific
notebook state.

## Releases

Build wheels and source distributions from a clean checkout after the full
locked gate. Smoke-test the built artifact in an isolated environment before any
publication. Publication remains disabled until the independent remote and
trusted publisher are explicitly configured.
