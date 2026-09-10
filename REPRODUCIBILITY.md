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

The release-candidate workflow resolves the declared version, validates any tag
and changelog identity, runs the full locked gate, and builds exactly one wheel
and one source distribution. It validates their metadata and contents before
generating a CycloneDX dependency SBOM, unsigned provenance record, and SHA-256
checksum manifest.

Every supported Python, operating-system, and optional-dependency smoke job
downloads that same candidate bundle. No matrix job rebuilds a distribution.
This separates evidence about one candidate from environmental installation
coverage and prevents different builds from being treated as one release.

Publication remains disabled until trusted publishing, signed attestation,
TestPyPI validation, and owner approval are complete. The exact workflow and
forward-fix procedure are documented in `docs/RELEASE_PROCESS.md`; current
evidence and blockers are recorded in `docs/RELEASE_READINESS.md`.
