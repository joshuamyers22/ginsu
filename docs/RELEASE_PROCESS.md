# Release process

Ginsu currently produces build-only release candidates. The automation cannot
publish to PyPI or TestPyPI and cannot create a GitHub release. The
machine-readable contract in `.github/release-policy.toml` sets publication to
false, and `scripts/release_policy.py` enforces that state in CI.

## Candidate prerequisites

- Work from a reviewed commit on `main` with a clean, current lockfile.
- Keep the project version in `pyproject.toml` and a dated version heading in
  `CHANGELOG.md` consistent. The `Unreleased` heading remains for work after
  that version.
- Ensure `make check`, `make test`, and `make execute-notebooks` pass.
- Review `docs/RELEASE_READINESS.md`; unresolved publication blockers may remain
  for a build-only candidate, but they prevent publication.

## Build-only automation

Run **Build release candidate** manually with the exact expected version, or
push an annotated `v<version>` tag only after the corresponding changelog entry
exists. Ordinary changes and pull requests must not create release tags.

The workflow runs with read-only repository permissions and performs these
steps in order:

1. Install the pinned `uv` release and sync every dependency, including the
   exact Hatchling build backend, from `uv.lock`.
2. Validate project, version, Python support, tag, and changelog identity.
3. Validate the release workflow against the publication policy.
4. Run code, type, test, documentation, and notebook gates.
5. Build one wheel and one source distribution.
6. Reject unexpected, unsafe, or inconsistent archive contents and metadata.
7. Create a CycloneDX 1.5 dependency SBOM and unsigned provenance record.
8. Create and immediately verify a `SHA256SUMS` manifest.
9. Upload the candidate bundle with 30-day retention.
10. Download the same bundle in every smoke job, verify its checksums, create the
   profile from `uv.lock`, install the artifact with dependency resolution
   disabled, and exercise the selected profile.

The smoke matrix is:

| Artifact/profile | Python | Operating system |
|---|---|---|
| Core wheel | 3.10, 3.11, 3.12 | Linux, macOS, Windows |
| Numba-optimized wheel | 3.10, 3.11, 3.12 | Linux |
| pandas/PyArrow compatibility wheel | 3.12 | Linux |
| Plotting wheel | 3.12 | Linux |
| Core source distribution | 3.12 | Linux |

The bundle contains the wheel, source distribution, CycloneDX SBOM, provenance
JSON, and checksum manifest. The provenance is useful build metadata, not a
cryptographic signature or SLSA attestation.

## Review a candidate

- Require every build and smoke job to pass without retrying around a product
  failure.
- Download the workflow artifact and verify it with
  `python scripts/release_tools.py verify-checksums --directory <directory>`.
- Confirm the artifact source SHA, ref, run identifier, lock hash, filenames,
  and hashes in the provenance record.
- Confirm the readiness record reflects the exact candidate and accountable
  approval decision.

## Publication gate

Publication activation is a reviewed code and operations change, not an effect
of creating a tag. Do not set `publication.enabled = true` or add publication
jobs until all of the following are separately approved and tested:

- PyPI project ownership and a least-privilege trusted-publisher environment.
- Protected GitHub environment with accountable maintainer approval.
- Signed build attestation and a documented verification path.
- Installation and quickstart validation from TestPyPI for Polars, PyArrow,
  pandas-interchange, plotting, and optimized profiles.
- Final license, vulnerability, dependency, and release-readiness review.

The enabled workflow contract is fail-closed:

- Production accepts only a version-tag push and uses the protected `pypi`
  environment. TestPyPI accepts only an explicit manual `testpypi` target and
  uses the protected `testpypi` environment.
- Both publication jobs depend on build, smoke, and preparation gates, download
  the same named staged artifact, and never rebuild distributions.
- The trusted-publisher action and every other action use full commit SHAs.
  Stored API tokens, `skip-existing`, and arbitrary shell commands in an OIDC
  publication job are forbidden.
- `id-token: write` is job-scoped; it is never granted at workflow scope. The
  publisher's artifact attestations remain enabled.

Repository tests verify this workflow shape. They cannot verify GitHub
environment reviewers or the PyPI/TestPyPI trusted-publisher registrations, so
those controls must be inspected and recorded in the readiness review. Adding
publication still requires explicit owner approval.

## Failure and forward fix

Before publication, discard a failed candidate, fix the source, and create new
candidate evidence from the new commit. Never replace files inside an existing
candidate and treat them as the same build.

After any future package-index publication, do not delete and overwrite the
version. Yank it if appropriate, document the issue, make the smallest safe
forward fix, increment the version, rebuild from the corrected commit, and run
the entire process again. GitHub release assets must correspond exactly to the
published package bytes and checksums.
