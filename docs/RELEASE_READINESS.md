# Release readiness

**Candidate version:** 0.1.0

**Review date:** 2026-09-10

**Decision:** Not approved for publication

This record distinguishes implemented candidate controls from external
publication authority. A successful build-only workflow is necessary evidence,
but it is not permission to publish.

## Implemented controls

- [x] Independent `ginsu` distribution and import identity.
- [x] Declared CPython 3.10--3.12 support and bounded dependency ranges.
- [x] Frozen dependency lock and exact `uv` and Hatchling versions in CI.
- [x] GitHub Actions pinned to reviewed full commit SHAs.
- [x] Full quality, coverage, documentation, and offline-notebook gates.
- [x] One wheel and one source distribution built once per candidate.
- [x] Project, tag, changelog, metadata, dependency, extras, archive-path, and
  package-content validation.
- [x] Same-artifact, lock-backed installed smoke matrix for supported Python
  and operating systems plus core, optimized, compatibility, plotting, wheel,
  and sdist profiles.
- [x] CycloneDX dependency SBOM, unsigned provenance, and SHA-256 checksums.
- [x] Security, contribution, reproducibility, migration, limitations, and
  forward-fix guidance.
- [x] Scheduled GitHub Actions and `uv` dependency updates.
- [x] Publication commands and credentials absent from release automation.

## Publication blockers

- [ ] Project owner approves PyPI publication authority and namespace.
- [ ] PyPI trusted publisher is configured with a protected GitHub environment.
- [ ] An accountable maintainer approval gate is implemented and exercised.
- [ ] Signed artifact attestation is generated and its verification documented.
- [ ] TestPyPI installs and documented quickstarts pass for every required
  interoperability and optional-dependency profile.
- [ ] A final candidate run is linked here with source SHA, tag, workflow run,
  artifact retention deadline, and reviewer.
- [ ] Final dependency, vulnerability, license, changelog, and known-risk review
  is recorded for that exact candidate.

## Known residual limits

- Candidate provenance is unsigned and must not be represented as an attested
  supply-chain statement.
- Optional dependency profiles run on Linux; macOS and Windows exercise the
  core wheel only.
- CI candidate artifacts expire after 30 days and are not a durable release
  archive.
- Ginsu remains pre-release software with no supported security release line or
  response-time SLA.

The next review is triggered by explicit publication approval or a material
change to version, dependencies, support matrix, artifact contents, security
posture, or release automation.
