# ADR 0016: Build-only release candidates

- Status: Accepted
- Date: 2026-09-10

## Context

The inherited release automation rebuilt distributions in separate workflows,
used mutable action references, and coupled irreversible publication to
incomplete artifact validation. Ginsu does not yet have approved package-index
authority, a trusted-publisher environment, or a signed-attestation policy.

Release evidence must describe one exact wheel and source distribution. A
matrix that independently rebuilds packages can validate several environments
while accidentally testing different bytes. Publication automation would also
cross the project owner's currently approved boundary.

## Decision

Use one build-only release-candidate workflow. It may run manually for a stated
version or on a matching version tag, but it never publishes a package or
creates a GitHub release.

The build job:

1. Uses the committed lock, an exact build-backend version, and commit-pinned
   actions.
2. Validates project name, semantic version, declared Python range, changelog
   heading, and tag/version equality when invoked by a tag.
3. Runs the complete quality and notebook gates.
4. Builds exactly one wheel and one source distribution.
5. Validates archive paths, package contents, core metadata, dependencies, and
   optional extras.
6. Generates a locked CycloneDX dependency SBOM, an unsigned provenance record,
   and a SHA-256 checksum manifest.
7. Retains the complete candidate bundle as a time-bounded CI artifact.

Every smoke job downloads and verifies that same bundle, creates its dependency
profile from the committed lock, and installs the artifact without resolving a
different graph. The declared matrix is CPython 3.10, 3.11, and 3.12. Core
wheels run on Linux, macOS, and Windows; Numba runs on Linux for all supported
Python versions; pandas/PyArrow
compatibility and Plotly run on Linux with Python 3.12; and the source
distribution receives a Linux core smoke test on Python 3.12.

Publishing stays absent from the workflow. A later publication change requires
an explicit owner decision, protected environment approval, PyPI trusted
publishing, signed attestations, TestPyPI evidence, and an updated readiness
record. Published versions are immutable and receive a new forward-fix version
rather than an attempted overwrite.

## Consequences

- All installation evidence refers to identical distribution bytes.
- A tag can produce auditable candidate evidence without publishing anything.
- Action and resolver updates arrive as reviewable dependency pull requests.
- The candidate provenance records source and hashes but is explicitly not a
  signed supply-chain attestation.
- macOS and Windows currently cover the core wheel rather than every optional
  dependency profile; the narrower extras coverage remains visible.
- Publication is a deliberate future phase, not an accidental side effect of
  validation.
