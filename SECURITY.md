# Security Policy

## Supported versions

Ginsu has not published its first independent release, so there is no supported
release line yet. Pre-release source snapshots and build-only candidates are
provided for evaluation and receive fixes on a best-effort basis. This table
will identify supported versions before publication.

## Reporting a vulnerability

Do not disclose suspected vulnerabilities in a public issue. Use the
[private security advisory form](https://github.com/joshuamyers22/ginsu/security/advisories/new).

Do not include secrets, proprietary datasets, raw model-debugging rows, or
sensitive slice values in a report. Use a minimal synthetic reproducer.

Include the affected Ginsu version or commit, Python version, operating system,
dependency-lock state, expected impact, and minimal reproduction steps when
they are safe to share. The project does not yet promise a response SLA; the
maintainer will acknowledge and coordinate privately as capacity permits.

## Current security boundary

Ginsu is an in-process analysis library. Primary risks include resource
exhaustion from adversarial cardinality, sensitive subgroup disclosure in
exports, untrusted artifact loading, dependency compromise, and publication of
unverified distributions. These risks and required controls are tracked in
`docs/POLARS_VISUALIZATION_PRODUCTION_PLAN.md`.

Release-candidate workflow artifacts include SHA-256 checksums, a CycloneDX
dependency SBOM, and unsigned build-provenance metadata. They are retained as
CI evidence only: they are not signed, published to a package index, or an
assurance that a release is supported. See `docs/RELEASE_PROCESS.md` and
`docs/RELEASE_READINESS.md`.
