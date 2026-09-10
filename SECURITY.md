# Security Policy

## Supported versions

Ginsu has not published its first independent release. Security support begins
with the first documented release candidate.

## Reporting a vulnerability

Do not disclose suspected vulnerabilities in a public issue. Use the
[private security advisory form](https://github.com/joshuamyers22/ginsu/security/advisories/new).

Do not include secrets, proprietary datasets, raw model-debugging rows, or
sensitive slice values in a report. Use a minimal synthetic reproducer.

## Current security boundary

Ginsu is an in-process analysis library. Primary risks include resource
exhaustion from adversarial cardinality, sensitive subgroup disclosure in
exports, untrusted artifact loading, dependency compromise, and publication of
unverified distributions. These risks and required controls are tracked in
`docs/POLARS_VISUALIZATION_PRODUCTION_PLAN.md`.
