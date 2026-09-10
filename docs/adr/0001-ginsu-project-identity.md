# ADR 0001: Establish Ginsu as an Independent Project

- Status: Accepted
- Date: 2026-09-09

## Context

The initial code is derived from DataDome's BSD-licensed Sliceline repository.
The project owner requested a new independent project named Ginsu rather than a
drop-in successor package.

## Decision

- Use `ginsu` as the repository directory, distribution, and Python import name.
- Start Ginsu at version `0.1.0`.
- Do not ship a `sliceline` compatibility import or top-level package.
- Preserve the upstream license notice, algorithm citation, and attribution.
- Keep the upstream Git remote fetch-only and defer an independent `origin`
  until the owner selects or creates the external repository.

## Consequences

Existing Sliceline imports do not work with Ginsu and must be changed
deliberately. This avoids two distributions owning the same Python import path
and gives Ginsu freedom to establish a coherent pre-1.0 API.
