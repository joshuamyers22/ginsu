# ADR 0017: Trusted-publication policy as code

- Status: Accepted
- Date: 2026-09-11

## Context

ADR 0016 deliberately removed publication from release automation. Ginsu now
needs a reviewable contract for adding PyPI and TestPyPI publication without
weakening the build-once candidate guarantees or giving every release job an
identity token. A prose checklist alone cannot prevent a later workflow edit
from adding a stored API token, publishing a manually selected production
version, rebuilding distributions, or bypassing smoke tests.

GitHub environment protection and PyPI trusted-publisher registration are
external controls. Repository tests cannot prove those settings exist, but
they can constrain the workflow that will use them and keep publication
disabled until the external readiness record is complete.

## Decision

Store the closed-schema activation contract in
`.github/release-policy.toml` and enforce it with
`scripts/release_policy.py` in tests and in the release build job.

While `publication.enabled` is false, the release workflow must contain no
PyPI publisher action, package-index command, stored credential reference,
GitHub Release command, or OIDC write permission.

Any reviewed change that enables publication must satisfy all of these
conditions:

1. Build the wheel and source distribution exactly once, then reuse one named
   prepared artifact after the build, smoke, and checksum gates.
2. Use only full-commit action references and the PyPA trusted-publisher
   action; never use an API token or `skip-existing` behavior.
3. Grant `id-token: write` only inside minimal publication jobs that execute no
   arbitrary shell commands.
4. Restrict production PyPI publication to version-tag push events and the
   protected `pypi` GitHub environment.
5. Restrict TestPyPI publication to an explicit manual target and the
   separately protected `testpypi` environment.
6. Leave the trusted-publisher action's attestations enabled.

Changing the Boolean is not sufficient authorization. The owner must first
complete the external configuration and record the exact candidate evidence
and accountable approval in `docs/RELEASE_READINESS.md`.

## Consequences

- The repository has executable regression coverage for the disabled posture
  and the future enabled posture before any publishing capability is added.
- Tests reject workflow-wide OIDC, mutable actions, stored secrets, unprotected
  environment names, manual production publishing, arbitrary commands in an
  OIDC job, rebuilt artifacts, skipped existing versions, and bypassed gates.
- GitHub and package-index settings still require human verification; policy
  code cannot inspect or create those external controls.
- Publication remains disabled until a later reviewed and explicitly approved
  workflow change satisfies both the repository policy and readiness record.
