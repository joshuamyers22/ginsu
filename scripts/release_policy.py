"""Validate Ginsu's release workflow against its publication policy."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class PublicationPolicyError(RuntimeError):
    """The release workflow violates the declared publication policy."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PublicationPolicyError(message)


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 CI
        import tomli as tomllib

    with path.open("rb") as stream:
        return tomllib.load(stream)


@dataclass(frozen=True)
class PublicationPolicy:
    """Machine-readable activation contract for trusted publishing."""

    enabled: bool
    action_repository: str
    production_job: str
    production_environment: str
    production_ref_prefix: str
    test_job: str
    test_environment: str
    test_input: str
    test_input_value: str
    preparation_job: str
    publication_artifact: str
    publication_directory: str
    required_needs: tuple[str, ...]
    require_job_scoped_oidc: bool
    require_build_once: bool
    require_attestations: bool
    allow_api_tokens: bool
    allow_skip_existing: bool


def load_policy(root: Path) -> PublicationPolicy:
    """Load and strictly validate the repository publication policy."""
    document = _load_toml(root / ".github/release-policy.toml")
    _require(document.get("schema_version") == 1, "Unsupported policy schema.")
    _require(
        document.get("project") == "ginsu", "Policy project must be ginsu."
    )
    _require(
        document.get("workflow") == ".github/workflows/release.yml",
        "Policy must govern the release workflow.",
    )
    publication = document.get("publication")
    _require(isinstance(publication, dict), "Policy has no publication table.")

    expected = {
        "enabled",
        "action_repository",
        "production_job",
        "production_environment",
        "production_ref_prefix",
        "test_job",
        "test_environment",
        "test_input",
        "test_input_value",
        "preparation_job",
        "publication_artifact",
        "publication_directory",
        "required_needs",
        "require_job_scoped_oidc",
        "require_build_once",
        "require_attestations",
        "allow_api_tokens",
        "allow_skip_existing",
    }
    _require(
        set(publication) == expected,
        "Publication policy fields differ from the supported closed schema.",
    )
    _require(
        isinstance(publication["enabled"], bool),
        "publication.enabled must be Boolean.",
    )
    strings = expected - {
        "enabled",
        "required_needs",
        "require_job_scoped_oidc",
        "require_build_once",
        "require_attestations",
        "allow_api_tokens",
        "allow_skip_existing",
    }
    _require(
        all(
            isinstance(publication[field], str) and publication[field]
            for field in strings
        ),
        "Publication policy string fields must be non-empty.",
    )
    boolean_fields = (
        "require_job_scoped_oidc",
        "require_build_once",
        "require_attestations",
        "allow_api_tokens",
        "allow_skip_existing",
    )
    _require(
        all(isinstance(publication[field], bool) for field in boolean_fields),
        "Publication policy control fields must be Boolean.",
    )
    needs = publication["required_needs"]
    _require(
        isinstance(needs, list)
        and needs
        and all(isinstance(item, str) and item for item in needs),
        "publication.required_needs must be a non-empty string list.",
    )
    _require(
        len(needs) == len(set(needs)),
        "publication.required_needs contains duplicates.",
    )
    return PublicationPolicy(**{**publication, "required_needs": tuple(needs)})


def _job_block(workflow: str, name: str) -> str:
    lines = workflow.splitlines(keepends=True)
    jobs_index = next(
        (
            index
            for index, line in enumerate(lines)
            if line.rstrip() == "jobs:"
        ),
        None,
    )
    _require(jobs_index is not None, "Release workflow has no jobs mapping.")
    header = re.compile(rf"^  {re.escape(name)}:\s*(?:#.*)?$")
    start = next(
        (
            index
            for index in range(jobs_index + 1, len(lines))
            if header.match(lines[index].rstrip("\n"))
        ),
        None,
    )
    _require(start is not None, f"Release workflow has no {name!r} job.")
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if re.match(
            r"^  [A-Za-z0-9_-]+:\s*(?:#.*)?$", lines[index].rstrip("\n")
        ):
            end = index
            break
    return "".join(lines[start:end])


def _job_property(job: str, name: str) -> str:
    lines = job.splitlines(keepends=True)
    header = re.compile(rf"^    {re.escape(name)}:\s*(.*)$")
    for start, line in enumerate(lines):
        match = header.match(line.rstrip("\n"))
        if match is None:
            continue
        value = match.group(1) + "\n"
        for following in lines[start + 1 :]:
            if re.match(r"^    [A-Za-z0-9_-]+:\s*", following):
                break
            value += following
        return value
    raise PublicationPolicyError(f"Job has no {name!r} property.")


def _require_pinned_actions(workflow: str) -> None:
    actions = re.findall(r"^\s*-\s+uses:\s+([^\s#]+)", workflow, re.MULTILINE)
    _require(bool(actions), "Release workflow uses no actions.")
    _require(
        all(re.search(r"@[0-9a-f]{40}$", action) for action in actions),
        "Every release action must be pinned to a full lowercase commit SHA.",
    )


def _require_no_credentials(workflow: str) -> None:
    lowered = workflow.lower()
    forbidden = (
        "secrets.",
        "pypi_token",
        "uv_publish_token",
        "twine_password",
        "password:",
    )
    _require(
        not any(value in lowered for value in forbidden),
        "Release workflow must not use stored publication credentials.",
    )


def _require_environment(job: str, expected: str) -> None:
    environment = _job_property(job, "environment")
    _require(
        re.search(
            rf"(?:^|\s)name:\s*{re.escape(expected)}(?:\s|$)", environment
        )
        is not None
        or environment.strip() == expected,
        f"Publication job must use the {expected!r} environment.",
    )


def _condition(job: str) -> str:
    value = _job_property(job, "if").strip()
    if value.startswith("${{") and value.endswith("}}"):
        value = value[3:-2].strip()
    return " ".join(value.split())


def _require_exact_needs(job: str, expected: tuple[str, ...]) -> None:
    value = _job_property(job, "needs")
    observed = re.findall(r"[A-Za-z0-9_-]+", value)
    _require(
        len(observed) == len(set(observed)) and set(observed) == set(expected),
        "Publication job dependencies must exactly match every required "
        "candidate gate.",
    )


def _validate_publish_job(
    workflow: str,
    *,
    name: str,
    environment: str,
    policy: PublicationPolicy,
) -> str:
    job = _job_block(workflow, name)
    _require_environment(job, environment)
    permissions = _job_property(job, "permissions")
    _require(
        " ".join(permissions.split()) == "id-token: write",
        "Publication jobs must grant only job-scoped OIDC write permission.",
    )
    _require(
        "run:" not in job,
        "Publication jobs must not execute arbitrary shell commands.",
    )
    _require(
        re.search(
            rf"uses:\s+{re.escape(policy.action_repository)}@[0-9a-f]{{40}}",
            job,
        )
        is not None,
        "Publication job must use the pinned trusted-publisher action.",
    )
    _require(
        re.search(r"uses:\s+actions/download-artifact@[0-9a-f]{40}", job)
        is not None,
        "Publication job must download the previously built artifact.",
    )
    _require(
        f"name: {policy.publication_artifact}" in job,
        "Publication job must download the prepared publication artifact.",
    )
    _require(
        f"path: {policy.publication_directory}/" in job
        and f"packages-dir: {policy.publication_directory}/" in job,
        "Publication job must publish only from the staged directory.",
    )
    if policy.require_attestations:
        _require(
            "attestations: true" in job,
            "Publication job must explicitly enable attestations.",
        )
    return job


def validate_workflow_policy(workflow: str, policy: PublicationPolicy) -> None:
    """Validate workflow source under either disabled or enabled policy."""
    _require_pinned_actions(workflow)
    _require_no_credentials(workflow)
    if policy.require_build_once:
        _require(
            workflow.count("uv build --clear") == 1,
            "Release workflow must build distributions exactly once.",
        )

    lowered = workflow.lower()
    if not policy.enabled:
        forbidden = (
            policy.action_repository.lower(),
            "uv publish",
            "twine upload",
            "gh release",
            "id-token: write",
        )
        _require(
            not any(value in lowered for value in forbidden),
            "Publication-disabled workflow contains publication capability.",
        )
        return

    _require(
        policy.require_job_scoped_oidc,
        "Enabled publication requires job-scoped OIDC policy.",
    )
    _require(
        policy.require_attestations,
        "Enabled publication requires attestations.",
    )
    _require(
        not policy.allow_api_tokens, "API-token publication is forbidden."
    )
    _require(
        not policy.allow_skip_existing,
        "Enabled publication must fail on an existing version.",
    )
    workflow_prefix = workflow.split("\njobs:", maxsplit=1)[0]
    _require(
        "id-token: write" not in workflow_prefix,
        "OIDC permission must not be granted at workflow scope.",
    )
    _require(
        "skip-existing: true" not in lowered,
        "Publishing must not silently skip an existing version.",
    )
    preparation = _job_block(workflow, policy.preparation_job)
    preparation_needs = _job_property(preparation, "needs")
    _require(
        all(
            re.search(rf"\b{re.escape(name)}\b", preparation_needs)
            for name in ("build", "smoke")
        ),
        "Publication preparation must follow build and smoke gates.",
    )
    _require(
        "verify-checksums" in preparation,
        "Publication preparation must verify the candidate checksums.",
    )
    _require(
        re.search(r"uses:\s+actions/upload-artifact@[0-9a-f]{40}", preparation)
        is not None
        and f"name: {policy.publication_artifact}" in preparation,
        "Publication preparation must stage one named artifact.",
    )
    _require(
        "uv build" not in preparation,
        "Publication preparation must not rebuild distributions.",
    )
    production = _validate_publish_job(
        workflow,
        name=policy.production_job,
        environment=policy.production_environment,
        policy=policy,
    )
    condition = _condition(production)
    expected_condition = (
        "github.event_name == 'push' && "
        f"startsWith(github.ref, '{policy.production_ref_prefix}')"
    )
    _require(
        condition == expected_condition,
        "Production publication must be restricted to version-tag pushes.",
    )
    _require_exact_needs(production, policy.required_needs)

    test = _validate_publish_job(
        workflow,
        name=policy.test_job,
        environment=policy.test_environment,
        policy=policy,
    )
    test_condition = _condition(test)
    expected_test_condition = (
        "github.event_name == 'workflow_dispatch' && "
        f"inputs.{policy.test_input} == '{policy.test_input_value}'"
    )
    _require(
        test_condition == expected_test_condition,
        "TestPyPI publication must require an explicit manual target.",
    )
    _require(
        "repository-url: https://test.pypi.org/legacy/" in test,
        "Test publication must target TestPyPI explicitly.",
    )
    _require_exact_needs(test, policy.required_needs)
    _require(
        "attestations: false" not in lowered,
        "Trusted-publisher attestations must remain enabled.",
    )


def validate_repository(root: Path) -> None:
    """Validate the configured release policy and workflow in a checkout."""
    policy = load_policy(root)
    workflow = (root / ".github/workflows/release.yml").read_text(
        encoding="utf-8"
    )
    validate_workflow_policy(workflow, policy)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    arguments = parser.parse_args(argv)
    try:
        validate_repository(arguments.root.resolve())
    except (OSError, KeyError, TypeError, PublicationPolicyError) as error:
        print(f"release policy failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
