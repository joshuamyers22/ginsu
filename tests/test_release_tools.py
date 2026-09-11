"""Release-candidate policy and evidence regression tests."""

from __future__ import annotations

import json
import re
import tarfile
import zipfile
from dataclasses import replace
from email.message import EmailMessage
from pathlib import Path

import pytest

from scripts.release_policy import (
    PublicationPolicyError,
    load_policy,
    validate_repository,
    validate_workflow_policy,
)
from scripts.release_tools import (
    CHECKSUM_FILE,
    ReleaseCheckError,
    project_version,
    validate_distributions,
    validate_source,
    verify_checksums,
    write_checksums,
    write_provenance,
)

ROOT = Path(__file__).resolve().parents[1]


def _metadata() -> bytes:
    message = EmailMessage()
    message["Metadata-Version"] = "2.5"
    message["Name"] = "ginsu"
    message["Version"] = "0.1.0"
    message["Requires-Python"] = "<3.13,>=3.10"
    for requirement in (
        "numpy<3,>=1.25",
        "polars<2,>=1.30",
        "scikit-learn<2,>=1.6.0",
        "scipy<2,>=1.12",
    ):
        message["Requires-Dist"] = requirement
    extras = {
        "compat": "pandas>=2.1.1",
        "dev": "pytest>=7.2.0",
        "notebooks": "jupyter>=1.0.0",
        "optimized": "numba>=0.60.0",
        "plot": "plotly<7,>=6",
    }
    for extra, requirement in extras.items():
        message["Provides-Extra"] = extra
        message["Requires-Dist"] = f"{requirement}; extra == '{extra}'"
    return message.as_bytes()


def _candidate_archives(tmp_path: Path) -> Path:
    directory = tmp_path / "candidate"
    directory.mkdir()
    wheel = directory / "ginsu-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("ginsu/__init__.py", "")
        archive.writestr("ginsu/py.typed", "")
        archive.writestr("ginsu-0.1.0.dist-info/METADATA", _metadata())
        archive.writestr("ginsu-0.1.0.dist-info/LICENSE", "BSD")

    source = tmp_path / "source"
    root = source / "ginsu-0.1.0"
    for relative in (
        "CHANGELOG.md",
        "LICENSE",
        "README.rst",
        "SECURITY.md",
        "pyproject.toml",
        "uv.lock",
        "ginsu/__init__.py",
        "ginsu/py.typed",
        "scripts/release_policy.py",
        "scripts/release_tools.py",
        "scripts/smoke_release.py",
        "tests/test_release_tools.py",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    with tarfile.open(directory / "ginsu-0.1.0.tar.gz", "w:gz") as archive:
        archive.add(root, arcname=root.name)
    return directory


def test_source_version_and_manual_candidate_contract() -> None:
    assert project_version(ROOT) == "0.1.0"
    assert validate_source(ROOT, expected_version="0.1.0") == "0.1.0"

    with pytest.raises(ReleaseCheckError, match="Expected version"):
        validate_source(ROOT, expected_version="0.2.0")
    with pytest.raises(ReleaseCheckError, match="Changelog needs"):
        validate_source(ROOT, ref="refs/tags/v0.1.0")


def test_distribution_metadata_and_contents_are_validated(
    tmp_path: Path,
) -> None:
    directory = _candidate_archives(tmp_path)

    wheel, sdist = validate_distributions(ROOT, directory)

    assert wheel.name.endswith(".whl")
    assert sdist.name.endswith(".tar.gz")


def test_distribution_validation_rejects_unsafe_archive_path(
    tmp_path: Path,
) -> None:
    directory = _candidate_archives(tmp_path)
    wheel = directory / "ginsu-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("../outside.txt", "unsafe")

    with pytest.raises(ReleaseCheckError, match="traverses its root"):
        validate_distributions(ROOT, directory)


def test_candidate_evidence_round_trip_detects_tampering(
    tmp_path: Path,
) -> None:
    directory = _candidate_archives(tmp_path)
    sbom = directory / "ginsu-0.1.0.cdx.json"
    sbom.write_text(
        json.dumps({"bomFormat": "CycloneDX", "specVersion": "1.5"}),
        encoding="utf-8",
    )
    provenance = write_provenance(
        ROOT,
        directory,
        commit="0" * 40,
        ref="refs/heads/main",
        run_id="123",
        run_attempt="1",
    )
    manifest = write_checksums(ROOT, directory)
    (directory / ".gitignore").write_text("*", encoding="ascii")
    verify_checksums(directory)

    evidence = json.loads(provenance.read_text(encoding="utf-8"))
    assert evidence["publication"] == "disabled"
    assert evidence["signed_attestation"] is False
    assert len(manifest.read_text(encoding="ascii").splitlines()) == 4

    wheel = directory / "ginsu-0.1.0-py3-none-any.whl"
    wheel.write_bytes(wheel.read_bytes() + b"tampered")
    with pytest.raises(ReleaseCheckError, match="Checksum mismatch"):
        verify_checksums(directory)


def test_candidate_checksum_rejects_modified_build_marker(
    tmp_path: Path,
) -> None:
    directory = _candidate_archives(tmp_path)
    (directory / "ginsu-0.1.0.cdx.json").write_text(
        json.dumps({"bomFormat": "CycloneDX", "specVersion": "1.5"}),
        encoding="utf-8",
    )
    write_provenance(
        ROOT,
        directory,
        commit="0" * 40,
        ref="refs/heads/main",
        run_id="123",
        run_attempt="1",
    )
    write_checksums(ROOT, directory)
    (directory / ".gitignore").write_text("secret", encoding="ascii")

    with pytest.raises(ReleaseCheckError, match="marker"):
        verify_checksums(directory)


def test_candidate_checksum_rejects_unlisted_directory(tmp_path: Path) -> None:
    directory = _candidate_archives(tmp_path)
    sbom = directory / "ginsu-0.1.0.cdx.json"
    sbom.write_text(
        json.dumps({"bomFormat": "CycloneDX", "specVersion": "1.5"}),
        encoding="utf-8",
    )
    write_provenance(
        ROOT,
        directory,
        commit="0" * 40,
        ref="refs/heads/main",
        run_id="123",
        run_attempt="1",
    )
    write_checksums(ROOT, directory)
    (directory / "unlisted").mkdir()

    with pytest.raises(ReleaseCheckError, match="directory contains"):
        verify_checksums(directory)


def test_workflows_pin_actions_to_immutable_commits() -> None:
    workflows = sorted((ROOT / ".github/workflows").glob("*.yml"))
    text = "\n".join(path.read_text(encoding="utf-8") for path in workflows)
    uses = re.findall(r"^\s*-\s+uses:\s+([^\s#]+)", text, re.MULTILINE)

    assert uses
    assert all(re.search(r"@[0-9a-f]{40}$", value) for value in uses)


def test_disabled_publication_policy_is_enforced() -> None:
    policy = load_policy(ROOT)

    assert policy.enabled is False
    assert policy.require_job_scoped_oidc is True
    assert policy.require_build_once is True
    assert policy.require_attestations is True
    assert policy.allow_api_tokens is False
    assert policy.allow_skip_existing is False
    validate_repository(ROOT)


def test_release_candidate_builds_once_and_reuses_artifacts() -> None:
    release = (ROOT / ".github/workflows/release.yml").read_text(
        encoding="utf-8"
    )
    assert release.count("uv build --clear") == 1
    assert "actions/download-artifact@" in release
    assert "uv sync --frozen --no-install-project" in release
    assert "uv pip install --python .venv --no-deps" in release
    assert "write-checksums" in release
    assert "cyclonedx1.5" in release
    assert "include-hidden-files: false" in release
    assert 'requires = ["hatchling==1.32.0"]' in (
        ROOT / "pyproject.toml"
    ).read_text(encoding="utf-8")
    assert not (ROOT / ".github/workflows/test-release.yml").exists()
    assert CHECKSUM_FILE == "SHA256SUMS"


def _enabled_workflow() -> str:
    sha = "0" * 40
    return f"""name: Release
on:
  workflow_dispatch:
    inputs:
      publish_target:
        type: choice
  push:
    tags:
      - "v[0-9]*"
permissions:
  contents: read
jobs:
  build:
    steps:
      - uses: actions/checkout@{sha}
      - run: uv build --clear
  smoke:
    needs: build
    steps:
      - uses: actions/download-artifact@{sha}
  prepare-publication:
    needs: [build, smoke]
    steps:
      - uses: actions/download-artifact@{sha}
      - run: python scripts/release_tools.py verify-checksums --directory candidate
      - uses: actions/upload-artifact@{sha}
        with:
          name: ginsu-publication-files
          path: publication/
  publish-pypi:
    if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')
    needs: [build, smoke, prepare-publication]
    environment:
      name: pypi
    permissions:
      id-token: write
    steps:
      - uses: actions/download-artifact@{sha}
        with:
          name: ginsu-publication-files
          path: publication/
      - uses: pypa/gh-action-pypi-publish@{sha}
        with:
          packages-dir: publication/
          attestations: true
  publish-testpypi:
    if: github.event_name == 'workflow_dispatch' && inputs.publish_target == 'testpypi'
    needs: [build, smoke, prepare-publication]
    environment: testpypi
    permissions:
      id-token: write
    steps:
      - uses: actions/download-artifact@{sha}
        with:
          name: ginsu-publication-files
          path: publication/
      - uses: pypa/gh-action-pypi-publish@{sha}
        with:
          repository-url: https://test.pypi.org/legacy/
          packages-dir: publication/
          attestations: true
"""


def test_enabled_policy_accepts_only_gated_trusted_publishing() -> None:
    policy = replace(load_policy(ROOT), enabled=True)

    validate_workflow_policy(_enabled_workflow(), policy)


@pytest.mark.parametrize(
    ("unsafe", "message"),
    [
        (
            lambda text: text.replace(
                "environment:\n      name: pypi",
                "environment: production",
            ),
            "environment",
        ),
        (
            lambda text: text.replace(
                "permissions:\n  contents: read",
                "permissions:\n  contents: read\n  id-token: write",
            ),
            "workflow scope",
        ),
        (
            lambda text: text.replace(
                "github.event_name == 'push'",
                "github.event_name == 'push' || github.event_name == 'workflow_dispatch'",
                1,
            ),
            "version-tag pushes",
        ),
        (
            lambda text: text.replace(
                "pypa/gh-action-pypi-publish@" + "0" * 40,
                "pypa/gh-action-pypi-publish@release/v1",
                1,
            ),
            "full lowercase commit SHA",
        ),
        (
            lambda text: text.replace(
                "path: publication/",
                "password: ${{ secrets.PYPI_TOKEN }}",
            ),
            "stored publication credentials",
        ),
        (
            lambda text: text.replace(
                "repository-url: https://test.pypi.org/legacy/",
                "repository-url: https://test.pypi.org/legacy/\n"
                "          skip-existing: true",
            ),
            "silently skip",
        ),
        (
            lambda text: text.replace(
                "      - uses: pypa/gh-action-pypi-publish@" + "0" * 40,
                "      - run: echo unsafe\n"
                "      - uses: pypa/gh-action-pypi-publish@" + "0" * 40,
                1,
            ),
            "arbitrary shell commands",
        ),
        (
            lambda text: text.replace(
                "needs: [build, smoke, prepare-publication]",
                "needs: [build, prepare-publication]",
                1,
            ),
            "every required candidate gate",
        ),
        (
            lambda text: text.replace(
                "attestations: true", "attestations: false", 1
            ),
            "explicitly enable attestations",
        ),
        (
            lambda text: text.replace(
                "packages-dir: publication/", "packages-dir: candidate/", 1
            ),
            "staged directory",
        ),
    ],
)
def test_enabled_policy_rejects_unsafe_workflows(unsafe, message: str) -> None:
    policy = replace(load_policy(ROOT), enabled=True)

    with pytest.raises(PublicationPolicyError, match=message):
        validate_workflow_policy(unsafe(_enabled_workflow()), policy)
