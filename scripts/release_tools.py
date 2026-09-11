"""Fail-closed checks and evidence generation for Ginsu release candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tarfile
import zipfile
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Any

PROJECT_NAME = "ginsu"
REQUIRES_PYTHON = ">=3.10,<3.13"
VERSION_PATTERN = re.compile(
    r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)"
    r"(?:(?:a|b|rc)(?:0|[1-9]\d*))?"
)
CHECKSUM_FILE = "SHA256SUMS"
PROHIBITED_SDIST_PREFIXES = (
    ".github/",
    ".git/",
    "AGENTS.md",
    "CLAUDE.md",
    "PROJECT_MEMORY.md",
    "docs/POLARS_VISUALIZATION_PRODUCTION_PLAN.md",
    "docs/PROJECT_BRIEF.md",
    "docs/project_notes/",
)
REQUIRED_SDIST_FILES = {
    "CHANGELOG.md",
    "LICENSE",
    "README.rst",
    "SECURITY.md",
    "pyproject.toml",
    "uv.lock",
    "ginsu/__init__.py",
    "ginsu/py.typed",
    "scripts/release_tools.py",
    "scripts/release_policy.py",
    "scripts/smoke_release.py",
    "tests/test_release_tools.py",
}


class ReleaseCheckError(RuntimeError):
    """A release-candidate invariant was not satisfied."""


def _normalize_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleaseCheckError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_project(root: Path) -> dict[str, Any]:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 CI
        import tomli as tomllib

    with (root / "pyproject.toml").open("rb") as stream:
        document = tomllib.load(stream)
    project = document.get("project")
    _require(isinstance(project, dict), "pyproject.toml has no project table.")
    return project


def project_version(root: Path) -> str:
    """Return the validated release version from pyproject.toml."""
    project = _load_project(root)
    name = project.get("name")
    version = project.get("version")
    _require(name == PROJECT_NAME, f"Project name must be {PROJECT_NAME!r}.")
    _require(isinstance(version, str), "Project version must be a string.")
    _require(
        VERSION_PATTERN.fullmatch(version) is not None,
        f"Unsupported release version {version!r}.",
    )
    return version


def validate_source(
    root: Path,
    *,
    expected_version: str | None = None,
    ref: str | None = None,
) -> str:
    """Validate source metadata and optional tag/changelog consistency."""
    project = _load_project(root)
    version = project_version(root)
    _require(
        project.get("requires-python") == REQUIRES_PYTHON,
        f"requires-python must be {REQUIRES_PYTHON!r}.",
    )
    if expected_version:
        _require(
            version == expected_version,
            f"Expected version {expected_version!r}, found {version!r}.",
        )

    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    _require(
        "## Unreleased" in changelog, "Changelog has no Unreleased section."
    )
    if ref and ref.startswith("refs/tags/"):
        tag = ref.removeprefix("refs/tags/")
        _require(
            tag == f"v{version}",
            f"Tag {tag!r} must equal 'v{version}'.",
        )
        release_heading = re.compile(
            rf"^## \[{re.escape(version)}\] - \d{{4}}-\d{{2}}-\d{{2}}$",
            re.MULTILINE,
        )
        _require(
            release_heading.search(changelog) is not None,
            f"Changelog needs a dated [{version}] release heading.",
        )
    return version


def _safe_archive_name(name: str) -> PurePosixPath:
    _require("\\" not in name, f"Archive path uses a backslash: {name!r}.")
    path = PurePosixPath(name)
    _require(not path.is_absolute(), f"Archive path is absolute: {name!r}.")
    _require(
        ".." not in path.parts,
        f"Archive path traverses its root: {name!r}.",
    )
    return path


def _require_single(paths: list[Path], description: str) -> Path:
    _require(
        len(paths) == 1, f"Expected one {description}, found {len(paths)}."
    )
    return paths[0]


def _requirement_names(requirements: list[str]) -> set[str]:
    names = set()
    for requirement in requirements:
        match = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", requirement)
        _require(match is not None, f"Invalid requirement {requirement!r}.")
        names.add(_normalize_name(match.group(0)))
    return names


def validate_distributions(root: Path, directory: Path) -> tuple[Path, Path]:
    """Inspect one wheel and one sdist against the source package contract."""
    project = _load_project(root)
    version = project_version(root)
    wheel = _require_single(list(directory.glob("*.whl")), "wheel")
    sdist = _require_single(list(directory.glob("*.tar.gz")), "sdist")
    _require(
        wheel.name == f"{PROJECT_NAME}-{version}-py3-none-any.whl",
        f"Unexpected wheel name {wheel.name!r}.",
    )
    _require(
        sdist.name == f"{PROJECT_NAME}-{version}.tar.gz",
        f"Unexpected sdist name {sdist.name!r}.",
    )

    dist_info = f"{PROJECT_NAME}-{version}.dist-info"
    with zipfile.ZipFile(wheel) as archive:
        information = archive.infolist()
        raw_names = [item.filename for item in information]
        _require(
            len(raw_names) == len(set(raw_names)),
            "Wheel contains duplicate paths.",
        )
        for item in information:
            mode = (item.external_attr >> 16) & 0o170000
            _require(
                mode != stat.S_IFLNK,
                f"Wheel contains a symbolic link: {item.filename!r}.",
            )
        names = [_safe_archive_name(name).as_posix() for name in raw_names]
        _require(
            "ginsu/__init__.py" in names, "Wheel omits ginsu/__init__.py."
        )
        _require("ginsu/py.typed" in names, "Wheel omits the py.typed marker.")
        _require(
            all(
                name.startswith("ginsu/") or name.startswith(f"{dist_info}/")
                for name in names
            ),
            "Wheel contains files outside ginsu and its dist-info directory.",
        )
        _require(
            not any(
                "sliceline" in PurePosixPath(name).parts for name in names
            ),
            "Wheel contains a sliceline package.",
        )
        metadata = BytesParser(policy=policy.default).parsebytes(
            archive.read(f"{dist_info}/METADATA")
        )

    _require(metadata["Name"] == PROJECT_NAME, "Wheel metadata name mismatch.")
    _require(
        metadata["Version"] == version, "Wheel metadata version mismatch."
    )
    _require(
        metadata["Requires-Python"] == "<3.13,>=3.10",
        "Wheel Requires-Python does not match the supported matrix.",
    )
    requires_dist = metadata.get_all("Requires-Dist", [])
    core_actual = _requirement_names(
        [item for item in requires_dist if "extra ==" not in item]
    )
    core_expected = _requirement_names(project.get("dependencies", []))
    _require(
        core_actual == core_expected,
        f"Wheel core dependencies differ: {core_actual!r} != {core_expected!r}.",
    )
    extras_actual = set(metadata.get_all("Provides-Extra", []))
    extras_expected = set(project.get("optional-dependencies", {}))
    _require(
        extras_actual == extras_expected,
        f"Wheel extras differ: {extras_actual!r} != {extras_expected!r}.",
    )

    expected_root = f"{PROJECT_NAME}-{version}"
    with tarfile.open(sdist, "r:gz") as archive:
        members = archive.getmembers()
        member_names = [member.name for member in members]
        _require(
            len(member_names) == len(set(member_names)),
            "Sdist contains duplicate paths.",
        )
        for member in members:
            _safe_archive_name(member.name)
            _require(
                member.isfile() or member.isdir(),
                f"Sdist contains a link or special file: {member.name!r}.",
            )
        roots = {PurePosixPath(member.name).parts[0] for member in members}
        _require(
            roots == {expected_root}, f"Unexpected sdist roots: {roots!r}."
        )
        relative_names = {
            PurePosixPath(member.name).relative_to(expected_root).as_posix()
            for member in members
            if PurePosixPath(member.name).as_posix() != expected_root
        }

    missing = REQUIRED_SDIST_FILES - relative_names
    _require(not missing, f"Sdist omits required files: {sorted(missing)!r}.")
    prohibited = sorted(
        name
        for name in relative_names
        if any(
            name == prefix.rstrip("/") or name.startswith(prefix)
            for prefix in PROHIBITED_SDIST_PREFIXES
        )
    )
    _require(
        not prohibited,
        f"Sdist contains repository-internal files: {prohibited!r}.",
    )
    _require(
        not any(
            "__pycache__" in PurePosixPath(name).parts
            for name in relative_names
        ),
        "Sdist contains bytecode cache files.",
    )
    return wheel, sdist


def write_provenance(
    root: Path,
    directory: Path,
    *,
    commit: str,
    ref: str,
    run_id: str,
    run_attempt: str,
) -> Path:
    """Write a deterministic-shape, unsigned candidate provenance record."""
    version = project_version(root)
    wheel, sdist = validate_distributions(root, directory)
    sbom = directory / f"{PROJECT_NAME}-{version}.cdx.json"
    _require(sbom.is_file(), f"Missing SBOM {sbom.name!r}.")
    sbom_document = json.loads(sbom.read_text(encoding="utf-8"))
    _require(
        sbom_document.get("bomFormat") == "CycloneDX"
        and sbom_document.get("specVersion") == "1.5",
        "SBOM must be CycloneDX 1.5.",
    )
    _require(
        commit == "local" or re.fullmatch(r"[0-9a-f]{40}", commit) is not None,
        "Commit must be 'local' or a full lowercase Git SHA.",
    )

    subjects = [
        {"name": path.name, "sha256": _sha256(path)}
        for path in (wheel, sdist, sbom)
    ]
    document = {
        "schema_version": 1,
        "project": PROJECT_NAME,
        "version": version,
        "generated_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "source": {"commit": commit, "ref": ref},
        "builder": {
            "name": "github-actions" if run_id != "local" else "local",
            "workflow": ".github/workflows/release.yml",
            "run_id": run_id,
            "run_attempt": run_attempt,
        },
        "materials": [
            {"name": "uv.lock", "sha256": _sha256(root / "uv.lock")}
        ],
        "subjects": subjects,
        "publication": "disabled",
        "signed_attestation": False,
    }
    destination = directory / f"{PROJECT_NAME}-{version}.provenance.json"
    destination.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def _candidate_files(root: Path, directory: Path) -> tuple[Path, ...]:
    version = project_version(root)
    return tuple(
        directory / name
        for name in (
            f"{PROJECT_NAME}-{version}-py3-none-any.whl",
            f"{PROJECT_NAME}-{version}.tar.gz",
            f"{PROJECT_NAME}-{version}.cdx.json",
            f"{PROJECT_NAME}-{version}.provenance.json",
        )
    )


def write_checksums(root: Path, directory: Path) -> Path:
    """Write sorted SHA-256 checksums for the complete candidate bundle."""
    paths = _candidate_files(root, directory)
    for path in paths:
        _require(path.is_file(), f"Missing candidate file {path.name!r}.")
    destination = directory / CHECKSUM_FILE
    destination.write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in paths),
        encoding="ascii",
    )
    return destination


def verify_checksums(directory: Path) -> None:
    """Verify a downloaded candidate without trusting a source checkout."""
    manifest = directory / CHECKSUM_FILE
    _require(manifest.is_file(), f"Missing {CHECKSUM_FILE}.")
    observed: dict[str, str] = {}
    for line in manifest.read_text(encoding="ascii").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9_.+-]+)", line)
        _require(match is not None, f"Invalid checksum line {line!r}.")
        digest, name = match.groups()
        _require(name not in observed, f"Duplicate checksum entry {name!r}.")
        observed[name] = digest

    wheels = [
        name
        for name in observed
        if re.fullmatch(
            rf"{PROJECT_NAME}-({VERSION_PATTERN.pattern})-py3-none-any\.whl",
            name,
        )
    ]
    _require(len(wheels) == 1, "Checksum manifest must list one Ginsu wheel.")
    wheel_match = re.fullmatch(
        rf"{PROJECT_NAME}-({VERSION_PATTERN.pattern})-py3-none-any\.whl",
        wheels[0],
    )
    _require(wheel_match is not None, "Could not resolve candidate version.")
    version = wheel_match.group(1)
    expected_names = {
        f"{PROJECT_NAME}-{version}-py3-none-any.whl",
        f"{PROJECT_NAME}-{version}.tar.gz",
        f"{PROJECT_NAME}-{version}.cdx.json",
        f"{PROJECT_NAME}-{version}.provenance.json",
    }
    _require(
        set(observed) == expected_names,
        "Checksum manifest does not list the exact candidate bundle.",
    )
    actual_entries = list(directory.iterdir())
    _require(
        all(
            path.is_file() and not path.is_symlink() for path in actual_entries
        ),
        "Candidate directory contains a link, directory, or special file.",
    )
    actual_names = {path.name for path in actual_entries}
    build_ignore = directory / ".gitignore"
    if build_ignore.is_file():
        _require(
            build_ignore.read_bytes() in {b"*", b"*\n"},
            "Candidate build marker has unexpected contents.",
        )
    _require(
        actual_names
        in (
            expected_names | {CHECKSUM_FILE},
            expected_names | {CHECKSUM_FILE, ".gitignore"},
        ),
        "Candidate directory contains unlisted or missing files.",
    )
    for name in sorted(expected_names):
        path = directory / name
        _require(
            _sha256(path) == observed[path.name],
            f"Checksum mismatch for {path.name!r}.",
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("version")
    source = commands.add_parser("validate-source")
    source.add_argument("--expected-version", default="")
    source.add_argument("--ref", default="")

    for name in (
        "validate-dist",
        "write-provenance",
        "write-checksums",
        "verify-checksums",
    ):
        command = commands.add_parser(name)
        command.add_argument("--directory", type=Path, required=True)
        if name == "write-provenance":
            command.add_argument(
                "--commit", default=os.environ.get("GITHUB_SHA", "local")
            )
            command.add_argument(
                "--ref", default=os.environ.get("GITHUB_REF", "local")
            )
            command.add_argument(
                "--run-id", default=os.environ.get("GITHUB_RUN_ID", "local")
            )
            command.add_argument(
                "--run-attempt",
                default=os.environ.get("GITHUB_RUN_ATTEMPT", "local"),
            )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run release checks from the command line."""
    arguments = _parser().parse_args(argv)
    root = arguments.root.resolve()
    try:
        if arguments.command == "version":
            print(project_version(root))
        elif arguments.command == "validate-source":
            print(
                validate_source(
                    root,
                    expected_version=arguments.expected_version or None,
                    ref=arguments.ref or None,
                )
            )
        elif arguments.command == "validate-dist":
            validate_distributions(root, arguments.directory.resolve())
        elif arguments.command == "write-provenance":
            write_provenance(
                root,
                arguments.directory.resolve(),
                commit=arguments.commit,
                ref=arguments.ref,
                run_id=arguments.run_id,
                run_attempt=arguments.run_attempt,
            )
        elif arguments.command == "write-checksums":
            write_checksums(root, arguments.directory.resolve())
        elif arguments.command == "verify-checksums":
            verify_checksums(arguments.directory.resolve())
        else:  # pragma: no cover - argparse enforces the command choices
            raise AssertionError(f"Unhandled command {arguments.command!r}.")
    except (
        OSError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
        ReleaseCheckError,
    ) as error:
        print(f"release check failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
