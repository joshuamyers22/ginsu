"""Round-trip and hostile-input tests for declarative analysis artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl
import pytest

from ginsu import (
    ArtifactError,
    ArtifactLimits,
    CategoryPolicy,
    DiscretizationPlan,
    QuantileBins,
    SliceAnalysis,
    Slicefinder,
)


@pytest.fixture
def analysis_fixture():
    raw = pl.DataFrame(
        {
            "age": [18, 22, 31, 39, 48, 57, 65, 72],
            "region": ["east", "east", "west", "west"] * 2,
        }
    )
    plan = DiscretizationPlan(
        numeric={"age": QuantileBins(4)},
        categorical={"region": CategoryPolicy(max_categories=1)},
    ).fit(raw)
    features = plan.transform(raw)
    finder = Slicefinder(
        alpha=0.95,
        k=10,
        max_l=2,
        min_sup=1,
        verbose=False,
    ).fit(features, [4.0, 3.0, 4.0, 3.0, 1.0, 1.0, 1.0, 1.0])
    analysis = SliceAnalysis.from_finder(
        finder,
        dataset_fingerprint="sha256:discovery-fixture",
        partition_fingerprints={
            "discovery": "sha256:discovery",
            "validation": "sha256:validation",
        },
        discretization=plan,
        dependency_lock_id="sha256:uv-lock-fixture",
    )
    return analysis, raw


def _read_manifest(root: Path) -> dict:
    return json.loads((root / "manifest.json").read_text())


def _write_manifest(root: Path, manifest: dict) -> None:
    (root / "manifest.json").write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_analysis_round_trip_preserves_tables_report_and_plan(
    tmp_path, analysis_fixture
):
    analysis, raw = analysis_fixture
    artifact = analysis.write(tmp_path / "analysis")

    restored = SliceAnalysis.read(artifact)

    assert {path.name for path in artifact.iterdir()} == {
        "manifest.json",
        "slices.arrow",
        "slice_statistics.arrow",
        "predicates.arrow",
    }
    assert restored.slices.equals(analysis.slices)
    assert restored.slice_statistics.equals(analysis.slice_statistics)
    assert restored.predicates.equals(analysis.predicates)
    assert restored.search_report == analysis.search_report
    assert restored.dataset_fingerprint == analysis.dataset_fingerprint
    assert restored.partition_fingerprints == analysis.partition_fingerprints
    assert restored.dependency_lock_id == analysis.dependency_lock_id
    assert restored.discretization is not None
    assert analysis.discretization is not None
    assert (
        restored.discretization.to_spec() == analysis.discretization.to_spec()
    )
    assert restored.discretization.transform(raw).equals(
        analysis.discretization.transform(raw)
    )


def test_artifact_is_deterministic_and_contains_no_raw_rows(
    tmp_path, analysis_fixture
):
    analysis, _ = analysis_fixture
    first = analysis.write(tmp_path / "first")
    second = analysis.write(tmp_path / "second")

    assert (first / "manifest.json").read_bytes() == (
        second / "manifest.json"
    ).read_bytes()
    manifest = _read_manifest(first)
    assert "membership" not in manifest
    assert "raw_data" not in manifest
    for filename in (
        "slices.arrow",
        "slice_statistics.arrow",
        "predicates.arrow",
    ):
        assert (first / filename).read_bytes() == (
            second / filename
        ).read_bytes()


def test_artifact_does_not_overwrite_existing_destination(
    tmp_path, analysis_fixture
):
    analysis, _ = analysis_fixture
    destination = tmp_path / "analysis"
    destination.mkdir()
    marker = destination / "owned-by-caller"
    marker.write_text("keep")

    with pytest.raises(FileExistsError):
        analysis.write(destination)

    assert marker.read_text() == "keep"


def test_unrelated_discretization_plan_cannot_be_attached(analysis_fixture):
    analysis, _ = analysis_fixture
    unrelated = DiscretizationPlan(numeric={"other": QuantileBins(2)}).fit(
        pl.DataFrame({"other": [1, 2, 3]})
    )
    finder_frame = pl.DataFrame({"feature": [0, 1, 0, 1]})
    finder = Slicefinder(min_sup=1, verbose=False).fit(
        finder_frame, [0.0, 1.0, 0.0, 1.0]
    )

    with pytest.raises(ValueError, match="output schema"):
        SliceAnalysis.from_finder(
            finder,
            dataset_fingerprint=analysis.dataset_fingerprint,
            discretization=unrelated,
        )


@pytest.mark.parametrize(
    "field,value,code",
    [
        ("artifact_version", "3.0.0", "GINSU_ARTIFACT_VERSION"),
        ("artifact_format", "other", "GINSU_ARTIFACT_FORMAT"),
        (
            "canonicalization_version",
            2,
            "GINSU_ARTIFACT_CANONICALIZATION",
        ),
    ],
)
def test_unknown_manifest_identity_fails_closed(
    tmp_path, analysis_fixture, field, value, code
):
    analysis, _ = analysis_fixture
    artifact = analysis.write(tmp_path / "analysis")
    manifest = _read_manifest(artifact)
    manifest[field] = value
    _write_manifest(artifact, manifest)

    with pytest.raises(ArtifactError) as raised:
        SliceAnalysis.read(artifact)

    assert raised.value.code == code


def test_version_one_artifact_migrates_without_timing_evidence(
    tmp_path, analysis_fixture
):
    analysis, _ = analysis_fixture
    artifact = analysis.write(tmp_path / "analysis-v1")
    manifest = _read_manifest(artifact)
    manifest["artifact_version"] = "1.0.0"
    del manifest["search_report"]["stages"]
    del manifest["search_report"]["memory_measurement"]
    del manifest["search_report"]["observed_peak_memory_bytes"]
    _write_manifest(artifact, manifest)

    restored = SliceAnalysis.read(artifact)

    assert restored.search_report.stages == ()
    assert restored.search_report.memory_measurement is None
    assert restored.search_report.observed_peak_memory_bytes is None


def test_version_two_round_trip_preserves_memory_diagnostics(tmp_path):
    ticks = iter(float(value) for value in range(6))
    memory = iter((100, 110, 120, 130, 140, 150))
    frame = pl.DataFrame({"segment": ["a", "a", "b", "b"]})
    finder = Slicefinder(
        min_sup=1,
        verbose=False,
        clock=lambda: next(ticks),
        memory_sampler=lambda: next(memory),
        memory_measurement="test_boundary_bytes",
    ).fit(frame, [2.0, 2.0, 1.0, 1.0])
    analysis = SliceAnalysis.from_finder(
        finder, dataset_fingerprint="sha256:memory-diagnostics"
    )

    artifact = analysis.write(tmp_path / "analysis-v2")
    restored = SliceAnalysis.read(artifact)

    assert _read_manifest(artifact)["artifact_version"] == "2.0.0"
    assert restored.search_report == analysis.search_report
    assert restored.search_report.memory_measurement == "test_boundary_bytes"
    assert restored.search_report.observed_peak_memory_bytes == 150


def test_artifact_report_version_schemas_remain_closed(
    tmp_path, analysis_fixture
):
    analysis, _ = analysis_fixture
    missing_v2 = analysis.write(tmp_path / "missing-v2")
    manifest = _read_manifest(missing_v2)
    del manifest["search_report"]["stages"]
    _write_manifest(missing_v2, manifest)
    with pytest.raises(ArtifactError) as raised:
        SliceAnalysis.read(missing_v2)
    assert raised.value.code == "GINSU_ARTIFACT_SCHEMA"

    extra_v1 = analysis.write(tmp_path / "extra-v1")
    manifest = _read_manifest(extra_v1)
    manifest["artifact_version"] = "1.0.0"
    del manifest["search_report"]["memory_measurement"]
    del manifest["search_report"]["observed_peak_memory_bytes"]
    _write_manifest(extra_v1, manifest)
    with pytest.raises(ArtifactError) as raised:
        SliceAnalysis.read(extra_v1)
    assert raised.value.code == "GINSU_ARTIFACT_SCHEMA"


def test_version_two_rejects_inconsistent_memory_evidence(
    tmp_path, analysis_fixture
):
    analysis, _ = analysis_fixture
    artifact = analysis.write(tmp_path / "invalid-memory")
    manifest = _read_manifest(artifact)
    manifest["search_report"]["memory_measurement"] = "test_bytes"
    manifest["search_report"]["observed_peak_memory_bytes"] = 1
    _write_manifest(artifact, manifest)

    with pytest.raises(ArtifactError) as raised:
        SliceAnalysis.read(artifact)

    assert raised.value.code == "GINSU_ARTIFACT_SCHEMA"


def test_hash_mismatch_and_unexpected_file_fail_closed(
    tmp_path, analysis_fixture
):
    analysis, _ = analysis_fixture
    hash_artifact = analysis.write(tmp_path / "hash")
    slices_path = hash_artifact / "slices.arrow"
    payload = bytearray(slices_path.read_bytes())
    payload[len(payload) // 2] ^= 1
    slices_path.write_bytes(payload)

    with pytest.raises(ArtifactError) as hash_error:
        SliceAnalysis.read(hash_artifact)
    assert hash_error.value.code == "GINSU_ARTIFACT_HASH"

    file_artifact = analysis.write(tmp_path / "file")
    (file_artifact / "unexpected.txt").write_text("unexpected")
    with pytest.raises(ArtifactError) as file_error:
        SliceAnalysis.read(file_artifact)
    assert file_error.value.code == "GINSU_ARTIFACT_FILES"


def test_manifest_path_traversal_is_rejected(tmp_path, analysis_fixture):
    analysis, _ = analysis_fixture
    artifact = analysis.write(tmp_path / "analysis")
    manifest = _read_manifest(artifact)
    manifest["tables"]["slices"]["path"] = "../slices.arrow"
    _write_manifest(artifact, manifest)

    with pytest.raises(ArtifactError) as raised:
        SliceAnalysis.read(artifact)

    assert raised.value.code == "GINSU_ARTIFACT_TABLE_PATH"


def test_duplicate_json_keys_are_rejected(tmp_path, analysis_fixture):
    analysis, _ = analysis_fixture
    artifact = analysis.write(tmp_path / "analysis")
    (artifact / "manifest.json").write_text(
        '{"artifact_format":"ginsu.slice-analysis","artifact_format":"other"}'
    )

    with pytest.raises(ArtifactError) as raised:
        SliceAnalysis.read(artifact)

    assert raised.value.code == "GINSU_ARTIFACT_DUPLICATE_KEY"


def test_read_limits_apply_before_table_materialization(
    tmp_path, analysis_fixture
):
    analysis, _ = analysis_fixture
    artifact = analysis.write(tmp_path / "analysis")

    with pytest.raises(ArtifactError) as metadata_error:
        SliceAnalysis.read(
            artifact,
            limits=ArtifactLimits(max_metadata_bytes=1),
        )
    assert metadata_error.value.code == "GINSU_ARTIFACT_METADATA_BYTES"

    with pytest.raises(ArtifactError) as table_error:
        SliceAnalysis.read(
            artifact,
            limits=ArtifactLimits(max_table_bytes=1),
        )
    assert table_error.value.code == "GINSU_ARTIFACT_TABLE_BYTES"

    with pytest.raises(ArtifactError) as row_error:
        SliceAnalysis.read(
            artifact,
            limits=ArtifactLimits(max_rows_per_table=1),
        )
    assert row_error.value.code == "GINSU_ARTIFACT_TABLE_ROWS"

    with pytest.raises(ArtifactError) as materialized_error:
        SliceAnalysis.read(
            artifact,
            limits=ArtifactLimits(max_materialized_table_bytes=1),
        )
    assert materialized_error.value.code == "GINSU_ARTIFACT_MATERIALIZED_BYTES"


def test_valid_hash_does_not_hide_broken_table_relationships(
    tmp_path, analysis_fixture
):
    analysis, _ = analysis_fixture
    artifact = analysis.write(tmp_path / "analysis")
    slices_path = artifact / "slices.arrow"
    slices = pl.read_ipc(slices_path, memory_map=False).with_columns(
        pl.when(pl.col("__ginsu_rank") == 1)
        .then(pl.lit("ginsu:v1:invalid"))
        .otherwise(pl.col("__ginsu_id"))
        .alias("__ginsu_id")
    )
    slices.write_ipc(slices_path, compression="uncompressed")
    manifest = _read_manifest(artifact)
    manifest["tables"]["slices"]["bytes"] = slices_path.stat().st_size
    manifest["tables"]["slices"]["sha256"] = _sha256(slices_path)
    _write_manifest(artifact, manifest)

    with pytest.raises(ArtifactError) as raised:
        SliceAnalysis.read(artifact)

    assert raised.value.code == "GINSU_ARTIFACT_RELATIONSHIP"


def test_empty_analysis_round_trip(tmp_path):
    frame = pl.DataFrame({"segment": ["a", "b"]})
    finder = Slicefinder(min_sup=3, verbose=False).fit(frame, [1.0, 1.0])
    analysis = SliceAnalysis.from_finder(
        finder, dataset_fingerprint="sha256:empty"
    )

    restored = SliceAnalysis.read(analysis.write(tmp_path / "empty"))

    assert restored.slices.is_empty()
    assert restored.slice_statistics.is_empty()
    assert restored.predicates.is_empty()
    assert restored.search_report.status == "no_valid_slices"
