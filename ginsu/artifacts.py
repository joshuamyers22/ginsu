"""Safe, versioned, non-executable Ginsu analysis artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, cast

import polars as pl
from sklearn.utils.validation import check_is_fitted

from ginsu._domain import (
    Predicate,
    Slice,
    canonical_json,
    value_from_canonical_json,
)
from ginsu._frame import schema_signature
from ginsu.diagnostics import SearchLevelReport, SearchLimits, SearchReport
from ginsu.discretization import DiscretizationPlan

ARTIFACT_FORMAT = "ginsu.slice-analysis"
ARTIFACT_VERSION = "1.0.0"
CANONICALIZATION_VERSION = 1
ALGORITHM_NAME = "sliceline"
ALGORITHM_VERSION = "1"

_TABLE_PATHS = {
    "slices": "slices.arrow",
    "slice_statistics": "slice_statistics.arrow",
    "predicates": "predicates.arrow",
}
_MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True, slots=True)
class ArtifactLimits:
    """Pre-materialization limits for reading a SliceAnalysis artifact."""

    max_metadata_bytes: int = 1024 * 1024
    max_table_bytes: int = 64 * 1024 * 1024
    max_total_bytes: int = 256 * 1024 * 1024
    max_rows_per_table: int = 1_000_000
    max_materialized_table_bytes: int = 128 * 1024 * 1024

    def __post_init__(self) -> None:
        for name in (
            "max_metadata_bytes",
            "max_table_bytes",
            "max_total_bytes",
            "max_rows_per_table",
            "max_materialized_table_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")


class ArtifactError(ValueError):
    """Raised when an analysis artifact fails closed validation."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True, slots=True)
class SliceAnalysis:
    """Declarative fitted-analysis state with no raw observations or code."""

    slices: pl.DataFrame
    slice_statistics: pl.DataFrame
    predicates: pl.DataFrame
    feature_schema: tuple[tuple[str, str], ...]
    search_parameters: tuple[tuple[str, int | float], ...]
    search_report: SearchReport
    dataset_fingerprint: str
    partition_fingerprints: tuple[tuple[str, str], ...] = ()
    discretization: DiscretizationPlan | None = None
    dependency_lock_id: str | None = None
    package_version: str = "0+unknown"

    @classmethod
    def from_finder(
        cls,
        finder: Any,
        *,
        dataset_fingerprint: str,
        partition_fingerprints: Mapping[str, str] | None = None,
        discretization: DiscretizationPlan | None = None,
        dependency_lock_id: str | None = None,
    ) -> SliceAnalysis:
        """Capture canonical fitted results and caller-supplied provenance."""
        check_is_fitted(
            finder,
            (
                "slices_",
                "slice_statistics_",
                "predicates_",
                "search_report_",
                "_feature_schema",
            ),
        )
        if not finder.search_report_.exhaustive:
            raise ValueError("Only exhaustive fitted results can be exported.")
        normalized_partitions = _fingerprint_pairs(
            partition_fingerprints or {}, "partition_fingerprints"
        )
        _required_string(dataset_fingerprint, "dataset_fingerprint")
        if dependency_lock_id is not None:
            _required_string(dependency_lock_id, "dependency_lock_id")
        restored_discretization = (
            DiscretizationPlan.from_spec(discretization.to_spec())
            if discretization is not None
            else None
        )
        try:
            package_version = metadata.version("ginsu")
        except metadata.PackageNotFoundError:
            package_version = "0+unknown"

        result = cls(
            slices=finder.slices_.clone(),
            slice_statistics=finder.slice_statistics_.clone(),
            predicates=finder.predicates_.clone(),
            feature_schema=tuple(finder._feature_schema),
            search_parameters=(
                ("alpha", float(finder.alpha)),
                ("k", int(finder.k)),
                ("max_l", int(finder.max_l)),
                ("min_sup", finder.min_sup),
            ),
            search_report=finder.search_report_,
            dataset_fingerprint=dataset_fingerprint,
            partition_fingerprints=normalized_partitions,
            discretization=restored_discretization,
            dependency_lock_id=dependency_lock_id,
            package_version=package_version,
        )
        _validate_analysis(result)
        return result

    def write(self, path: str | os.PathLike[str]) -> Path:
        """Write atomically without replacing an existing destination."""
        _validate_analysis(self)
        destination = Path(path)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"Artifact destination already exists: {destination}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.tmp-", dir=destination.parent
            )
        )
        try:
            tables = {
                "slices": self.slices,
                "slice_statistics": self.slice_statistics,
                "predicates": self.predicates,
            }
            table_manifest: dict[str, dict[str, Any]] = {}
            for name, frame in tables.items():
                relative_path = _TABLE_PATHS[name]
                table_path = temporary / relative_path
                frame.write_ipc(table_path, compression="uncompressed")
                table_manifest[name] = {
                    "path": relative_path,
                    "sha256": _sha256(table_path),
                    "bytes": table_path.stat().st_size,
                    "rows": frame.height,
                    "schema": _schema_data(frame),
                }

            manifest = self._manifest(table_manifest)
            manifest_bytes = _canonical_json_bytes(manifest)
            (temporary / _MANIFEST_NAME).write_bytes(manifest_bytes)
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(
                    f"Artifact destination already exists: {destination}"
                )
            temporary.rename(destination)
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
        return destination

    @classmethod
    def read(
        cls,
        path: str | os.PathLike[str],
        *,
        limits: ArtifactLimits | None = None,
    ) -> SliceAnalysis:
        """Load a closed-schema artifact after bounded integrity checks."""
        active_limits = limits or ArtifactLimits()
        root = Path(path)
        if root.is_symlink() or not root.is_dir():
            raise ArtifactError(
                "GINSU_ARTIFACT_PATH", "artifact path must be a real directory"
            )
        entries = list(root.iterdir())
        if any(entry.is_symlink() or not entry.is_file() for entry in entries):
            raise ArtifactError(
                "GINSU_ARTIFACT_FILE_TYPE",
                "artifact entries must be regular files, not links or directories",
            )
        actual_names = {entry.name for entry in entries}
        expected_names = {_MANIFEST_NAME, *_TABLE_PATHS.values()}
        if actual_names != expected_names:
            raise ArtifactError(
                "GINSU_ARTIFACT_FILES",
                f"expected files {sorted(expected_names)!r}; received {sorted(actual_names)!r}",
            )

        manifest_path = root / _MANIFEST_NAME
        metadata_bytes = manifest_path.stat().st_size
        if metadata_bytes > active_limits.max_metadata_bytes:
            raise ArtifactError(
                "GINSU_ARTIFACT_METADATA_BYTES",
                "manifest exceeds max_metadata_bytes",
            )
        total_bytes = sum(entry.stat().st_size for entry in entries)
        if total_bytes > active_limits.max_total_bytes:
            raise ArtifactError(
                "GINSU_ARTIFACT_TOTAL_BYTES",
                "artifact exceeds max_total_bytes",
            )
        manifest = _load_json(manifest_path.read_bytes())
        try:
            parsed = _parse_manifest(manifest)
        except ArtifactError:
            raise
        except (TypeError, ValueError) as error:
            raise ArtifactError(
                "GINSU_ARTIFACT_SCHEMA", "manifest values are invalid"
            ) from error

        loaded: dict[str, pl.DataFrame] = {}
        for name, expected_path in _TABLE_PATHS.items():
            table_data = parsed["tables"][name]
            if table_data["path"] != expected_path:
                raise ArtifactError(
                    "GINSU_ARTIFACT_TABLE_PATH",
                    f"unexpected path for table {name!r}",
                )
            table_path = root / expected_path
            file_bytes = table_path.stat().st_size
            if file_bytes != table_data["bytes"]:
                raise ArtifactError(
                    "GINSU_ARTIFACT_TABLE_BYTES",
                    f"byte count mismatch for table {name!r}",
                )
            if file_bytes > active_limits.max_table_bytes:
                raise ArtifactError(
                    "GINSU_ARTIFACT_TABLE_BYTES",
                    f"table {name!r} exceeds max_table_bytes",
                )
            if table_data["rows"] > active_limits.max_rows_per_table:
                raise ArtifactError(
                    "GINSU_ARTIFACT_TABLE_ROWS",
                    f"table {name!r} exceeds max_rows_per_table",
                )
            if _sha256(table_path) != table_data["sha256"]:
                raise ArtifactError(
                    "GINSU_ARTIFACT_HASH", f"hash mismatch for table {name!r}"
                )
            try:
                frame = pl.read_ipc(table_path, memory_map=False)
            except Exception as error:
                raise ArtifactError(
                    "GINSU_ARTIFACT_TABLE_READ",
                    f"could not read table {name!r}",
                ) from error
            if frame.height != table_data["rows"]:
                raise ArtifactError(
                    "GINSU_ARTIFACT_TABLE_ROWS",
                    f"row count mismatch for table {name!r}",
                )
            if (
                frame.estimated_size()
                > active_limits.max_materialized_table_bytes
            ):
                raise ArtifactError(
                    "GINSU_ARTIFACT_MATERIALIZED_BYTES",
                    f"table {name!r} exceeds max_materialized_table_bytes",
                )
            if _schema_data(frame) != table_data["schema"]:
                raise ArtifactError(
                    "GINSU_ARTIFACT_TABLE_SCHEMA",
                    f"schema mismatch for table {name!r}",
                )
            loaded[name] = frame

        result = cls(
            slices=loaded["slices"],
            slice_statistics=loaded["slice_statistics"],
            predicates=loaded["predicates"],
            feature_schema=parsed["feature_schema"],
            search_parameters=parsed["search_parameters"],
            search_report=parsed["search_report"],
            dataset_fingerprint=parsed["dataset_fingerprint"],
            partition_fingerprints=parsed["partition_fingerprints"],
            discretization=parsed["discretization"],
            dependency_lock_id=parsed["dependency_lock_id"],
            package_version=parsed["package_version"],
        )
        try:
            _validate_analysis(result)
        except ArtifactError:
            raise
        except (TypeError, ValueError) as error:
            raise ArtifactError(
                "GINSU_ARTIFACT_RELATIONSHIP",
                "artifact tables or metadata are inconsistent",
            ) from error
        return result

    def _manifest(self, tables: dict[str, dict[str, Any]]) -> dict[str, Any]:
        return {
            "artifact_format": ARTIFACT_FORMAT,
            "artifact_version": ARTIFACT_VERSION,
            "canonicalization_version": CANONICALIZATION_VERSION,
            "package_version": self.package_version,
            "algorithm": ALGORITHM_NAME,
            "algorithm_version": ALGORITHM_VERSION,
            "feature_schema": [
                {"name": name, "dtype": dtype}
                for name, dtype in self.feature_schema
            ],
            "search_parameters": dict(self.search_parameters),
            "search_report": asdict(self.search_report),
            "dataset_fingerprint": self.dataset_fingerprint,
            "partition_fingerprints": dict(self.partition_fingerprints),
            "dependency_lock_id": self.dependency_lock_id,
            "discretization_plan": (
                self.discretization.to_spec()
                if self.discretization is not None
                else None
            ),
            "tables": tables,
        }


def _validate_analysis(analysis: SliceAnalysis) -> None:
    _required_string(analysis.dataset_fingerprint, "dataset_fingerprint")
    _required_string(analysis.package_version, "package_version")
    _fingerprint_pairs(
        dict(analysis.partition_fingerprints), "partition_fingerprints"
    )
    if analysis.dependency_lock_id is not None:
        _required_string(analysis.dependency_lock_id, "dependency_lock_id")
    if not analysis.search_report.exhaustive:
        raise ValueError(
            "Artifact search report must describe an exhaustive result."
        )
    if analysis.search_report.input_schema != analysis.feature_schema:
        raise ValueError(
            "Artifact feature schema differs from the search report."
        )
    if analysis.search_report.row_count <= 0:
        raise ValueError("Artifact search report must contain input rows.")

    feature_names = [name for name, _dtype in analysis.feature_schema]
    if not feature_names or len(set(feature_names)) != len(feature_names):
        raise ValueError(
            "Artifact feature names must be non-empty and unique."
        )
    if analysis.search_report.feature_count != len(feature_names):
        raise ValueError("Artifact feature count differs from its schema.")
    if [
        name
        for name, _cardinality in analysis.search_report.feature_cardinalities
    ] != feature_names or any(
        cardinality <= 0
        for _name, cardinality in analysis.search_report.feature_cardinalities
    ):
        raise ValueError("Artifact feature cardinalities are invalid.")
    if (
        analysis.search_report.backend == "numba"
    ) != analysis.search_report.numba_used:
        raise ValueError("Artifact backend and Numba-use flag disagree.")
    if analysis.search_report.termination_reason is not None:
        raise ValueError(
            "An exhaustive artifact cannot have a termination reason."
        )
    expected_slice_columns = [
        "__ginsu_id",
        "__ginsu_rank",
        "__ginsu_rule",
        *feature_names,
    ]
    if analysis.slices.columns != expected_slice_columns:
        raise ValueError("Artifact slices table has unexpected columns.")
    if list(analysis.slices.schema.values())[:3] != [
        pl.String,
        pl.UInt32,
        pl.String,
    ]:
        raise ValueError("Artifact slices table has invalid identity dtypes.")
    if schema_signature(analysis.slices)[3:] != analysis.feature_schema:
        raise ValueError(
            "Artifact slice feature schema does not match metadata."
        )
    expected_statistics = [
        "__ginsu_id",
        "rank",
        "slice_score",
        "support_count",
        "support_fraction",
        "error_sum",
        "error_max",
        "error_mean",
        "baseline_error_mean",
        "error_lift",
        "excess_error",
        "predicate_count",
    ]
    expected_predicates = [
        "__ginsu_id",
        "rank",
        "predicate_position",
        "feature",
        "operator",
        "value_json",
        "display_value",
        "source_dtype",
    ]
    if analysis.slice_statistics.columns != expected_statistics:
        raise ValueError("Artifact statistics table has unexpected columns.")
    if analysis.predicates.columns != expected_predicates:
        raise ValueError("Artifact predicates table has unexpected columns.")
    if analysis.slice_statistics.schema != {
        "__ginsu_id": pl.String,
        "rank": pl.UInt32,
        "slice_score": pl.Float64,
        "support_count": pl.UInt64,
        "support_fraction": pl.Float64,
        "error_sum": pl.Float64,
        "error_max": pl.Float64,
        "error_mean": pl.Float64,
        "baseline_error_mean": pl.Float64,
        "error_lift": pl.Float64,
        "excess_error": pl.Float64,
        "predicate_count": pl.UInt32,
    }:
        raise ValueError("Artifact statistics table has invalid dtypes.")
    if analysis.predicates.schema != {
        "__ginsu_id": pl.String,
        "rank": pl.UInt32,
        "predicate_position": pl.UInt32,
        "feature": pl.String,
        "operator": pl.String,
        "value_json": pl.String,
        "display_value": pl.String,
        "source_dtype": pl.String,
    }:
        raise ValueError("Artifact predicates table has invalid dtypes.")

    identifiers = analysis.slices.get_column("__ginsu_id").to_list()
    ranks = analysis.slices.get_column("__ginsu_rank").to_list()
    if len(set(identifiers)) != len(identifiers) or ranks != list(
        range(1, len(identifiers) + 1)
    ):
        raise ValueError("Artifact slice identifiers or ranks are invalid.")
    if bool(identifiers) != (analysis.search_report.status == "complete"):
        raise ValueError("Artifact search status does not match its results.")
    if (
        analysis.slice_statistics.get_column("__ginsu_id").to_list()
        != identifiers
    ):
        raise ValueError(
            "Artifact statistics identifiers do not match slices."
        )
    if analysis.slice_statistics.get_column("rank").to_list() != ranks:
        raise ValueError("Artifact statistics ranks do not match slices.")
    if not set(
        analysis.predicates.get_column("__ginsu_id").to_list()
    ).issubset(identifiers):
        raise ValueError("Artifact predicates reference unknown slices.")

    numeric_columns = [
        "slice_score",
        "support_fraction",
        "error_sum",
        "error_max",
        "error_mean",
        "baseline_error_mean",
        "error_lift",
        "excess_error",
    ]
    for column in numeric_columns:
        values = analysis.slice_statistics.get_column(column)
        if values.null_count() or not values.is_finite().all():
            raise ValueError(f"Artifact statistic {column!r} must be finite.")
    if analysis.slice_statistics.height:
        if (
            not analysis.slice_statistics.get_column("support_fraction")
            .is_between(0, 1)
            .all()
        ):
            raise ValueError("Artifact support fractions must be in [0, 1].")
        support_min = cast(
            int,
            analysis.slice_statistics.get_column("support_count").min(),
        )
        support_max = cast(
            int,
            analysis.slice_statistics.get_column("support_count").max(),
        )
        if support_min <= 0 or support_max > analysis.search_report.row_count:
            raise ValueError(
                "Artifact support counts are outside discovery bounds."
            )
        baseline_min = cast(
            float,
            analysis.slice_statistics.get_column("baseline_error_mean").min(),
        )
        if baseline_min <= 0:
            raise ValueError("Artifact baseline error means must be positive.")

    predicates_by_id: dict[str, list[dict[str, Any]]] = {
        identifier: [] for identifier in identifiers
    }
    for row in analysis.predicates.iter_rows(named=True):
        predicates_by_id[row["__ginsu_id"]].append(row)
    slice_rows = {
        row["__ginsu_id"]: row for row in analysis.slices.iter_rows(named=True)
    }
    counts = analysis.slice_statistics.get_column("predicate_count").to_list()
    dtype_lookup = dict(analysis.feature_schema)
    for identifier, rank, expected_count in zip(
        identifiers, ranks, counts, strict=True
    ):
        rows = sorted(
            predicates_by_id[identifier],
            key=lambda item: item["predicate_position"],
        )
        if [row["predicate_position"] for row in rows] != list(
            range(len(rows))
        ) or len(rows) != expected_count:
            raise ValueError(
                "Artifact predicate positions or counts are invalid."
            )
        if any(row["rank"] != rank for row in rows):
            raise ValueError("Artifact predicate ranks do not match slices.")
        predicates = tuple(
            Predicate(
                feature=row["feature"],
                operator=row["operator"],
                value=value_from_canonical_json(row["value_json"]),
            )
            for row in rows
        )
        item = Slice(predicates)
        slice_row = slice_rows[identifier]
        if item.id != identifier or item.rule != slice_row["__ginsu_rule"]:
            raise ValueError("Artifact stable slice identity is invalid.")
        for row, predicate in zip(rows, predicates, strict=True):
            if row["feature"] not in dtype_lookup:
                raise ValueError(
                    "Artifact predicate references an unknown feature."
                )
            if row["source_dtype"] != dtype_lookup[row["feature"]]:
                raise ValueError(
                    "Artifact predicate dtype does not match schema."
                )
            if row["display_value"] != predicate.display_value:
                raise ValueError(
                    "Artifact predicate display value is invalid."
                )
            if canonical_json(slice_row[row["feature"]]) != row["value_json"]:
                raise ValueError(
                    "Artifact slice cells do not match predicates."
                )

    if analysis.discretization is not None:
        restored = DiscretizationPlan.from_spec(
            analysis.discretization.to_spec()
        )
        if restored.input_schema_ != analysis.discretization.input_schema_:
            raise ValueError("Artifact discretization schema is invalid.")
        output_schema = tuple(
            (
                name,
                "UInt32"
                if name in restored.numeric
                else "String"
                if name in restored.categorical
                else dtype,
            )
            for name, dtype in restored.input_schema_
        )
        if output_schema != analysis.feature_schema:
            raise ValueError(
                "Artifact discretization output schema differs from search input."
            )


def _parse_manifest(value: Any) -> dict[str, Any]:
    root = _closed_mapping(
        value,
        {
            "artifact_format",
            "artifact_version",
            "canonicalization_version",
            "package_version",
            "algorithm",
            "algorithm_version",
            "feature_schema",
            "search_parameters",
            "search_report",
            "dataset_fingerprint",
            "partition_fingerprints",
            "dependency_lock_id",
            "discretization_plan",
            "tables",
        },
        "manifest",
    )
    if root["artifact_format"] != ARTIFACT_FORMAT:
        raise ArtifactError("GINSU_ARTIFACT_FORMAT", "unknown artifact format")
    if root["artifact_version"] != ARTIFACT_VERSION:
        raise ArtifactError(
            "GINSU_ARTIFACT_VERSION", "unsupported artifact version"
        )
    if root["canonicalization_version"] != CANONICALIZATION_VERSION:
        raise ArtifactError(
            "GINSU_ARTIFACT_CANONICALIZATION",
            "unsupported canonicalization version",
        )
    if (
        root["algorithm"] != ALGORITHM_NAME
        or root["algorithm_version"] != ALGORITHM_VERSION
    ):
        raise ArtifactError(
            "GINSU_ARTIFACT_ALGORITHM", "unsupported algorithm"
        )

    feature_schema = _parse_feature_schema(root["feature_schema"])
    search_parameters = _parse_search_parameters(root["search_parameters"])
    report = _parse_search_report(root["search_report"])
    dataset_fingerprint = _required_string(
        root["dataset_fingerprint"], "dataset_fingerprint"
    )
    partitions_object = _plain_dict(
        root["partition_fingerprints"], "partition_fingerprints"
    )
    partitions = _fingerprint_pairs(
        partitions_object, "partition_fingerprints"
    )
    dependency_lock_id = root["dependency_lock_id"]
    if dependency_lock_id is not None:
        dependency_lock_id = _required_string(
            dependency_lock_id, "dependency_lock_id"
        )
    package_version = _required_string(
        root["package_version"], "package_version"
    )
    discretization = (
        None
        if root["discretization_plan"] is None
        else DiscretizationPlan.from_spec(root["discretization_plan"])
    )

    tables_object = _closed_mapping(
        root["tables"], set(_TABLE_PATHS), "tables"
    )
    tables = {
        name: _parse_table_manifest(tables_object[name], name)
        for name in _TABLE_PATHS
    }
    return {
        "feature_schema": feature_schema,
        "search_parameters": search_parameters,
        "search_report": report,
        "dataset_fingerprint": dataset_fingerprint,
        "partition_fingerprints": partitions,
        "dependency_lock_id": dependency_lock_id,
        "package_version": package_version,
        "discretization": discretization,
        "tables": tables,
    }


def _parse_search_report(value: Any) -> SearchReport:
    root = _closed_mapping(
        value,
        {
            "status",
            "backend",
            "numba_used",
            "input_kind",
            "input_schema",
            "row_count",
            "feature_count",
            "feature_cardinalities",
            "encoded_feature_count",
            "copy_boundaries",
            "levels",
            "elapsed_seconds",
            "limits",
            "warning_codes",
            "termination_reason",
        },
        "search_report",
    )
    status = _required_string(root["status"], "search_report.status")
    if status not in (
        "complete",
        "no_valid_slices",
        "limit_reached",
        "cancelled",
        "failed",
    ):
        raise ArtifactError("GINSU_ARTIFACT_REPORT", "invalid search status")
    backend = _required_string(root["backend"], "search_report.backend")
    if backend not in ("numba", "numpy"):
        raise ArtifactError("GINSU_ARTIFACT_REPORT", "invalid search backend")
    if type(root["numba_used"]) is not bool:
        raise ArtifactError(
            "GINSU_ARTIFACT_REPORT", "numba_used must be Boolean"
        )
    limits_data = _closed_mapping(
        root["limits"],
        set(SearchLimits.__dataclass_fields__),
        "search_report.limits",
    )
    active_limits = SearchLimits(
        max_feature_cardinality=_optional_int(
            limits_data["max_feature_cardinality"],
            "limits.max_feature_cardinality",
        ),
        max_encoded_features=_optional_int(
            limits_data["max_encoded_features"],
            "limits.max_encoded_features",
        ),
        max_pair_matrix_bytes=_optional_int(
            limits_data["max_pair_matrix_bytes"],
            "limits.max_pair_matrix_bytes",
        ),
        max_candidates_per_level=_optional_int(
            limits_data["max_candidates_per_level"],
            "limits.max_candidates_per_level",
        ),
        max_total_candidates=_optional_int(
            limits_data["max_total_candidates"],
            "limits.max_total_candidates",
        ),
        max_search_seconds=_optional_float(
            limits_data["max_search_seconds"],
            "limits.max_search_seconds",
        ),
        max_tied_slices=_optional_int(
            limits_data["max_tied_slices"], "limits.max_tied_slices"
        ),
    )
    levels = []
    for index, item in enumerate(
        _plain_list(root["levels"], "search_report.levels")
    ):
        level_data = _closed_mapping(
            item,
            set(SearchLevelReport.__dataclass_fields__),
            f"levels[{index}]",
        )
        levels.append(
            SearchLevelReport(
                **{
                    name: _nonnegative_int(
                        level_data[name], f"levels[{index}].{name}"
                    )
                    for name in level_data
                }
            )
        )
    encoded = root["encoded_feature_count"]
    encoded_count = (
        None
        if encoded is None
        else _nonnegative_int(encoded, "encoded_feature_count")
    )
    elapsed = _finite_number(root["elapsed_seconds"], "elapsed_seconds")
    if elapsed < 0:
        raise ArtifactError(
            "GINSU_ARTIFACT_REPORT", "elapsed_seconds is negative"
        )
    termination = root["termination_reason"]
    if termination is not None and type(termination) is not str:
        raise ArtifactError(
            "GINSU_ARTIFACT_REPORT", "termination_reason is invalid"
        )
    return SearchReport(
        status=status,  # type: ignore[arg-type]
        backend=backend,  # type: ignore[arg-type]
        numba_used=root["numba_used"],
        input_kind=_required_string(root["input_kind"], "input_kind"),
        input_schema=_parse_schema_pairs(
            root["input_schema"], "search_report.input_schema"
        ),
        row_count=_nonnegative_int(root["row_count"], "row_count"),
        feature_count=_nonnegative_int(root["feature_count"], "feature_count"),
        feature_cardinalities=_parse_cardinalities(
            root["feature_cardinalities"]
        ),
        encoded_feature_count=encoded_count,
        copy_boundaries=_string_tuple(
            root["copy_boundaries"], "copy_boundaries"
        ),
        levels=tuple(levels),
        elapsed_seconds=elapsed,
        limits=active_limits,
        warning_codes=_string_tuple(root["warning_codes"], "warning_codes"),
        termination_reason=termination,
    )


def _parse_table_manifest(value: Any, name: str) -> dict[str, Any]:
    root = _closed_mapping(
        value, {"path", "sha256", "bytes", "rows", "schema"}, name
    )
    digest = _required_string(root["sha256"], f"{name}.sha256")
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ArtifactError(
            "GINSU_ARTIFACT_HASH", f"invalid hash for table {name!r}"
        )
    return {
        "path": _required_string(root["path"], f"{name}.path"),
        "sha256": digest,
        "bytes": _nonnegative_int(root["bytes"], f"{name}.bytes"),
        "rows": _nonnegative_int(root["rows"], f"{name}.rows"),
        "schema": _parse_schema_data(root["schema"], f"{name}.schema"),
    }


def _parse_feature_schema(value: Any) -> tuple[tuple[str, str], ...]:
    rows = _plain_list(value, "feature_schema")
    result = tuple(
        (
            _required_string(
                _closed_mapping(row, {"name", "dtype"}, "feature_schema row")[
                    "name"
                ],
                "feature name",
            ),
            _required_string(row["dtype"], "feature dtype"),
        )
        for row in rows
    )
    if not result or len({name for name, _dtype in result}) != len(result):
        raise ArtifactError(
            "GINSU_ARTIFACT_SCHEMA", "feature schema is invalid"
        )
    return result


def _parse_cardinalities(value: Any) -> tuple[tuple[str, int], ...]:
    rows = _plain_list(value, "feature_cardinalities")
    result = []
    for index, row in enumerate(rows):
        values = _plain_list(row, f"feature_cardinalities[{index}]")
        if len(values) != 2:
            raise ArtifactError(
                "GINSU_ARTIFACT_REPORT", "invalid cardinality row"
            )
        result.append(
            (
                _required_string(values[0], "cardinality feature"),
                _nonnegative_int(values[1], "cardinality value"),
            )
        )
    return tuple(result)


def _parse_schema_pairs(
    value: Any, context: str
) -> tuple[tuple[str, str], ...]:
    result = []
    for index, row in enumerate(_plain_list(value, context)):
        values = _plain_list(row, f"{context}[{index}]")
        if len(values) != 2:
            raise ArtifactError(
                "GINSU_ARTIFACT_SCHEMA", f"{context} row must have two values"
            )
        result.append(
            (
                _required_string(values[0], f"{context} name"),
                _required_string(values[1], f"{context} dtype"),
            )
        )
    if not result or len({name for name, _dtype in result}) != len(result):
        raise ArtifactError("GINSU_ARTIFACT_SCHEMA", f"{context} is invalid")
    return tuple(result)


def _parse_search_parameters(
    value: Any,
) -> tuple[tuple[str, int | float], ...]:
    root = _closed_mapping(
        value, {"alpha", "k", "max_l", "min_sup"}, "search_parameters"
    )
    alpha = _finite_number(root["alpha"], "alpha")
    k = _nonnegative_int(root["k"], "k")
    max_l = _nonnegative_int(root["max_l"], "max_l")
    min_sup = _finite_number(root["min_sup"], "min_sup")
    return (("alpha", alpha), ("k", k), ("max_l", max_l), ("min_sup", min_sup))


def _schema_data(frame: pl.DataFrame) -> list[dict[str, str]]:
    return [
        {"name": name, "dtype": str(dtype)}
        for name, dtype in frame.schema.items()
    ]


def _parse_schema_data(value: Any, context: str) -> list[dict[str, str]]:
    result = []
    for row in _plain_list(value, context):
        item = _closed_mapping(row, {"name", "dtype"}, f"{context} row")
        result.append(
            {
                "name": _required_string(item["name"], f"{context}.name"),
                "dtype": _required_string(item["dtype"], f"{context}.dtype"),
            }
        )
    return result


def _closed_mapping(
    value: Any, keys: set[str], context: str
) -> dict[str, Any]:
    root = _plain_dict(value, context)
    if set(root) != keys:
        raise ArtifactError(
            "GINSU_ARTIFACT_SCHEMA",
            f"{context} keys must be {sorted(keys)!r}; received {sorted(root)!r}",
        )
    return root


def _plain_dict(value: Any, context: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ArtifactError(
            "GINSU_ARTIFACT_SCHEMA", f"{context} must be an object"
        )
    return value


def _plain_list(value: Any, context: str) -> list[Any]:
    if type(value) is not list:
        raise ArtifactError(
            "GINSU_ARTIFACT_SCHEMA", f"{context} must be an array"
        )
    return value


def _required_string(value: Any, context: str) -> str:
    if type(value) is not str or not value or len(value) > 4096:
        raise ArtifactError(
            "GINSU_ARTIFACT_VALUE",
            f"{context} must be a bounded non-empty string",
        )
    return value


def _fingerprint_pairs(
    value: Mapping[str, str], context: str
) -> tuple[tuple[str, str], ...]:
    result = []
    for key, fingerprint in sorted(value.items()):
        result.append(
            (
                _required_string(key, f"{context} key"),
                _required_string(fingerprint, f"{context}[{key!r}]"),
            )
        )
    return tuple(result)


def _nonnegative_int(value: Any, context: str) -> int:
    if type(value) is not int or value < 0:
        raise ArtifactError(
            "GINSU_ARTIFACT_VALUE", f"{context} must be a nonnegative integer"
        )
    return value


def _finite_number(value: Any, context: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ArtifactError(
            "GINSU_ARTIFACT_VALUE", f"{context} must be a finite number"
        )
    return float(value)


def _optional_int(value: Any, context: str) -> int | None:
    if value is None:
        return None
    return _nonnegative_int(value, context)


def _optional_float(value: Any, context: str) -> float | None:
    if value is None:
        return None
    return _finite_number(value, context)


def _string_tuple(value: Any, context: str) -> tuple[str, ...]:
    return tuple(
        _required_string(item, context) for item in _plain_list(value, context)
    )


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _load_json(payload: bytes) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ArtifactError(
                    "GINSU_ARTIFACT_DUPLICATE_KEY",
                    f"duplicate JSON key {key!r}",
                )
            result[key] = value
        return result

    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant {value!r}")
            ),
        )
    except ArtifactError:
        raise
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        ValueError,
    ) as error:
        raise ArtifactError(
            "GINSU_ARTIFACT_JSON", "invalid manifest JSON"
        ) from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
