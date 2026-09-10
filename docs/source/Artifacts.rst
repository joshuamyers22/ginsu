Safe analysis artifacts
=======================

``SliceAnalysis`` saves fitted result tables and execution evidence without
pickling an estimator or embedding source observations.

.. code:: python

   from ginsu import ArtifactLimits, SliceAnalysis

   analysis = SliceAnalysis.from_finder(
       finder,
       dataset_fingerprint="sha256:caller-computed-discovery-digest",
       partition_fingerprints={
           "discovery": "sha256:caller-computed-discovery-digest",
           "validation": "sha256:caller-computed-validation-digest",
       },
       discretization=binning,
       dependency_lock_id="sha256:caller-computed-uv-lock-digest",
   )
   analysis.write("artifacts/model-v2-slices")

   restored = SliceAnalysis.read(
       "artifacts/model-v2-slices",
       limits=ArtifactLimits(),
   )

Format version 2
----------------

The artifact is a directory containing canonical UTF-8 ``manifest.json`` and
three uncompressed Arrow IPC tables: ``slices.arrow``,
``slice_statistics.arrow``, and ``predicates.arrow``. The manifest records the
exact path, SHA-256 digest, byte size, row count, and ordered schema for every
table. It also records:

- artifact, canonicalization, package, and algorithm versions;
- the ordered feature schema;
- search parameters, limits, per-level report, termination status, per-stage
  timing, and optional labeled boundary-memory observations;
- a complete fitted discretization specification when supplied; and
- caller-supplied dataset, partition, and dependency-lock fingerprints.

Raw observations, per-row membership, executable objects, HTML, callables, and
arbitrary class references are never included. Ginsu does not calculate a
dataset fingerprint implicitly because the caller must define which columns,
ordering, partition identity, and privacy treatment belong to that digest.

Fail-closed loading
-------------------

The reader checks metadata, table, total-byte, row, and materialized-size
limits before or during materialization. It rejects symlinks, nested or
unexpected files, duplicate
JSON keys, nonstandard JSON constants, path changes, unknown versions, hashes,
declared sizes, schemas, canonical slice IDs, predicate values, ranks, and
cross-table relationships that do not validate.

The defaults allow a 1 MiB manifest, 64 MiB per table, 256 MiB total, one
million rows per table, and 128 MiB of materialized data per table. Raising
them is an explicit trust and resource decision. Ginsu writes Arrow IPC
uncompressed in version 2. The reader also checks materialized size because an
untrusted producer could still supply a compressed IPC body; callers handling
hostile files should combine these limits with operating-system process limits.

Writes use a sibling staging directory and never overwrite an existing
destination.

Compatibility and migration
---------------------------

The writer emits ``ginsu.slice-analysis`` version ``2.0.0`` and
canonicalization version 1. The reader also accepts legacy version ``1.0.0``;
its absent stage-timing and memory fields are restored as empty or unavailable
evidence rather than inferred values. Unknown versions fail closed. A future
format change must add a reviewed reader or explicit migration function with
golden round-trip and hostile-input tests; files must not be edited ad hoc to
bypass a version check.

Memory sampling is opt-in at fit time. The manifest stores only integer byte
observations and the caller-declared measurement label, never the sampler
callable. Values are observations at stage boundaries, not allocation
attribution or a continuous peak unless the configured sampler itself reports
peak-to-date values.
