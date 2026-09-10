# ADR 0004: Safe declarative SliceAnalysis artifacts

- Status: Accepted
- Date: 2026-09-09

## Context

Ginsu analyses need to move between discovery, validation, comparison, and
reporting processes without trusting executable estimator pickles. The artifact
must preserve canonical rules, typed result tables, search evidence, fitted
discretization, and caller-supplied provenance while rejecting untrusted or
oversized input before expensive materialization where possible.

## Decision

Artifact format `ginsu.slice-analysis` version `1.0.0` is a directory containing:

- `manifest.json`, encoded as canonical UTF-8 JSON with a closed schema; and
- uncompressed Arrow IPC files for slices, slice statistics, and predicates.

The manifest records exact relative paths, byte sizes, row counts, ordered
schemas, and SHA-256 hashes for every table. It also records the ordered fitted
feature schema, canonicalization version, package and algorithm versions,
search parameters and report, active limits, fitted discretization plan when
present, dependency-lock identity when supplied, and caller-supplied dataset or
partition fingerprints. Raw observations and row membership are excluded.

The reader accepts only the exact version it implements. It rejects unexpected
files or keys, symlinks, absolute or traversing paths, duplicate JSON keys,
unknown versions, hash/size/row/schema mismatches, excessive materialized table
size, broken table relationships, and configured resource-limit violations. No
module name, callable, HTML, pickle, or arbitrary object payload is loaded from
the artifact.

Writes are staged in a sibling temporary directory and renamed into place. An
existing destination is never overwritten.

## Consequences

- Polars can read and write the artifact with core dependencies only.
- Uncompressed IPC is larger than compressed Parquet but gives a useful normal
  pre-materialization byte bound. Because an attacker can construct a compressed
  IPC body, hostile-file deployments should also impose process-level memory
  limits; Ginsu checks materialized size immediately after reading.
- Version 1 has exactly three result tables. Validation, stability, selection,
  and comparison tables require a compatible schema-version extension rather
  than ad hoc files.
- Migration is explicit: unknown versions fail closed until a reviewed reader
  and migration test are added.
