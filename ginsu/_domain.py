"""Immutable slice-domain values and stable canonical identifiers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import numpy as np


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def canonical_value(value: Any) -> dict[str, Any]:
    """Encode supported scalar values without losing their semantic type."""
    value = _python_scalar(value)
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": str(value)}
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("Predicate float values must be finite.")
        return {"type": "float", "value": value.hex()}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    if isinstance(value, datetime):
        return {"type": "datetime", "value": value.isoformat()}
    if isinstance(value, date):
        return {"type": "date", "value": value.isoformat()}
    raise TypeError(
        f"Unsupported predicate value type: {type(value).__name__}."
    )


def canonical_json(value: Any) -> str:
    """Serialize one scalar in the stable Ginsu v1 encoding."""
    return json.dumps(
        canonical_value(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def value_from_canonical_json(payload: str) -> Any:
    """Decode one strictly canonical Ginsu scalar representation."""
    if type(payload) is not str:
        raise TypeError("Canonical predicate payload must be a string.")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate canonical predicate key {key!r}.")
            result[key] = value
        return result

    try:
        data = json.loads(payload, object_pairs_hook=reject_duplicate_keys)
    except (json.JSONDecodeError, RecursionError) as error:
        raise ValueError("Invalid canonical predicate JSON.") from error
    if type(data) is not dict or set(data) != {"type", "value"}:
        raise ValueError("Canonical predicate JSON has an invalid schema.")
    kind = data["type"]
    value = data["value"]
    if kind == "bool" and type(value) is bool:
        result: Any = value
    elif kind == "int" and type(value) is str:
        try:
            result = int(value)
        except ValueError as error:
            raise ValueError("Invalid canonical integer predicate.") from error
        if str(result) != value:
            raise ValueError("Integer predicate is not canonically encoded.")
    elif kind == "float" and type(value) is str:
        try:
            result = float.fromhex(value)
        except ValueError as error:
            raise ValueError("Invalid canonical float predicate.") from error
        if not np.isfinite(result) or result.hex() != value:
            raise ValueError("Float predicate is not canonically encoded.")
    elif kind == "str" and type(value) is str:
        result = value
    elif kind == "date" and type(value) is str:
        try:
            result = date.fromisoformat(value)
        except ValueError as error:
            raise ValueError("Invalid canonical date predicate.") from error
        if result.isoformat() != value:
            raise ValueError("Date predicate is not canonically encoded.")
    elif kind == "datetime" and type(value) is str:
        try:
            result = datetime.fromisoformat(value)
        except ValueError as error:
            raise ValueError(
                "Invalid canonical datetime predicate."
            ) from error
        if result.tzinfo is not None or result.isoformat() != value:
            raise ValueError("Datetime predicate is not canonically encoded.")
    else:
        raise ValueError("Unsupported canonical predicate type or value.")

    if canonical_json(result) != payload:
        raise ValueError("Predicate payload is not in canonical JSON form.")
    return result


@dataclass(frozen=True, slots=True)
class Predicate:
    """An immutable equality predicate."""

    feature: str
    value: Any
    operator: str = "eq"

    def __post_init__(self) -> None:
        if not self.feature:
            raise ValueError("Predicate feature names must not be empty.")
        if self.operator != "eq":
            raise ValueError(
                "Only equality predicates are currently supported."
            )
        canonical_value(self.value)

    @property
    def value_json(self) -> str:
        """Stable typed JSON representation of the predicate value."""
        return canonical_json(self.value)

    @property
    def display_value(self) -> str:
        """Human-readable value that is never parsed as data."""
        value = _python_scalar(self.value)
        return (
            value.isoformat()
            if isinstance(value, (date, datetime))
            else str(value)
        )

    def canonical_data(self) -> dict[str, Any]:
        """Return the versioned serialization payload for this predicate."""
        return {
            "feature": self.feature,
            "operator": self.operator,
            "value": canonical_value(self.value),
        }


@dataclass(frozen=True, slots=True)
class Slice:
    """An immutable, canonically ordered conjunction of predicates."""

    predicates: tuple[Predicate, ...]

    def __post_init__(self) -> None:
        ordered = tuple(
            sorted(
                self.predicates,
                key=lambda item: (
                    item.feature,
                    item.operator,
                    item.value_json,
                ),
            )
        )
        if len({predicate.feature for predicate in ordered}) != len(ordered):
            raise ValueError(
                "A slice may contain at most one predicate per feature."
            )
        object.__setattr__(self, "predicates", ordered)

    @property
    def id(self) -> str:
        """Stable content identifier using the Ginsu v1 encoding."""
        payload = json.dumps(
            {
                "predicates": [
                    predicate.canonical_data() for predicate in self.predicates
                ],
                "version": 1,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        return f"ginsu:v1:{hashlib.sha256(payload).hexdigest()}"

    @property
    def rule(self) -> str:
        """Escaped display-only conjunction."""
        return " and ".join(
            f"{predicate.feature} == {predicate.value_json}"
            for predicate in self.predicates
        )

    def is_parent_of(self, other: Slice) -> bool:
        """Return whether this rule is an exact proper predicate subset."""
        return len(self.predicates) < len(other.predicates) and set(
            self.predicates
        ).issubset(other.predicates)
