"""A small, dependency-free JSON-schema validator for model output.

Only the subset the application's schemas use is implemented - and anything outside it is
*rejected* rather than silently ignored (:func:`check_supported`), so a schema can never
look stricter than its enforcement:

``type`` (single or list), ``properties``, ``required``, ``additionalProperties: false``,
``enum``, ``items``, ``minItems``/``maxItems``, ``minLength``/``maxLength``,
``minimum``/``maximum``, ``anyOf``, plus the annotations ``description``/``title``/
``default``/``format``.

:func:`check_strict` enforces the shape strict structured output needs: every object
declares ``additionalProperties: false`` and a ``required`` list.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

_TYPE_CHECKS: dict[str, Any] = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v),
    "null": lambda v: v is None,
}
_KEYWORDS = frozenset(
    {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "enum",
        "items",
        "minItems",
        "maxItems",
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "anyOf",
        "description",
        "title",
        "default",
        "format",
    }
)
MAX_ERRORS = 20


class SchemaDefinitionError(ValueError):
    """The schema itself uses unsupported keywords or is not strict."""


def _walk(schema: dict[str, Any], path: str) -> Iterator[tuple[str, dict[str, Any]]]:
    yield path, schema
    for name, sub in (schema.get("properties") or {}).items():
        yield from _walk(sub, f"{path}.{name}")
    items = schema.get("items")
    if isinstance(items, dict):
        yield from _walk(items, f"{path}[]")
    for index, variant in enumerate(schema.get("anyOf") or []):
        yield from _walk(variant, f"{path}|{index}")


def check_supported(schema: dict[str, Any]) -> None:
    for path, node in _walk(schema, "$"):
        unknown = set(node) - _KEYWORDS
        if unknown:
            raise SchemaDefinitionError(f"{path}: unsupported keywords {sorted(unknown)}")
        types = node.get("type")
        for type_name in types if isinstance(types, list) else [types] if types else []:
            if type_name not in _TYPE_CHECKS:
                raise SchemaDefinitionError(f"{path}: unsupported type {type_name!r}")


def check_strict(schema: dict[str, Any]) -> None:
    """Every object node must forbid extra properties and list its required keys."""
    check_supported(schema)
    for path, node in _walk(schema, "$"):
        types = node.get("type")
        is_object = types == "object" or (isinstance(types, list) and "object" in types)
        if not is_object:
            continue
        if node.get("additionalProperties") is not False:
            raise SchemaDefinitionError(f"{path}: additionalProperties must be false")
        if not isinstance(node.get("required"), list):
            raise SchemaDefinitionError(f"{path}: 'required' list is missing")
        unknown_required = set(node["required"]) - set(node.get("properties") or {})
        if unknown_required:
            raise SchemaDefinitionError(f"{path}: required names unknown properties")


def validate(instance: Any, schema: dict[str, Any]) -> list[str]:
    """Return human-readable violations (paths + rule names only, never the values)."""
    errors: list[str] = []
    _validate(instance, schema, "$", errors)
    return errors[:MAX_ERRORS]


def is_valid(instance: Any, schema: dict[str, Any]) -> bool:
    return not validate(instance, schema)


def _validate(value: Any, schema: dict[str, Any], path: str, errors: list[str]) -> None:
    if len(errors) >= MAX_ERRORS:
        return
    if "anyOf" in schema and not any(is_valid(value, v) for v in schema["anyOf"]):
        errors.append(f"{path}: matches none of anyOf")
        return
    types = schema.get("type")
    if types is not None:
        names = types if isinstance(types, list) else [types]
        if not any(_TYPE_CHECKS[name](value) for name in names):
            errors.append(f"{path}: expected type {'|'.join(names)}")
            return
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: value not in enum")
    if isinstance(value, str):
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: longer than maxLength {schema['maxLength']}")
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path}: shorter than minLength {schema['minLength']}")
    if isinstance(value, int | float) and not isinstance(value, bool):
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: above maximum")
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: below minimum")
    if isinstance(value, list):
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: more than maxItems {schema['maxItems']}")
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: fewer than minItems {schema['minItems']}")
        items = schema.get("items")
        if isinstance(items, dict):
            for index, item in enumerate(value):
                _validate(item, items, f"{path}[{index}]", errors)
    if isinstance(value, dict):
        properties: dict[str, Any] = schema.get("properties") or {}
        errors.extend(
            f"{path}: missing required property {name!r}"
            for name in schema.get("required") or []
            if name not in value
        )
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value) - set(properties))
            if extra:
                errors.append(f"{path}: {len(extra)} unexpected properties")
        for name, sub in properties.items():
            if name in value:
                _validate(value[name], sub, f"{path}.{name}", errors)
