"""Utilities for loading YAML prompt assets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class PromptError(RuntimeError):
    """Raised when prompt assets cannot be loaded or rendered."""


_PROMPTS_ROOT = Path(__file__).resolve().parent
_DOC_CACHE: dict[str, dict[str, Any]] = {}
_PROMPT_CACHE: dict[str, str] = {}
_LINES_CACHE: dict[str, list[str]] = {}
_SCHEMA_CACHE: dict[str, dict[str, str]] = {}


def _prompt_path(name: str) -> Path:
    return _PROMPTS_ROOT / f"{name}.yaml"


def _load_doc(name: str) -> dict[str, Any]:
    if name in _DOC_CACHE:
        return _DOC_CACHE[name]

    path = _prompt_path(name)
    if not path.exists():
        raise PromptError(f"Prompt file not found for '{name}': {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PromptError(f"Invalid YAML for prompt '{name}' at {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise PromptError(f"Prompt '{name}' must be a YAML object at {path}")

    _DOC_CACHE[name] = raw
    return raw


def _resolve_prompt(name: str, stack: tuple[str, ...]) -> str:
    if name in _PROMPT_CACHE:
        return _PROMPT_CACHE[name]
    if name in stack:
        trace = " -> ".join((*stack, name))
        raise PromptError(f"Cyclic prompt reference detected: {trace}")

    doc = _load_doc(name)
    path = _prompt_path(name)
    has_content = "content" in doc
    has_parts = "parts" in doc
    if has_content == has_parts:
        raise PromptError(
            f"Prompt '{name}' at {path} must define exactly one of 'content' or 'parts'"
        )

    if has_content:
        content = doc["content"]
        if not isinstance(content, str):
            raise PromptError(f"Prompt '{name}' content must be a string at {path}")
        _PROMPT_CACHE[name] = content
        return content

    parts = doc["parts"]
    if not isinstance(parts, list):
        raise PromptError(f"Prompt '{name}' parts must be a list at {path}")

    resolved_parts: list[str] = []
    next_stack = (*stack, name)
    for idx, part in enumerate(parts):
        if isinstance(part, str):
            resolved_parts.append(part)
            continue
        if isinstance(part, dict) and set(part) == {"$ref"}:
            ref_name = part["$ref"]
            if not isinstance(ref_name, str):
                raise PromptError(
                    f"Prompt '{name}' has non-string $ref at index {idx} in {path}"
                )
            resolved_parts.append(_resolve_prompt(ref_name, next_stack))
            continue
        raise PromptError(
            f"Prompt '{name}' has invalid part at index {idx} in {path}; "
            "expected a string or {$ref: <name>}"
        )

    resolved = "".join(resolved_parts)
    _PROMPT_CACHE[name] = resolved
    return resolved


def load_prompt(name: str) -> str:
    """Load a prompt by logical name."""
    return _resolve_prompt(name, ())


def load_lines(name: str) -> list[str]:
    """Load an ordered list fragment by logical name."""
    if name in _LINES_CACHE:
        return list(_LINES_CACHE[name])

    doc = _load_doc(name)
    path = _prompt_path(name)
    lines = doc.get("lines")
    if not isinstance(lines, list):
        raise PromptError(f"Prompt '{name}' must define 'lines' as a list at {path}")

    normalized: list[str] = []
    for idx, line in enumerate(lines):
        if not isinstance(line, str):
            raise PromptError(
                f"Prompt '{name}' line at index {idx} must be a string in {path}"
            )
        normalized.append(line)

    _LINES_CACHE[name] = normalized
    return list(normalized)


def render(name: str, **kwargs: Any) -> str:
    """Render a prompt with Python str.format placeholders."""
    template = load_prompt(name)
    try:
        return template.format(**kwargs)
    except KeyError as exc:
        missing = exc.args[0]
        raise PromptError(
            f"Missing placeholder '{missing}' when rendering prompt '{name}'"
        ) from exc


def load_schema(name: str) -> dict[str, str]:
    """Load a schema description mapping by logical name."""
    if name in _SCHEMA_CACHE:
        return dict(_SCHEMA_CACHE[name])

    doc = _load_doc(name)
    path = _prompt_path(name)
    fields = doc.get("fields")
    if not isinstance(fields, dict):
        raise PromptError(f"Prompt '{name}' must define 'fields' mapping at {path}")

    normalized: dict[str, str] = {}
    for key, value in fields.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise PromptError(
                f"Prompt '{name}' fields must map string keys to string values in {path}"
            )
        normalized[key] = value

    _SCHEMA_CACHE[name] = normalized
    return dict(normalized)

