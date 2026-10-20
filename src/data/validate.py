from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.data.clean import normalize_text


@dataclass
class ValidationIssue:
    line: int
    code: str
    message: str
    raw: str | None = None


@dataclass
class ValidationResult:
    records: list[dict[str, Any]] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)
    sha256: str = ""


def stable_record_id(quote: str, scene: int | None) -> str:
    payload = json.dumps([normalize_text(quote), scene], ensure_ascii=False, separators=(",", ":"))
    return f"jlm_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def validate_record(value: Any, line_number: int) -> tuple[dict[str, Any] | None, ValidationIssue | None]:
    if not isinstance(value, dict):
        return None, ValidationIssue(line_number, "not_object", "JSON value must be an object")
    quote = value.get("quote")
    if not isinstance(quote, str) or not normalize_text(quote):
        return None, ValidationIssue(line_number, "invalid_quote", "quote must be a non-empty string")
    scene = value.get("scene")
    if scene is not None and (not isinstance(scene, int) or isinstance(scene, bool)):
        return None, ValidationIssue(line_number, "invalid_scene", "scene must be an integer or null")
    record = dict(value)
    record["quote"] = normalize_text(quote)
    record["scene"] = scene
    record.setdefault("id", stable_record_id(record["quote"], scene))
    return record, None


def read_jsonl(path: str | Path) -> ValidationResult:
    source = Path(path)
    result = ValidationResult(sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    with source.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                result.issues.append(ValidationIssue(line_number, "blank_line", "line is blank"))
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                result.issues.append(ValidationIssue(line_number, "invalid_json", str(exc), raw=line[:500]))
                continue
            record, issue = validate_record(value, line_number)
            if issue:
                issue.raw = line[:500]
                result.issues.append(issue)
            else:
                if record is not None:
                    record.setdefault("original_line", line_number)
                result.records.append(record or {})
    return result


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
