from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any


class CollectionError(RuntimeError):
    pass


def load_checkpoint(path: str | Path) -> set[str]:
    checkpoint = Path(path)
    if not checkpoint.exists():
        return set()
    value = json.loads(checkpoint.read_text(encoding="utf-8"))
    return set(value.get("completed_source_ids", []))


def save_checkpoint(path: str | Path, completed: set[str]) -> None:
    checkpoint = Path(path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text(json.dumps({"completed_source_ids": sorted(completed)}, indent=2) + "\n", encoding="utf-8")


def incremental_collect(
    source_ids: Iterable[str],
    fetch: Callable[[str], dict[str, Any]],
    checkpoint_path: str | Path,
    retries: int = 2,
    backoff_seconds: float = 1.0,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Generic retry/checkpoint loop for user-authorized read-only source adapters."""
    completed = load_checkpoint(checkpoint_path)
    outputs: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for source_id in source_ids:
        if source_id in completed:
            continue
        for attempt in range(retries + 1):
            try:
                payload = fetch(source_id)
                # Persist versioned responses before acknowledging completion, so a crash cannot lose results.
                encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
                digest = hashlib.sha256(encoded).hexdigest()
                archive = Path(checkpoint_path).parent / "responses" / f"{digest}.json"
                archive.parent.mkdir(parents=True, exist_ok=True)
                if not archive.exists():
                    archive.write_bytes(encoded)
                outputs.append(payload)
                completed.add(source_id)
                save_checkpoint(checkpoint_path, completed)
                break
            except Exception as exc:  # boundary adapter records external failures
                if attempt == retries:
                    failures.append({"source_id": source_id, "error": str(exc)})
                else:
                    time.sleep(backoff_seconds * (2**attempt))
    return outputs, failures
