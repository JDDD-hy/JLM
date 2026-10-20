from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from rapidfuzz.fuzz import ratio

from src.data.clean import canonical_for_dedup


@dataclass(frozen=True)
class DuplicatePair:
    left_id: str
    right_id: str
    similarity: float
    kind: str


def exact_deduplicate(records: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[DuplicatePair]]:
    kept: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    duplicates: list[DuplicatePair] = []
    retained_by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        key = canonical_for_dedup(record["quote"])
        record_id = str(record["id"])
        if key in seen:
            duplicates.append(DuplicatePair(seen[key], record_id, 100.0, "exact"))
            retained = retained_by_id[seen[key]]
            sources = set(retained.get("source_groups", []))
            for item in (retained, record):
                sources.update(item.get("source_groups", []))
                source_group = item.get("source_group") or item.get("source_id")
                if source_group:
                    sources.add(str(source_group))
            retained["source_groups"] = sorted(sources)
        else:
            seen[key] = record_id
            retained = dict(record)
            kept.append(retained)
            retained_by_id[record_id] = retained
    return kept, duplicates


def near_duplicate_candidates(records: list[dict[str, Any]], threshold: float = 92.0) -> list[DuplicatePair]:
    """Return review candidates; never deletes near duplicates automatically."""
    pairs: list[DuplicatePair] = []
    normalized = [canonical_for_dedup(record["quote"]) for record in records]
    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            score = float(ratio(normalized[left], normalized[right]))
            if threshold <= score < 100.0:
                pairs.append(DuplicatePair(str(records[left]["id"]), str(records[right]["id"]), score, "near"))
    return sorted(pairs, key=lambda pair: pair.similarity, reverse=True)
