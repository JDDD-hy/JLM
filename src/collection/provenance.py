from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def provenance_record(
    *,
    record_id: str,
    quote: str,
    source_platform: str,
    source_type: str,
    collection_method: str,
    source_id: str | None = None,
    source_url: str | None = None,
    scene: int | None = None,
    license_status: str = "pending_review",
    quality_status: str = "unreviewed",
    origin: str = "unknown",
    collected_at: str | None = None,
) -> dict[str, Any]:
    """Build metadata without inventing unavailable timestamps or permissions."""
    return {
        "id": record_id,
        "quote": quote,
        "scene": scene,
        "source_platform": source_platform,
        "source_id": source_id,
        "source_url": source_url,
        "source_type": source_type,
        "collection_method": collection_method,
        "license_status": license_status,
        "origin": origin,
        "quality_status": quality_status,
        "created_at": None,
        "collected_at": collected_at,
    }


def collection_timestamp() -> str:
    return datetime.now(UTC).isoformat()
