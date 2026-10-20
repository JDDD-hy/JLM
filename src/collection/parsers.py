from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from src.data.clean import normalize_text

_TEXT_KEYS = ("content", "text", "desc", "description")
_COMMENT_KEYS = ("comments", "comment_list", "items")


def _first_text(value: dict[str, Any]) -> str | None:
    for key in _TEXT_KEYS:
        candidate = value.get(key)
        if isinstance(candidate, str) and normalize_text(candidate):
            return normalize_text(candidate)
    return None


def parse_feed_detail(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract only text plus source identifiers from common read-only CLI payload shapes.

    Usernames, avatars, account IDs, and interaction metrics are intentionally dropped.
    """
    root = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    note = root.get("note") or root.get("noteCard") or root
    feed_id = note.get("noteId") or note.get("id") or note.get("note_id") or root.get("feed_id")
    extracted: list[dict[str, Any]] = []
    note_text = _first_text(note)
    if note_text:
        extracted.append({"text": note_text, "source_type": "public_note", "source_id": feed_id})
    comments: Iterable[Any] = []
    for key in _COMMENT_KEYS:
        if isinstance(root.get(key), list):
            comments = root[key]
            break
    for comment in comments:
        if not isinstance(comment, dict):
            continue
        text = _first_text(comment)
        if not text:
            continue
        comment_id = comment.get("id") or comment.get("comment_id")
        extracted.append(
            {
                "text": text,
                "source_type": "public_comment",
                "source_id": f"{feed_id}:{comment_id}" if comment_id else str(feed_id or ""),
            }
        )
    return extracted
