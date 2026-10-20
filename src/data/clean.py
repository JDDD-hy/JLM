from __future__ import annotations

import html
import re
import unicodedata

_UI_LINES = {
    "展开全文",
    "收起",
    "复制链接",
    "查看全部评论",
    "加载中",
}
_ZERO_WIDTH = re.compile(r"[\u200b-\u200f\u2060\ufeff]")
_HORIZONTAL_SPACE = re.compile(r"[^\S\n]+")
_EXCESS_NEWLINES = re.compile(r"\n{3,}")


def normalize_text(value: str) -> str:
    """Normalize transport noise without correcting grammar or intentional spelling."""
    text = html.unescape(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)
    text = _ZERO_WIDTH.sub("", text)
    text = _HORIZONTAL_SPACE.sub(" ", text)
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line and line not in _UI_LINES]
    return _EXCESS_NEWLINES.sub("\n\n", "\n".join(lines)).strip()


def canonical_for_dedup(value: str) -> str:
    """Create a conservative exact-dedup key while preserving the exported quote."""
    normalized = normalize_text(value).casefold()
    return re.sub(r"\s+", "", normalized)
