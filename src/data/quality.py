from __future__ import annotations

import re
from dataclasses import dataclass

LABELS = {
    "genuine_jiahao_style",
    "possible_jiahao_style",
    "meta_discussion",
    "ordinary_text",
    "spam",
    "uncertain",
}

_META = re.compile(r"(什么是嘉豪|嘉豪是什么|太嘉豪了|这很嘉豪|嘉豪文学是什么意思)")
_SPAM = re.compile(r"(加微|私信领取|免费咨询|点击链接|代购|招代理)")
_LOW_CONTENT = re.compile(r"^(哈+|哈哈哈+|[\W_]+)$", re.IGNORECASE)
_STYLE_SIGNALS = (
    re.compile(r"(?:我|本人|哥)[^。！？]{0,20}(?:世界|命运|规则|时代|宇宙|强者|实力)"),
    re.compile(r"(?:不是|并非).{0,16}(?:而是|只是|因为)"),
    re.compile(r"(?:他们|别人|众人).{0,16}(?:不懂|畏惧|以为|看见)"),
)


@dataclass(frozen=True)
class QualitySuggestion:
    label: str
    reasons: tuple[str, ...]
    requires_human_review: bool = True


def suggest_quality_label(text: str) -> QualitySuggestion:
    """Conservative rule triage; output is never treated as a final label."""
    if _LOW_CONTENT.fullmatch(text.strip()) or _SPAM.search(text):
        return QualitySuggestion("spam", ("low-content or promotional pattern",))
    if _META.search(text):
        return QualitySuggestion("meta_discussion", ("discussion about the style rather than a quote",))
    hits = sum(bool(pattern.search(text)) for pattern in _STYLE_SIGNALS)
    if hits >= 2:
        return QualitySuggestion("possible_jiahao_style", (f"{hits} conservative style signals",))
    if len(text.strip()) < 4:
        return QualitySuggestion("ordinary_text", ("too short for reliable style judgment",))
    return QualitySuggestion("uncertain", ("rules are intentionally non-authoritative",))


def is_training_eligible(record: dict, require_license_approved: bool = True) -> bool:
    if record.get("quality_status") != "genuine_jiahao_style":
        return False
    if require_license_approved and record.get("license_status") != "approved_for_training":
        return False
    return True
