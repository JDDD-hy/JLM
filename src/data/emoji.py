"""Convert Xiaohongshu (RedNote) bracket emoji codes like [笑哭R] to Unicode emoji.

[xxxR] is RedNote's own sticker syntax (R = RedNote); some posts also contain real emoji such as 😂.
[doge] is kept as-is by request. Known codes are mapped (with or without the trailing R); codes without a
sensible emoji are removed; unknown [xxxR] stickers are removed; other unknown [xxx] brackets are left alone.
"""

from __future__ import annotations

import re

EMOJI_MAP = {
    "笑哭": "😂",
    "笑哭了": "😂",
    "捂脸": "🤦",
    "偷笑": "🤭",
    "哭惹": "😭",
    "微笑": "🙂",
    "害羞": "☺️",
    "得意": "😏",
    "石化": "🗿",
    "色色": "😍",
    "汗颜": "😅",
    "生气": "😠",
    "派对": "🥳",
    "飞吻": "😘",
    "抱拳": "🙏",
    "嘻嘻": "😁",
    "坏笑": "😈",
    "叹气": "😮‍💨",
    "尬住": "😬",
    "尬笑": "😅",
    "失望": "😞",
    "暗中观察": "👀",
    "泪崩": "😭",
    "哇": "😮",
    "皱眉": "😟",
    "呃": "😑",
    "惊讶": "😲",
    "大笑": "😆",
    "可怜": "🥺",
    "赞": "👍",
    "点赞": "👍",
    "玫瑰": "🌹",
    "握手": "🤝",
    "扶墙": "😵",
    "抽泣": "😢",
    "哭": "😢",
    "心碎": "💔",
    "生病": "🤒",
    "黑薯问号": "❓",
    "拔草": "",
    "抠鼻": "",
    "黄金薯": "",
    "跳迪": "",
    "回招呼": "👋",
    "棒": "👍",
    "怒": "😠",
}

# [笑哭R] / [笑哭] / [doge]; excludes hashtag markers like [话题] handled elsewhere.
STICKER = re.compile(r"\[([^\[\]\s]{1,8}?)(R?)\]")
KEEP = {"话题", "doge"}


def _replace(match: re.Match) -> str:
    name, rednote = match.group(1), match.group(2)
    if name in KEEP:
        return match.group(0)
    if name in EMOJI_MAP:
        return EMOJI_MAP[name]
    return "" if rednote else match.group(0)


def to_emoji(text: str, max_repeat: int = 2) -> str:
    """Replace RedNote sticker codes with emoji, collapse long runs of the same emoji, tidy spaces."""
    converted = STICKER.sub(_replace, text)
    run = re.compile("([\U0001f300-\U0001faff☀-➿])\\1{" + str(max_repeat) + ",}")
    converted = run.sub(lambda m: m.group(1) * max_repeat, converted)
    converted = re.sub(r"[ \t]{2,}", " ", converted)
    return converted.strip()
