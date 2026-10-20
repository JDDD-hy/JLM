from __future__ import annotations

from scripts.chat import clean
from src.data.emoji import to_emoji


def test_rednote_tags_become_emoji_and_doge_is_kept():
    assert to_emoji("就很...[笑哭R]") == "就很...😂"
    assert to_emoji("控分[doge]") == "控分[doge]"
    assert to_emoji("我[捂脸R][捂脸R][捂脸R]") == "我🤦🤦"
    assert to_emoji("路过[黄金薯R]") == "路过"
    assert to_emoji("未知[某某R]表情") == "未知表情"
    assert to_emoji("#考研[话题]#") == "#考研[话题]#"
    assert to_emoji("哈哈哈哈") == "哈哈哈哈"


def test_chat_clean_converts_tags():
    assert clean("会，我写代码[偷笑R]\n我：没毛病[得意R]", multiline=True) == "会，我写代码🤭\n我：没毛病😏"
