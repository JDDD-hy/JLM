from __future__ import annotations

from scripts.chat import clean, is_comparison_template, is_incomplete_comparison


def test_multiline_reply_keeps_second_half():
    raw = "别人：刷题\n我：给题目出题"
    assert clean(raw) == "别人：刷题"
    assert clean(raw, multiline=True) == "别人：刷题\n我：给题目出题"


def test_incomplete_comparison_detected():
    assert is_incomplete_comparison("别人：期末周")
    assert not is_incomplete_comparison("别人：期末周\n我：期末月")
    assert is_comparison_template("同龄人在玩，我在学")
    assert not is_comparison_template("不晕，一般是车晕我。")
