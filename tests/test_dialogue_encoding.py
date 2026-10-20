from __future__ import annotations

from src.data.prepare import dialogue_prefix, encode_dialogue


class FakeTokenizer:
    eos_token_id = 0
    pad_token_id = 0
    bos_token_id = None
    eos_token = "<eos>"

    def encode(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = [ord(ch) % 1000 + 1 for ch in text]
        return ids[:max_length] if truncation and max_length else ids


def test_prompt_is_loss_masked_and_reply_trained():
    tok = FakeTokenizer()
    out = encode_dialogue(tok, "你好", "我是来送分的", 128)
    prefix_len = 1 + len(dialogue_prefix("你好"))
    assert out["labels"][:prefix_len] == [-100] * prefix_len
    assert out["labels"][prefix_len:] == out["input_ids"][prefix_len:]
    assert out["input_ids"][0] == 0 and out["input_ids"][-1] == 0
    assert len(out["labels"][prefix_len:]) == len("我是来送分的") + 1


def test_history_framing_matches_chat_layout():
    text = dialogue_prefix("在吗", [("你好", "嗯")])
    assert text.endswith("我回复：") and text.count("评论区有人说：") == 2


def test_chat_template_encoding_with_cached_qwen_tokenizer():
    import pytest

    transformers = pytest.importorskip("transformers")
    from src.data.prepare import encode_dialogue_chat

    try:
        tok = transformers.AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-1.7B",
            revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
            cache_dir=".cache/huggingface",
            local_files_only=True,
        )
    except OSError:
        pytest.skip("Qwen3-1.7B chat tokenizer not cached")
    out = encode_dialogue_chat(tok, "你好", "我是来送分的", 128, "你是嘉豪。")
    end_id = tok.convert_tokens_to_ids("<|im_end|>")
    trained = [i for i, label in zip(out["input_ids"], out["labels"], strict=True) if label != -100]
    assert tok.decode(trained[:-1]) == "我是来送分的"
    assert trained[-1] == end_id
    prefix_text = tok.decode(out["input_ids"][: len(out["input_ids"]) - len(trained)])
    assert prefix_text.endswith("<think>\n\n</think>\n\n") and "你是嘉豪。" in prefix_text


def test_handwritten_qwen_prompt_matches_official_template():
    import pytest

    transformers = pytest.importorskip("transformers")
    from src.inference.selection import qwen_chat_prompt

    try:
        tok = transformers.AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-4B",
            revision="1cfa9a7208912126459214e8b04321603b3df60c",
            cache_dir=".cache/huggingface",
            local_files_only=True,
        )
    except OSError:
        pytest.skip("Qwen3-4B tokenizer not cached")
    history = [("你会游泳吗", "会，水看见我都让道。")]
    messages = [{"role": "system", "content": "你是嘉豪。"}]
    for user, reply in history:
        messages += [{"role": "user", "content": user}, {"role": "assistant", "content": reply}]
    messages.append({"role": "user", "content": "那你游多快"})
    official = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert qwen_chat_prompt("那你游多快", history) == official
