from __future__ import annotations

import argparse
import json
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

from src.config import load_yaml


def _dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def load_generation_model(
    model_config: dict[str, Any], adapter_path: str | None = None, quantized: bool = True
) -> tuple[Any, Any]:
    name = model_config["model_name"]
    cache_dir = model_config.get("cache_dir")
    common = {
        "cache_dir": cache_dir,
        "revision": model_config.get("revision", "main"),
        "local_files_only": bool(model_config.get("local_files_only", False)),
    }
    tokenizer = AutoTokenizer.from_pretrained(name, **common)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs: dict[str, Any] = {**common, "low_cpu_mem_usage": True}
    if torch.cuda.is_available():
        kwargs["device_map"] = {"": 0}
        kwargs["dtype"] = _dtype()
        if quantized:
            quantization = model_config["quantization"]
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=quantization["quant_type"],
                bnb_4bit_use_double_quant=bool(quantization["use_double_quant"]),
                bnb_4bit_compute_dtype=_dtype(),
            )
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    model.config.use_cache = True
    return model, tokenizer


def generate_quotes(
    model: Any,
    tokenizer: Any,
    *,
    prefix: str = "",
    temperature: float = 0.9,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    max_new_tokens: int = 96,
    num_return_sequences: int = 5,
    seed: int = 42,
) -> list[str]:
    set_seed(seed)
    if prefix:
        encoded = tokenizer(prefix, return_tensors="pt", add_special_tokens=True)
        prompt_length = encoded["input_ids"].shape[1]
    else:
        start_id = tokenizer.bos_token_id
        if start_id is None:
            start_id = tokenizer.eos_token_id
        if start_id is None:
            raise ValueError("Tokenizer provides neither BOS nor EOS for empty-prefix generation")
        encoded = {
            "input_ids": torch.tensor([[start_id]], dtype=torch.long),
            "attention_mask": torch.ones((1, 1), dtype=torch.long),
        }
        prompt_length = 1
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    if num_return_sequences < 1 or max_new_tokens < 1 or temperature <= 0:
        raise ValueError("Sample count, token budget and temperature must be positive")
    results: list[str] = []
    with torch.inference_mode():
        # Keep KV-cache memory bounded on an 8GB GPU, including a 50+ sample evaluation.
        for offset in range(0, num_return_sequences, 4):
            outputs = model.generate(
                **encoded,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
                num_return_sequences=min(4, num_return_sequences - offset),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            results.extend(
                tokenizer.decode(output[prompt_length:], skip_special_tokens=True).strip() for output in outputs
            )
    return results


def main() -> None:
    defaults = load_yaml("configs/generation.yaml")
    parser = argparse.ArgumentParser(description="Generate independent JLM quote samples")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--prefix", default=defaults["prefix"])
    parser.add_argument("--temperature", type=float, default=defaults["temperature"])
    parser.add_argument("--top-p", type=float, default=defaults["top_p"])
    parser.add_argument("--top-k", type=int, default=defaults["top_k"])
    parser.add_argument("--repetition-penalty", type=float, default=defaults["repetition_penalty"])
    parser.add_argument("--max-new-tokens", type=int, default=defaults["max_new_tokens"])
    parser.add_argument("--num-return-sequences", type=int, default=defaults["num_return_sequences"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--no-quantization", action="store_true")
    args = parser.parse_args()
    model, tokenizer = load_generation_model(
        load_yaml(args.model_config), args.adapter, quantized=not args.no_quantization
    )
    quotes = generate_quotes(
        model,
        tokenizer,
        prefix=args.prefix,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens,
        num_return_sequences=args.num_return_sequences,
        seed=args.seed,
    )
    print(json.dumps(quotes, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
