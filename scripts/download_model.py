from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from src.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable official Hugging Face model download and load test")
    parser.add_argument("--config", default="configs/model.yaml")
    parser.add_argument("--skip-load-test", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Validate and load an existing cached snapshot")
    args = parser.parse_args()
    config = load_yaml(args.config)
    name, revision = config["model_name"], config.get("revision", "main")
    cache_dir = Path(config["cache_dir"]).resolve()
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_XET_CACHE", str(cache_dir / "xet"))
    # The ordinary Hub HTTP backend was validated on this Windows machine.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import model_info, snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    resolved_revision = revision if args.offline else model_info(name, revision=revision).sha
    snapshot = snapshot_download(
        repo_id=name, revision=resolved_revision, cache_dir=cache_dir, local_files_only=args.offline
    )
    required = ["config.json", "tokenizer_config.json"]
    missing = [filename for filename in required if not (Path(snapshot) / filename).exists()]
    weights = list(Path(snapshot).glob("*.safetensors"))
    if missing or not weights:
        raise RuntimeError(f"Incomplete snapshot: missing={missing}, safetensors={len(weights)}")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    result: dict[str, object] = {
        "repo_id": name,
        "requested_revision": revision,
        "resolved_revision": Path(snapshot).name,
        "snapshot_path": snapshot,
        "safetensors_files": len(weights),
        "tokenizer_class": tokenizer.__class__.__name__,
    }
    if not args.skip_load_test:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable for the required 4-bit load test")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            snapshot,
            local_files_only=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            ),
            device_map={"": 0},
        )
        start_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
        inputs = torch.tensor([[start_id]], device="cuda")
        attention_mask = torch.ones_like(inputs)
        with torch.inference_mode():
            output = model.generate(
                inputs,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=16,
                pad_token_id=tokenizer.eos_token_id,
            )
        result["generation_smoke"] = tokenizer.decode(output[0], skip_special_tokens=True)
        result["peak_vram_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)
    stem = Path(args.config).stem
    output_path = Path("logs/model_download.json" if stem == "model" else f"logs/model_download.{stem}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
