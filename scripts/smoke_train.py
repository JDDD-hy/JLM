from __future__ import annotations

import json
from pathlib import Path

from src.config import load_yaml
from src.inference.generate import generate_quotes, load_generation_model
from src.training.train import train


def main() -> None:
    fixture = "tests/fixtures/synthetic_quotes.jsonl"
    metadata = train(
        "configs/model.yaml",
        "configs/smoke_train.yaml",
        fixture,
        fixture,
        max_steps=2,
    )
    if not metadata["lora_parameters_changed"]:
        raise RuntimeError("Smoke training finished but the sampled LoRA parameter did not change")
    adapter = Path("adapters/smoke")
    if not (adapter / "adapter_config.json").exists():
        raise RuntimeError("Adapter was not saved")
    model, tokenizer = load_generation_model(load_yaml("configs/model.yaml"), str(adapter))
    generations = generate_quotes(model, tokenizer, max_new_tokens=12, num_return_sequences=1, seed=42)
    if not generations or not generations[0]:
        raise RuntimeError("Reloaded adapter generated no decoded text")
    result = {
        "forward_backward_completed": True,
        "lora_parameters_changed": metadata["lora_parameters_changed"],
        "adapter_saved": True,
        "adapter_reloaded": True,
        "data_origin": "synthetic",
        "style_quality_validated": False,
        "training_seconds": metadata["metrics"]["train_runtime"],
        "peak_vram_mib": metadata["peak_vram_mib"],
        "trainable_parameters": metadata["parameter_counts"]["trainable"],
        "generation": generations,
    }
    Path("logs").mkdir(exist_ok=True)
    Path("logs/smoke_train.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
