from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.config import load_yaml
from src.data.prepare import encode_dialogue, encode_dialogue_chat, encode_quote, tokenizer_special_tokens
from src.training.callbacks import GpuMemoryCallback
from src.training.data_gate import verify_training_inputs


@dataclass
class CausalLMCollator:
    tokenizer: Any
    pad_to_multiple_of: int | None = 8

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        longest = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of:
            longest = math.ceil(longest / self.pad_to_multiple_of) * self.pad_to_multiple_of
        pad_id = int(self.tokenizer.pad_token_id)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            padding = longest - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [pad_id] * padding)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * padding)
            batch["labels"].append(feature["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def choose_compute_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    raise RuntimeError("QLoRA requires CUDA in JLM v0.1; CPU/disk offload is intentionally disabled")


def load_qlora_model(model_config: dict[str, Any]) -> tuple[Any, Any, str | None]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing hidden CPU/disk offload")
    model_name = model_config["model_name"]
    cache_dir = model_config.get("cache_dir")
    dtype = choose_compute_dtype()
    quantization = model_config["quantization"]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=bool(quantization["load_in_4bit"]),
        bnb_4bit_quant_type=quantization["quant_type"],
        bnb_4bit_use_double_quant=bool(quantization["use_double_quant"]),
        bnb_4bit_compute_dtype=dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=model_config.get("revision", "main"),
        cache_dir=cache_dir,
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
        local_files_only=bool(model_config.get("local_files_only", False)),
    )
    tokenizer_special_tokens(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=model_config.get("revision", "main"),
        cache_dir=cache_dir,
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
        quantization_config=bnb_config,
        dtype=dtype,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        local_files_only=bool(model_config.get("local_files_only", False)),
    )
    revision = getattr(model.config, "_commit_hash", None)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    lora = model_config["lora"]
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            revision=model_config.get("revision"),
            r=int(lora["r"]),
            lora_alpha=int(lora["alpha"]),
            lora_dropout=float(lora["dropout"]),
            target_modules=lora["target_modules"],
            bias=lora.get("bias", "none"),
        ),
    )
    return model, tokenizer, revision


def parameter_counts(model: Any) -> dict[str, int | float]:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    # Packed 4-bit tensors store two logical weights per uint8 element.
    total = sum(
        parameter.numel() * (2 if parameter.__class__.__name__ == "Params4bit" else 1)
        for parameter in model.parameters()
    )
    return {"trainable": trainable, "total": total, "percent": 100 * trainable / total}


def load_split(
    path: str | Path, tokenizer: Any, max_length: int, model_config: dict[str, Any] | None = None
) -> Dataset:
    with Path(path).open(encoding="utf-8-sig") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if any(not isinstance(row.get("text"), str) or not row["text"].strip() for row in rows):
        raise ValueError(f"Empty or invalid training text in {path}")
    dataset = Dataset.from_list(rows)
    if not len(dataset):
        raise ValueError(f"Dataset is empty: {path}")

    dialogue = "prompt" in dataset.column_names
    chat_format = (model_config or {}).get("dialogue_format") == "chat_template"
    system_prompt = (model_config or {}).get("system_prompt")

    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        if dialogue and chat_format:
            encoded = [
                encode_dialogue_chat(tokenizer, prompt, text, max_length, system_prompt)
                for prompt, text in zip(batch["prompt"], batch["text"], strict=True)
            ]
        elif dialogue:
            encoded = [
                encode_dialogue(tokenizer, prompt, text, max_length)
                for prompt, text in zip(batch["prompt"], batch["text"], strict=True)
            ]
        else:
            encoded = [encode_quote(tokenizer, text, max_length) for text in batch["text"]]
        return {key: [item[key] for item in encoded] for key in encoded[0]}

    columns = dataset.column_names
    return dataset.map(tokenize, batched=True, remove_columns=columns, desc=f"Tokenizing {path}")


def _training_arguments(config: dict[str, Any], *, max_steps: int = -1) -> TrainingArguments:
    dtype = choose_compute_dtype()
    kwargs: dict[str, Any] = {
        "output_dir": config["output_dir"],
        "learning_rate": float(config["learning_rate"]),
        "per_device_train_batch_size": int(config["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(config["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": int(config["gradient_accumulation_steps"]),
        "num_train_epochs": float(config["num_train_epochs"]),
        "max_steps": max_steps,
        "warmup_ratio": float(config["warmup_ratio"]),
        "weight_decay": float(config["weight_decay"]),
        "lr_scheduler_type": config["lr_scheduler_type"],
        "gradient_checkpointing": bool(config["gradient_checkpointing"]),
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "logging_steps": int(config["logging_steps"]),
        "eval_steps": int(config["eval_steps"]),
        "save_steps": int(config["save_steps"]),
        "save_total_limit": int(config["save_total_limit"]),
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "bf16": dtype == torch.bfloat16,
        "fp16": dtype == torch.float16,
        "optim": "paged_adamw_8bit",
        "report_to": "none",
        "seed": int(config["seed"]),
        "data_seed": int(config["seed"]),
        "remove_unused_columns": False,
    }
    signature = inspect.signature(TrainingArguments.__init__)
    kwargs["eval_strategy" if "eval_strategy" in signature.parameters else "evaluation_strategy"] = "steps"
    kwargs["save_strategy"] = "steps"
    return TrainingArguments(**kwargs)


def train(
    model_config_path: str,
    train_config_path: str,
    train_path: str,
    validation_path: str,
    max_steps: int = -1,
    resume_from_checkpoint: str | None = None,
    unreviewed_pilot: bool = False,
) -> dict[str, Any]:
    model_config = load_yaml(model_config_path)
    train_config = load_yaml(train_config_path)
    if train_config.get("packing", False):
        raise ValueError("Packing is not implemented in v0.1. Keep packing=false for independent quotes.")
    if unreviewed_pilot and not all("pilot" in str(train_config[key]).lower() for key in ("adapter_dir", "output_dir")):
        raise ValueError("Unreviewed pilot runs must write to adapter/output paths containing 'pilot'")
    data_scope = verify_training_inputs(train_path, validation_path, allow_unreviewed_pilot=unreviewed_pilot)
    set_seed(int(train_config["seed"]))
    started = time.perf_counter()
    try:
        model, tokenizer, revision = load_qlora_model(model_config)
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(
            "CUDA OOM while loading QLoRA. Close GPU users, keep batch_size=1, then reduce "
            "max_seq_length before changing the base model."
        ) from exc
    counts = parameter_counts(model)
    if any(parameter.requires_grad and "lora_" not in name for name, parameter in model.named_parameters()):
        raise RuntimeError("A non-LoRA parameter is trainable")
    first_trainable_name, first_trainable = next(
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    initial_trainable = first_trainable.detach().float().cpu().clone()
    print(json.dumps({"parameter_counts": counts}, indent=2))
    train_dataset = load_split(train_path, tokenizer, int(train_config["max_seq_length"]), model_config)
    eval_dataset = load_split(validation_path, tokenizer, int(train_config["max_seq_length"]), model_config)
    args = _training_arguments(train_config, max_steps=max_steps)
    effective_steps = (
        max_steps
        if max_steps > 0
        else math.ceil(
            len(train_dataset)
            * float(train_config["num_train_epochs"])
            / (int(train_config["gradient_accumulation_steps"]) * int(train_config["per_device_train_batch_size"]))
        )
    )
    print(json.dumps({"estimated_optimizer_steps": effective_steps, "model_revision": revision}))
    callbacks: list[Any] = [GpuMemoryCallback()]
    if max_steps < 0:
        callbacks.append(EarlyStoppingCallback(int(train_config["early_stopping_patience"])))
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CausalLMCollator(tokenizer),
        callbacks=callbacks,
    )
    resume = resume_from_checkpoint or train_config.get("resume_from_checkpoint")
    try:
        train_result = trainer.train(resume_from_checkpoint=resume or None)
    except torch.cuda.OutOfMemoryError as exc:
        memory = torch.cuda.memory_summary(abbreviated=True) if torch.cuda.is_available() else "CUDA unavailable"
        raise RuntimeError(f"CUDA OOM during training. Memory summary:\n{memory}") from exc
    trained_parameter = dict(model.named_parameters())[first_trainable_name].detach().float().cpu()
    lora_parameters_changed = not torch.equal(initial_trainable, trained_parameter)
    adapter_dir = Path(train_config["adapter_dir"])
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(adapter_dir)
    trainer.save_state()
    tokenizer.save_pretrained(adapter_dir)
    metrics = dict(train_result.metrics)
    metrics["duration_seconds_wall"] = time.perf_counter() - started
    metadata = {
        "base_model": model_config["model_name"],
        "data_scope": data_scope,
        "data_sha256": {
            path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in [train_path, validation_path]
        },
        "model_revision": revision,
        "parameter_counts": counts,
        "lora_parameters_changed": lora_parameters_changed,
        "compute_dtype": str(choose_compute_dtype()),
        "torch_version": torch.__version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "train_config": train_config,
        "model_config": model_config,
        "metrics": metrics,
        "global_step": trainer.state.global_step,
        "log_history": trainer.state.log_history,
        "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 2**20),
    }
    (adapter_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    trainer.accelerator.free_memory()
    del trainer, model, first_trainable
    gc.collect()
    torch.cuda.empty_cache()
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a QLoRA adapter for quote-only causal LM")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--train-file", default="data/splits/train.jsonl")
    parser.add_argument("--validation-file", default="data/splits/validation.jsonl")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument(
        "--unreviewed-pilot",
        action="store_true",
        help="Train on assistant-triaged data explicitly marked as an unreviewed pilot (pilot paths only)",
    )
    args = parser.parse_args()
    metadata = train(
        args.model_config,
        args.train_config,
        args.train_file,
        args.validation_file,
        args.max_steps,
        args.resume_from_checkpoint,
        args.unreviewed_pilot,
    )
    print(json.dumps(metadata["metrics"], indent=2, default=str))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
