"""Bounded engineering evaluation; all training/evaluation data here is synthetic."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import torch
from peft import PeftModel

from src.config import load_yaml
from src.data.prepare import prepare_splits
from src.evaluation.evaluate import diversity_metrics, evaluate_loss, read_texts
from src.evaluation.novelty import novelty_dicts
from src.evaluation.report import render_markdown
from src.inference.generate import generate_quotes, load_generation_model
from src.training.train import load_split, train


def main() -> None:
    output = Path("outputs/engineering")
    output.mkdir(parents=True, exist_ok=True)
    fixture = "tests/fixtures/synthetic_quotes.jsonl"
    # Recreate the two-step checkpoint so this command is repeatable after cache/checkpoint cleanup.
    initial = train("configs/model.yaml", "configs/smoke_train.yaml", fixture, fixture, max_steps=2)
    if not initial["lora_parameters_changed"]:
        raise RuntimeError("LoRA did not update")
    del initial
    gc.collect()
    torch.cuda.empty_cache()
    resumed = train(
        "configs/model.yaml",
        "configs/smoke_resume.yaml",
        fixture,
        fixture,
        max_steps=3,
        resume_from_checkpoint="checkpoints/smoke/checkpoint-2",
    )
    if resumed["global_step"] != 3:
        raise RuntimeError("Resume failed to reach step 3")
    del resumed
    gc.collect()
    torch.cuda.empty_cache()
    model, tokenizer = load_generation_model(load_yaml("configs/model.yaml"))
    dataset = load_split(fixture, tokenizer, 64)
    training_quotes = read_texts(fixture)
    settings = {**load_yaml("configs/generation.yaml"), "num_return_sequences": 4, "max_new_tokens": 24}
    base_samples = generate_quotes(model, tokenizer, **settings)
    base_loss = evaluate_loss(model, dataset, tokenizer)
    model = PeftModel.from_pretrained(model, "adapters/smoke-resumed")
    adapted_samples = generate_quotes(model, tokenizer, **settings)
    report = {
        "scope": "synthetic engineering acceptance only; NOT a style or held-out generalization result",
        "resume_from_step": 2,
        "adapter_path": "adapters/smoke-resumed",
        "resume_to_step": 3,
        "sampling": settings,
        "model_revision": load_yaml("configs/model.yaml")["revision"],
        "base_language_modeling": base_loss,
        "language_modeling": evaluate_loss(model, dataset, tokenizer),
        "base_diversity": diversity_metrics(base_samples, tokenizer),
        "diversity": diversity_metrics(adapted_samples, tokenizer),
        "novelty": novelty_dicts(adapted_samples, training_quotes),
        "base_samples": base_samples,
        "generations": adapted_samples,
        "style_fidelity": "not_evaluated_requires_real_data_and_human_blind_review",
    }
    records = [
        {"id": str(index), "quote": quote, "origin": "synthetic", "source_id": str(index)}
        for index, quote in enumerate(training_quotes)
    ]
    report["synthetic_split_statistics"] = prepare_splits(
        records,
        tokenizer,
        output / "splits",
        {"train": 0.8, "validation": 0.1, "test": 0.1},
        42,
        64,
        99,
    )
    (output / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = render_markdown(report)
    markdown = markdown.replace(
        "# JLM v0.1 Evaluation Report",
        "# JLM v0.1 Synthetic Engineering Evaluation\n\n"
        "All examples are synthetic; evaluation includes training samples. "
        "These metrics verify execution only, not style fidelity or generalization.",
    )
    (output / "evaluation.md").write_text(markdown, encoding="utf-8")
    print(
        json.dumps(
            {
                "resume_passed": True,
                "base_loss": base_loss,
                "adapted_loss": report["language_modeling"],
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
