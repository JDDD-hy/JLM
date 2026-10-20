from __future__ import annotations

import argparse
import shlex

from src.config import load_yaml
from src.inference.generate import generate_quotes, load_generation_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive JLM sampler")
    parser.add_argument("--adapter", default="adapters/jlm-v0.1")
    parser.add_argument("--model-config", default="configs/model.yaml")
    args = parser.parse_args()
    settings = load_yaml("configs/generation.yaml")
    model, tokenizer = load_generation_model(load_yaml(args.model_config), args.adapter)
    print("Commands: /set key value, /show, /quit. Empty input generates with no prefix.")
    while True:
        raw = input("jlm> ").strip()
        if raw == "/quit":
            break
        if raw == "/show":
            print(settings)
            continue
        if raw.startswith("/set "):
            parts = shlex.split(raw)
            if len(parts) != 3 or parts[1] not in settings:
                print("Usage: /set <existing-setting> <value>")
                continue
            current = settings[parts[1]]
            caster = type(current) if current is not None else str
            settings[parts[1]] = caster(parts[2])
            continue
        settings["prefix"] = raw
        for quote in generate_quotes(model, tokenizer, **settings):
            print(f"- {quote}")


if __name__ == "__main__":
    main()
