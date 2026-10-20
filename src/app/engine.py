"""Chat engine for the 嘉豪 app: model loading, candidate generation, topic-first re-ranking, session state."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_ADAPTER = "adapters/jlm-v0.1-pilot-dialogue-chat4b-v4"
DEFAULT_MODEL_CONFIG = "configs/model_chat4b_r16.yaml"


def quiet_libraries() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    warnings.filterwarnings("ignore")
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except ImportError:
        pass


@dataclass
class Reply:
    text: str
    relevance: float
    style: float
    candidates: list[dict]


@dataclass
class JiahaoEngine:
    adapter: str = DEFAULT_ADAPTER
    model_config_path: str = DEFAULT_MODEL_CONFIG
    temperature: float = 1.0
    candidate_count: int = 8
    history_turns: int = 2
    history: list[tuple[str, str]] = field(default_factory=list)
    last: Reply | None = None

    def load(self) -> None:
        quiet_libraries()
        from src.config import load_yaml
        from src.inference.generate import load_generation_model

        root = Path(__file__).resolve().parents[2]
        self.model_config = load_yaml(str(root / self.model_config_path))
        if not Path(self.model_config.get("cache_dir", "")).is_absolute():
            self.model_config["cache_dir"] = str(root / self.model_config["cache_dir"])
        self.model, self.tokenizer = load_generation_model(self.model_config, str(root / self.adapter))

    def reply(self, message: str) -> Reply:
        from scripts.chat import build_input_ids, candidates, judged_rank

        ids, stops = build_input_ids(self.tokenizer, self.model_config, self.history, message, self.history_turns)
        texts = candidates(
            self.model, self.tokenizer, ids, stops, self.candidate_count, self.temperature, None, multiline=True
        )
        ranked = judged_rank(self.model, self.tokenizer, message, texts)
        if not ranked:
            ranked = [{"text": "……", "relevance": 0.0, "style": 0.0, "swagger": 0.0, "final": 0.0}]
        best = ranked[0]
        self.history.append((message, best["text"]))
        self.last = Reply(best["text"], best["relevance"], best["swagger"], ranked)
        return self.last

    def pick(self, number: int) -> Reply:
        if not self.last or not self.history or not 1 <= number <= len(self.last.candidates):
            raise IndexError("no such candidate")
        chosen = self.last.candidates[number - 1]
        self.history[-1] = (self.history[-1][0], chosen["text"])
        self.last = Reply(chosen["text"], chosen["relevance"], chosen["swagger"], self.last.candidates)
        return self.last

    def reset(self) -> None:
        self.history.clear()
        self.last = None
