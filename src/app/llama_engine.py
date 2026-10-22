"""llama.cpp backend for the 嘉豪 app (CPU build) — standard library only.

One llama-server process holds the 4-bit base model plus the 嘉豪 LoRA (loaded unapplied). Each request
sets the LoRA scale: 1.0 to speak as 嘉豪, 0.0 to judge with the plain base model — the same split the GPU
build gets from PEFT's disable_adapter().
"""

from __future__ import annotations

import json
import os
import random
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from src.app.process_lifetime import KillOnCloseJob, stop_process
from src.inference.selection import RELEVANCE_TEMPLATE, STYLE_TEMPLATE, clean, qwen_chat_prompt, rank, score

MISSING_LOGPROB = -30.0


@dataclass
class Reply:
    text: str
    relevance: float
    style: float
    candidates: list[dict]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@dataclass
class LlamaEngine:
    server_exe: str
    model_path: str
    lora_path: str
    temperature: float = 1.0
    candidate_count: int = 3
    history_turns: int = 2
    threads: int = 0
    context: int = 2048
    gpu_layers: int = 0
    history: list[tuple[str, str]] = field(default_factory=list)
    last: Reply | None = None
    process: subprocess.Popen | None = None
    job: KillOnCloseJob | None = field(default=None, init=False, repr=False)
    port: int = 0

    # ---------- server ----------
    def load(self, timeout: float = 300.0) -> None:
        self.server_exe, self.model_path, self.lora_path = (
            str(Path(p).resolve()) for p in (self.server_exe, self.model_path, self.lora_path)
        )
        for path in (self.server_exe, self.model_path, self.lora_path):
            if not Path(path).exists():
                raise FileNotFoundError(f"找不到文件：{path}")
        self.port = free_port()
        threads = self.threads or max(1, (os.cpu_count() or 4) // 2)
        command = [
            self.server_exe,
            "-m",
            self.model_path,
            "--lora",
            self.lora_path,
            "--lora-init-without-apply",
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "-c",
            str(self.context),
            "-np",
            "1",
            "-t",
            str(threads),
            "--log-disable",
        ]
        if self.gpu_layers:
            command += ["--n-gpu-layers", str(self.gpu_layers)]
        self.job = KillOnCloseJob()
        try:
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            self.job.assign(self.process)
            deadline = time.time() + timeout
            while time.time() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError("llama-server 启动失败，可能是内存不足或文件损坏。")
                try:
                    if self._get("/health").get("status") == "ok":
                        return
                except (OSError, ValueError):
                    pass
                time.sleep(1)
            raise TimeoutError("模型加载超时。")
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        stop_process(self.process)
        self.process = None
        if self.job:
            self.job.close()
            self.job = None

    def _get(self, path: str) -> dict:
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=5) as response:
            return json.loads(response.read())

    def _post(self, path: str, body: dict, timeout: float = 300.0) -> dict:
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())

    # ---------- generation and judging ----------
    def _complete(self, prompt: str, lora_scale: float, **options) -> dict:
        body = {"prompt": prompt, "lora": [{"id": 0, "scale": lora_scale}], "cache_prompt": True, **options}
        return self._post("/completion", body)

    def candidates(self, message: str) -> list[str]:
        prompt = qwen_chat_prompt(message, self.history[-self.history_turns :] if self.history_turns else [])
        complete, truncated = [], []
        for _ in range(self.candidate_count):
            result = self._complete(
                prompt,
                1.0,
                n_predict=96,
                temperature=self.temperature,
                top_p=0.9,
                repeat_penalty=1.1,
                stop=["<|im_end|>"],
                seed=random.randrange(2**31),
            )
            text = clean(result.get("content", ""), multiline=True)
            if text:
                (truncated if result.get("stop_type") == "limit" or result.get("stopped_limit") else complete).append(
                    text
                )
        return sorted(set(complete or truncated), key=score, reverse=True)

    def yes_margin(self, question: str) -> float:
        """log P(是) − log P(否) for the first answer token, from the plain base model (LoRA scale 0)."""
        result = self._complete(qwen_chat_prompt(question, system=None), 0.0, n_predict=1, n_probs=40, temperature=0)
        probs = (result.get("completion_probabilities") or [{}])[0]
        tops = probs.get("top_logprobs") or probs.get("probs") or []
        logprob = {}
        for item in tops:
            token = item.get("token") or item.get("tok_str")
            value = item.get("logprob")
            if value is None and item.get("prob") is not None:
                value = float("-inf") if item["prob"] <= 0 else __import__("math").log(item["prob"])
            if token is not None and value is not None:
                logprob.setdefault(token.strip(), value)
        return logprob.get("是", MISSING_LOGPROB) - logprob.get("否", MISSING_LOGPROB)

    def judged(self, message: str, texts: list[str]) -> list[dict]:
        from src.inference.selection import sigmoid

        relevance = [sigmoid(self.yes_margin(RELEVANCE_TEMPLATE.format(prompt=message, reply=t))) for t in texts]
        margins = [self.yes_margin(STYLE_TEMPLATE.format(reply=t)) for t in texts]
        return rank(texts, relevance, margins)

    # ---------- engine interface used by the TUI ----------
    def reply(self, message: str) -> Reply:
        ranked = self.judged(message, self.candidates(message)) or [
            {"text": "……", "relevance": 0.0, "style": 0.0, "swagger": 0.0, "final": 0.0}
        ]
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


def package_root() -> Path:
    """External models and llama-server live beside the frozen macOS executable."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[3]


def main() -> None:
    """Entry point of the CPU package: paths are relative to the package root (parent of app/)."""
    import argparse

    from src.app.branding import prepare_console
    from src.app.tui import JiahaoApp

    prepare_console()
    root = package_root()
    server_name = "llama-server.exe" if os.name == "nt" else "llama-server"
    parser = argparse.ArgumentParser(description="Jiahao Language Model (CPU)")
    parser.add_argument("--server", default=str(root / "llama" / server_name))
    parser.add_argument("--model", default=str(root / "models" / "qwen3-4b-q4_k_m.gguf"))
    parser.add_argument("--lora", default=str(root / "models" / "jiahao-lora-f16.gguf"))
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--gpu-layers", type=int, default=99 if sys.platform == "darwin" else 0)
    args = parser.parse_args()
    engine = LlamaEngine(
        args.server,
        args.model,
        args.lora,
        temperature=args.temperature,
        candidate_count=args.candidates,
        gpu_layers=args.gpu_layers,
    )
    output = Path.home() / "Documents" / "JLM聊天记录"
    try:
        JiahaoApp(engine, output_dir=str(output)).run()
    finally:
        engine.close()


if __name__ == "__main__":
    main()
