from __future__ import annotations

import importlib.util
from pathlib import Path

from src.app import llama_engine

MODULE_PATH = Path(__file__).parents[1] / "packaging" / "build_macos.py"
SPEC = importlib.util.spec_from_file_location("build_macos", MODULE_PATH)
assert SPEC and SPEC.loader
build_macos = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_macos)


def test_frozen_package_root_is_executable_directory(tmp_path: Path, monkeypatch) -> None:
    executable = tmp_path / "JLM"
    monkeypatch.setattr(llama_engine.sys, "frozen", True, raising=False)
    monkeypatch.setattr(llama_engine.sys, "executable", str(executable))

    assert llama_engine.package_root() == tmp_path


def test_verify_macos_package_requires_complete_layout(tmp_path: Path) -> None:
    package = tmp_path / "JLM-Mac"
    for path in (
        package / "JLM",
        package / "和嘉豪聊天.command",
        package / "llama" / "llama-server",
        *(package / "models" / name for name in build_macos.MODEL_NAMES),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    build_macos.verify_package(package)


def test_gpu_layers_are_forwarded_to_llama_server(tmp_path: Path, monkeypatch) -> None:
    paths = [tmp_path / name for name in ("llama-server", "base.gguf", "lora.gguf")]
    for path in paths:
        path.touch()
    commands = []

    class Process:
        def poll(self):
            return None

    class Job:
        def assign(self, _process):
            pass

        def close(self):
            pass

    monkeypatch.setattr(llama_engine, "KillOnCloseJob", Job)
    monkeypatch.setattr(
        llama_engine.subprocess,
        "Popen",
        lambda command, **_kwargs: commands.append(command) or Process(),
    )
    monkeypatch.setattr(llama_engine.LlamaEngine, "_get", lambda _self, _path: {"status": "ok"})
    engine = llama_engine.LlamaEngine(*(str(path) for path in paths), gpu_layers=99)

    engine.load(timeout=0.1)

    assert commands[0][-2:] == ["--n-gpu-layers", "99"]
