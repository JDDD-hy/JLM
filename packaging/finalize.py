"""Refresh the built packages with a chosen adapter and the latest app code, then make shareable 7z archives.

GPU package: copies the PEFT adapter. CPU package: converts the adapter to a LoRA GGUF for llama.cpp.
Archives are stored (no compression — model weights don't shrink) and split under netdisk file-size limits.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
SEVEN_ZIP = ROOT / "tools" / "7zip" / "7zr.exe"
APP_FILES_GPU = ["src/__init__.py", "src/config.py", "src/app", "src/data", "src/inference", "scripts/chat.py"]
APP_FILES_CPU = [
    "src/__init__.py",
    "src/app/__init__.py",
    "src/app/branding.py",
    "src/app/tui.py",
    "src/app/engine.py",
    "src/app/llama_engine.py",
    "src/data/__init__.py",
    "src/data/emoji.py",
    "src/inference/__init__.py",
    "src/inference/selection.py",
]


def refresh_code(package: Path, files: list[str]) -> None:
    for item in files:
        source, target = ROOT / item, package / "app" / item
        if source.is_dir():
            shutil.rmtree(target, ignore_errors=True)
            shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "static"))
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def refresh_gpu(adapter: Path) -> None:
    package = DIST / "JLM-GPU"
    refresh_code(package, APP_FILES_GPU)
    adapters = package / "app" / "adapters"
    shutil.rmtree(adapters, ignore_errors=True)
    (adapters / adapter.name).mkdir(parents=True)
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        shutil.copy2(adapter / name, adapters / adapter.name / name)


def refresh_cpu(adapter: Path) -> None:
    package = DIST / "JLM-CPU"
    refresh_code(package, APP_FILES_CPU)
    snapshot = next((ROOT / ".cache" / "huggingface" / "models--Qwen--Qwen3-4B" / "snapshots").iterdir())
    env = {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        "PYTHONPATH": os.pathsep.join([str(ROOT / "tools/llamacpp/src/gguf-py"), str(ROOT / "tools/llamacpp/src")]),
    }
    out = ROOT / "build" / "gguf" / "jiahao-lora-f16.gguf"
    subprocess.run(
        [
            str(ROOT / ".venv/Scripts/python.exe"),
            str(ROOT / "tools/llamacpp/src/convert_lora_to_gguf.py"),
            "--base",
            str(snapshot),
            "--outtype",
            "f16",
            "--outfile",
            str(out),
            str(adapter),
        ],
        check=True,
        env=env,
        stdout=subprocess.DEVNULL,
    )
    shutil.copy2(out, package / "models" / "jiahao-lora-f16.gguf")


def archive(name: str, volume: str | None) -> list[Path]:
    target = DIST / "share" / f"{name}.7z"
    target.parent.mkdir(parents=True, exist_ok=True)
    for old in target.parent.glob(f"{name}.7z*"):
        old.unlink()
    command = [str(SEVEN_ZIP), "a", "-mx=0", "-bso0", "-bsp0", str(target), str(DIST / name)]
    if volume:
        command.insert(2, f"-v{volume}")
    subprocess.run(command, check=True)
    return sorted(target.parent.glob(f"{name}.7z*"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter", required=True, help="Adapter folder, e.g. adapters/jlm-v0.1-pilot-dialogue-chat4b-v4"
    )
    parser.add_argument("--skip-archive", action="store_true")
    parser.add_argument("--gpu", action="store_true", help="Also refresh/archive the GPU package (build it first)")
    args = parser.parse_args()
    adapter = (ROOT / args.adapter).resolve()
    if args.gpu:
        if not (DIST / "JLM-GPU").exists():
            raise SystemExit("dist/JLM-GPU 不存在，先运行 packaging/build_packages.py gpu")
        refresh_gpu(adapter)
    refresh_cpu(adapter)
    print(f"packages refreshed with {adapter.name}")
    if args.skip_archive:
        return
    targets = ([("JLM-GPU", "3900m")] if args.gpu else []) + [("JLM-CPU", None)]
    for name, volume in targets:
        for part in archive(name, volume):
            print(f"{part.name}  {part.stat().st_size / 2**30:.2f} GB")


if __name__ == "__main__":
    main()
