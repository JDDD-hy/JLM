"""Assemble the portable Windows packages for friends: JLM-GPU (PyTorch + CUDA) and JLM-CPU (llama.cpp).

Each package is a self-contained folder: embeddable Python, the app code, the model, and a double-click .bat.
Nothing is installed on the recipient's machine.
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EMBED_ZIP = ROOT / "build" / "python-3.11.9-embed-amd64.zip"
SITE = ROOT / ".venv" / "Lib" / "site-packages"
ADAPTER = ROOT / "adapters" / "jlm-v0.1-pilot-dialogue-chat4b-v4"
HF_MODEL = ROOT / ".cache" / "huggingface" / "models--Qwen--Qwen3-4B"
VC_RUNTIME = [
    "msvcp140.dll",
    "msvcp140_1.dll",
    "msvcp140_2.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "concrt140.dll",
]

# Large or development-only packages the chat app never imports.
GPU_EXCLUDE = {
    "pyarrow",
    "pandas",
    "sklearn",
    "scikit_learn",
    "datasets",
    "trl",
    "pytest",
    "_pytest",
    "pluggy",
    "iniconfig",
    "ruff",
    "PyInstaller",
    "pyinstaller",
    "pyinstaller_hooks_contrib",
    "altgraph",
    "pefile",
    "fastapi",
    "starlette",
    "uvicorn",
    "pyfiglet",
    "lxml",
    "pip",
    "sentencepiece",
    "evaluate",
    "multiprocess",
    "dill",
    "xxhash",
    "aiohttp",
    "frozenlist",
    "multidict",
    "yarl",
    "aiosignal",
    "propcache",
    "hf_xet",
}
CPU_PACKAGES = {"rich", "pygments", "markdown_it", "mdurl"}

GPU_BAT = r"""@echo off
chcp 65001 >nul
title Jiahao Language Model (GPU)
cd /d "%~dp0app"
set PYTHONIOENCODING=utf-8
set PYTHONWARNINGS=ignore
set TRANSFORMERS_VERBOSITY=error
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set TOKENIZERS_PARALLELISM=false
"%~dp0python\python.exe" -m src.app %*
echo.
pause
"""
CPU_BAT = r"""@echo off
chcp 65001 >nul
title Jiahao Language Model (CPU)
cd /d "%~dp0app"
set PYTHONIOENCODING=utf-8
"%~dp0python\python.exe" -m src.app.llama_engine %*
echo.
pause
"""
README_GPU = """JLM 嘉豪语言模型 · 本地显卡版   by jddd
====================================

【需要】
- Windows 10/11 64 位
- NVIDIA 显卡，显存 6 GB 以上（RTX 20 系或更新）
- 显卡驱动 570 以上（太旧请到 NVIDIA 官网更新）
- 硬盘剩余 20 GB

【怎么用】
1. 把整个文件夹解压到硬盘（路径里最好不要有中文）。
2. 双击「和嘉豪聊天.bat」。
3. 第一次加载约 30–60 秒，看到「嘉豪已就位」就能聊。
4. 输入 /help 看玩法，/topic 让嘉豪出题，/quit 退出看战绩。

【说明】
- 完全离线运行，不联网，不安装任何东西。删掉文件夹即卸载。
- 聊天记录保存在 app\\outputs\\chats 里。
- 嘉豪是网络梗角色扮演，回复由 AI 生成，可能夸张冒犯，仅供娱乐。
"""
README_CPU = """JLM 嘉豪语言模型 · 本地 CPU 版   by jddd
=====================================

【需要】
- Windows 10/11 64 位
- 内存 8 GB 以上（推荐 16 GB）
- 硬盘剩余 4 GB
- 不需要显卡

【怎么用】
1. 把整个文件夹解压到硬盘。
2. 双击「和嘉豪聊天.bat」。
3. 加载约 10–20 秒，看到「嘉豪已就位」就能聊。
4. 每句要等十几秒到半分钟（CPU 在努力思考），聊天前关掉其他大程序会快一些。
5. 输入 /help 看玩法，/n 2 可以减少候选让回复更快。

【说明】
- 完全离线运行，不联网，不安装任何东西。删掉文件夹即卸载。
- 聊天记录保存在「文档\\JLM聊天记录」里。
- 嘉豪是网络梗角色扮演，回复由 AI 生成，可能夸张冒犯，仅供娱乐。
"""


def make_python(dest: Path, include: set[str] | None, exclude: set[str]) -> None:
    """Embeddable Python + selected site-packages; the ._pth makes app/ and site-packages importable."""
    python = dest / "python"
    zipfile.ZipFile(EMBED_ZIP).extractall(python)
    pth = next(python.glob("python3*._pth"))
    pth.write_text("python311.zip\n.\nLib\\site-packages\n..\\app\nimport site\n", encoding="utf-8")
    target = python / "Lib" / "site-packages"
    target.mkdir(parents=True)
    for entry in SITE.iterdir():
        base = entry.name.split("-")[0].split(".")[0]
        if entry.name.startswith(("_distutils_hack", "distutils-precedence", "__pycache__", "~")):
            continue
        if include is not None and base.lower() not in {p.lower() for p in include}:
            continue
        if base in exclude or base.lower() in {p.lower() for p in exclude}:
            continue
        if entry.is_dir():
            shutil.copytree(entry, target / entry.name, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(entry, target / entry.name)
    for dll in VC_RUNTIME:
        system = Path("C:/Windows/System32") / dll
        if system.exists() and not (python / dll).exists():
            shutil.copy2(system, python / dll)


def copy_app(dest: Path, modules: list[str]) -> Path:
    app = dest / "app"
    for module in modules:
        source = ROOT / module
        if source.is_dir():
            shutil.copytree(source, app / module, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "static"))
        else:
            (app / module).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, app / module)
    return app


def build_gpu(out: Path) -> Path:
    dest = out / "JLM-GPU"
    make_python(dest, None, GPU_EXCLUDE)
    app = copy_app(
        dest,
        [
            "src/__init__.py",
            "src/config.py",
            "src/app",
            "src/data",
            "src/inference",

            "scripts/chat.py",
            "configs/model_chat4b_r16.yaml",
        ],
    )
    adapter = app / "adapters" / ADAPTER.name
    adapter.mkdir(parents=True)
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        shutil.copy2(ADAPTER / name, adapter / name)
    shutil.copytree(HF_MODEL, app / ".cache" / "huggingface" / HF_MODEL.name)
    (dest / "和嘉豪聊天.bat").write_text(GPU_BAT, encoding="utf-8")
    (dest / "使用说明.txt").write_text(README_GPU, encoding="utf-8-sig")
    return dest


def build_cpu(out: Path, gguf_dir: Path) -> Path:
    dest = out / "JLM-CPU"
    make_python(dest, CPU_PACKAGES, set())
    copy_app(
        dest,
        [
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
        ],
    )
    llama = dest / "llama"
    llama.mkdir()
    for item in (ROOT / "tools" / "llamacpp" / "bin").iterdir():
        if item.suffix.lower() == ".dll" or item.name in {"llama-server.exe", "LICENSE-LLVM-OpenMP"}:
            shutil.copy2(item, llama / item.name)
    models = dest / "models"
    models.mkdir()
    for name in ("qwen3-4b-q4_k_m.gguf", "jiahao-lora-f16.gguf"):
        shutil.copy2(gguf_dir / name, models / name)
    (dest / "和嘉豪聊天.bat").write_text(CPU_BAT, encoding="utf-8")
    (dest / "使用说明.txt").write_text(README_CPU, encoding="utf-8-sig")
    return dest


def folder_size_gb(path: Path) -> float:
    return round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 2**30, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=["gpu", "cpu"])
    parser.add_argument("--out", default=str(ROOT / "dist"))
    parser.add_argument("--gguf-dir", default=str(ROOT / "build" / "gguf"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    existing = out / ("JLM-GPU" if args.target == "gpu" else "JLM-CPU")
    if existing.exists():
        shutil.rmtree(existing)
    dest = build_gpu(out) if args.target == "gpu" else build_cpu(out, Path(args.gguf_dir))
    print(f"{dest}  {folder_size_gb(dest)} GB")


if __name__ == "__main__":
    main()
