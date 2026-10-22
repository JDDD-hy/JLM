"""Build the self-contained Apple Silicon JLM package on macOS."""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "packaging" / "jlm_macos_entry.py"
MODEL_NAMES = ("qwen3-4b-q4_k_m.gguf", "jiahao-lora-f16.gguf")
PYINSTALLER_EXCLUDES = ("datasets", "peft", "torch", "transformers", "trl")
LAUNCHER = """#!/bin/zsh
cd -- "$(dirname -- "$0")"
./JLM
status=$?
echo
read -k 1 "?按任意键关闭……"
echo
exit $status
"""
README = """JLM 嘉豪语言模型 · Mac 本地版   by jddd
======================================

需要：Apple Silicon Mac（M1/M2/M3/M4），内存至少 8 GB，建议 16 GB。

使用：
1. 保持整个 JLM-Mac 文件夹完整。
2. 双击“和嘉豪聊天.command”。
3. 首次启动若被 macOS 拦截，请在“系统设置 → 隐私与安全性”中确认打开。

模型完全在本机运行；聊天记录保存在“文稿/JLM聊天记录”。
"""


def verify_package(package: Path) -> None:
    required = [
        package / "JLM",
        package / "和嘉豪聊天.command",
        package / "llama" / "llama-server",
        *(package / "models" / name for name in MODEL_NAMES),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("Mac 包不完整：" + ", ".join(missing))


def build(package: Path, llama_bin: Path, gguf_dir: Path) -> None:
    if sys.platform != "darwin" or platform.machine() not in {"arm64", "aarch64"}:
        raise SystemExit("Mac 包必须在 Apple Silicon Mac 上构建。")
    server = llama_bin / "llama-server"
    for path in (server, *(gguf_dir / name for name in MODEL_NAMES)):
        if not path.is_file():
            raise SystemExit(f"缺少构建文件：{path}")
    if package.exists():
        shutil.rmtree(package)
    package.mkdir(parents=True)
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--target-arch",
        "arm64",
        "--name",
        "JLM",
        "--paths",
        str(ROOT),
        "--distpath",
        str(package),
        "--workpath",
        str(ROOT / "build" / "JLM-macos"),
        "--specpath",
        str(ROOT / "build"),
        str(ENTRY),
    ]
    for module in PYINSTALLER_EXCLUDES:
        command[3:3] = ["--exclude-module", module]
    subprocess.run(
        command,
        check=True,
        cwd=ROOT,
    )
    llama = package / "llama"
    llama.mkdir()
    shutil.copy2(server, llama / server.name)
    for library in llama_bin.glob("*.dylib"):
        shutil.copy2(library, llama / library.name)
    models = package / "models"
    models.mkdir()
    for name in MODEL_NAMES:
        shutil.copy2(gguf_dir / name, models / name)
    launcher = package / "和嘉豪聊天.command"
    launcher.write_text(LAUNCHER, encoding="utf-8", newline="\n")
    launcher.chmod(0o755)
    (package / "使用说明.txt").write_text(README, encoding="utf-8")
    verify_package(package)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-bin", required=True, type=Path, help="llama.cpp 的 build/bin 目录")
    parser.add_argument("--gguf-dir", type=Path, default=ROOT / "build" / "gguf")
    parser.add_argument("--out", type=Path, default=ROOT / "dist" / "JLM-Mac")
    args = parser.parse_args()
    build(args.out.resolve(), args.llama_bin.resolve(), args.gguf_dir.resolve())
    print(args.out.resolve())


if __name__ == "__main__":
    main()
