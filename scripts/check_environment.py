from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


class MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_ulong),
        ("memory_load", ctypes.c_ulong),
        ("total_physical", ctypes.c_ulonglong),
        ("available_physical", ctypes.c_ulonglong),
        ("total_page_file", ctypes.c_ulonglong),
        ("available_page_file", ctypes.c_ulonglong),
        ("total_virtual", ctypes.c_ulonglong),
        ("available_virtual", ctypes.c_ulonglong),
        ("available_extended_virtual", ctypes.c_ulonglong),
    ]


def system_memory() -> dict[str, float] | None:
    if os.name != "nt":
        return None
    status = MemoryStatus()
    status.length = ctypes.sizeof(MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return {
        "total_gib": round(status.total_physical / 2**30, 2),
        "available_gib": round(status.available_physical / 2**30, 2),
    }


def command_output(command: list[str]) -> dict[str, object]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
        return {"exit_code": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"exit_code": None, "error": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    disk = shutil.disk_usage(root)
    report: dict[str, object] = {
        "python": {"version": sys.version, "executable": sys.executable, "in_venv": sys.prefix != sys.base_prefix},
        "system_ram": system_memory(),
        "disk": {"path": str(root), "free_gib": round(disk.free / 2**30, 2), "total_gib": round(disk.total / 2**30, 2)},
        "git": command_output(["git", "status", "--short", "--branch"]),
    }
    failures: list[str] = []
    if sys.prefix == sys.base_prefix:
        failures.append("Expected the project venv interpreter")
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        cuda_report = {
            "torch_version": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "available": cuda_available,
            "bf16_supported": torch.cuda.is_bf16_supported() if cuda_available else False,
            "device_count": torch.cuda.device_count(),
        }
        if cuda_available:
            props = torch.cuda.get_device_properties(0)
            free_vram, _ = torch.cuda.mem_get_info()
            cuda_report |= {
                "device_name": props.name,
                "total_vram_mib": round(props.total_memory / 2**20),
                "compute_capability": f"{props.major}.{props.minor}",
                "free_vram_mib": round(free_vram / 2**20),
            }
            left = torch.randn((64, 64), device="cuda")
            right = torch.randn((64, 64), device="cuda")
            cuda_report["matmul_smoke_sum"] = float((left @ right).sum().cpu())
            import bitsandbytes as bnb

            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            sample = torch.randn(2, 16, device="cuda", dtype=dtype, requires_grad=True)
            quantized = bnb.nn.Linear4bit(
                16, 8, bias=False, quant_type="nf4", compress_statistics=True, compute_dtype=dtype
            ).cuda()
            quantized(sample).float().square().mean().backward()
            cuda_report["nf4_backward"] = sample.grad is not None
        else:
            failures.append("PyTorch CUDA is unavailable")
        report["pytorch"] = cuda_report
    except Exception as exc:
        report["pytorch"] = {"error": repr(exc)}
        failures.append(f"PyTorch check failed: {exc}")
    versions: dict[str, str] = {}
    for distribution in (
        "transformers",
        "datasets",
        "accelerate",
        "peft",
        "trl",
        "bitsandbytes",
        "huggingface-hub",
        "safetensors",
    ):
        try:
            from importlib.metadata import version

            versions[distribution] = version(distribution)
        except Exception:
            versions[distribution] = "not-installed"
            failures.append(f"Missing package: {distribution}")
    report["packages"] = versions
    report["failures"] = failures
    output = root / "logs" / "environment.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.strict and failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
