"""JLM 配置检测：识别电脑配置，推荐本地显卡版 / 本地 CPU 版 / 网页版。只读，不安装任何东西。"""

from __future__ import annotations

import ctypes
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.app.branding import AUTHOR, BANNER_COLORS, GOLD, SHADE_COLOR, TITLE_ART, prepare_console
from src.app.links import CPU_DOWNLOAD_URL
from src.app.verdict import shortfalls, verdict

MODEL_GB = 2.5  # 4B model at 4-bit; every generated token streams roughly this much memory
TOKENS_PER_REPLY = 30
CPU_MIN_RAM_GB = 8.0
CPU_FAST_SECONDS = 8.0
CPU_OK_SECONDS = 20.0
DISK_CPU_GB = 5.0
# Calibrated on a Ryzen 7 8845H (45 GB/s copy bandwidth): 18 tok/s, ~6.5 s per 3-candidate judged reply.
SECONDS_LITE_AT_1TPS = 117.0
SECONDS_FULL_AT_1TPS = 280.0
SECONDS_SINGLE_AT_1TPS = 40.0


@dataclass
class Spec:
    os: str = ""
    cpu: str = ""
    cores: int = 0
    threads: int = 0
    ram_total_gb: float = 0.0
    ram_free_gb: float = 0.0
    disk_free_gb: float = 0.0
    nvidia_gpus: list[dict] = field(default_factory=list)
    other_gpus: list[str] = field(default_factory=list)
    bandwidth_gbps: float = 0.0
    is_mac: bool = False
    apple_silicon: bool = False


def _sysctl(name: str) -> str:
    return _run(["sysctl", "-n", name])


def collect_mac(spec: Spec) -> None:
    spec.is_mac = True
    spec.apple_silicon = platform.machine() == "arm64"
    spec.os = f"macOS {platform.mac_ver()[0] or platform.release()}"
    spec.cpu = _sysctl("machdep.cpu.brand_string") or ("Apple 芯片" if spec.apple_silicon else "Intel")
    spec.cores = int(_sysctl("hw.physicalcpu") or 0)
    spec.threads = int(_sysctl("hw.logicalcpu") or os.cpu_count() or 0)
    memsize = _sysctl("hw.memsize")
    spec.ram_total_gb = round(int(memsize) / 2**30, 1) if memsize.isdigit() else 0.0


def _run(command: list[str], timeout: float = 15.0) -> str:
    try:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="ignore",
            creationflags=flags,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _powershell_json(query: str) -> list[dict]:
    raw = _run(["powershell", "-NoProfile", "-Command", f"{query} | ConvertTo-Json -Compress"])
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else [data]


def memory_gb() -> tuple[float, float]:
    if os.name != "nt":
        return 0.0, 0.0

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(MemoryStatus)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return status.ullTotalPhys / 2**30, status.ullAvailPhys / 2**30


def nvidia_gpus() -> list[dict]:
    exe = shutil.which("nvidia-smi") or r"C:\Windows\System32\nvidia-smi.exe"
    if not Path(exe).exists():
        return []
    raw = _run([exe, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"])
    gpus = []
    for line in raw.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2 and parts[1].replace(".", "").isdigit():
            gpus.append(
                {
                    "name": parts[0],
                    "vram_gb": round(float(parts[1]) / 1024, 1),
                    "driver": parts[2] if len(parts) > 2 else "",
                }
            )
    return gpus


def memory_bandwidth_gbps(seconds: float = 2.0) -> float:
    """Rough read+write bandwidth with numpy copies of a buffer far larger than CPU cache."""
    try:
        import numpy as np
    except ImportError:
        return 0.0
    size = 256 * 2**20
    source = np.ones(size // 8, dtype=np.float64)
    target = np.empty_like(source)
    np.copyto(target, source)
    copies, start = 0, time.perf_counter()
    while time.perf_counter() - start < seconds:
        np.copyto(target, source)
        copies += 1
    elapsed = time.perf_counter() - start
    return round(copies * size * 2 / elapsed / 2**30, 1)


def collect(progress=None) -> Spec:
    if platform.system() == "Darwin":
        spec = Spec()
        collect_mac(spec)
        spec.disk_free_gb = round(shutil.disk_usage("/").free / 2**30, 1)
        if progress:
            progress("正在测内存带宽（约 2 秒）……")
        spec.bandwidth_gbps = memory_bandwidth_gbps()
        return spec
    version = platform.version()
    release = platform.release()
    if platform.system() == "Windows" and version.split(".")[-1].isdigit() and int(version.split(".")[-1]) >= 22000:
        release = "11"  # platform.release() still reports "10" on Windows 11
    spec = Spec(os=f"{platform.system()} {release} ({version})")
    cpus = _powershell_json(
        "Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors"
    )
    if cpus:
        spec.cpu = str(cpus[0].get("Name", "")).strip()
        spec.cores = int(sum(int(c.get("NumberOfCores") or 0) for c in cpus))
        spec.threads = int(sum(int(c.get("NumberOfLogicalProcessors") or 0) for c in cpus))
    spec.cpu = spec.cpu or platform.processor() or "未知"
    spec.threads = spec.threads or (os.cpu_count() or 0)
    total, free = memory_gb()
    spec.ram_total_gb, spec.ram_free_gb = round(total, 1), round(free, 1)
    spec.disk_free_gb = round(shutil.disk_usage(Path(sys.executable).anchor or ".").free / 2**30, 1)
    if progress:
        progress("正在检测显卡……")
    spec.nvidia_gpus = nvidia_gpus()
    videos = _powershell_json("Get-CimInstance Win32_VideoController | Select-Object Name")
    nvidia_names = {g["name"] for g in spec.nvidia_gpus}
    spec.other_gpus = [
        str(v.get("Name", "")).strip() for v in videos if v.get("Name") and v.get("Name") not in nvidia_names
    ]
    if progress:
        progress("正在测内存带宽（约 2 秒）……")
    spec.bandwidth_gbps = memory_bandwidth_gbps()
    return spec


def cpu_tokens_per_second(spec: Spec) -> float:
    # Token generation is memory-bound; llama.cpp reaches roughly half the measured copy bandwidth.
    return round(spec.bandwidth_gbps / MODEL_GB, 1) if spec.bandwidth_gbps else 0.0


def recommend(spec: Spec) -> dict:
    """Only the CPU package is distributed: fast/ok machines run it locally, everything else uses the web."""
    best_gpu = max(spec.nvidia_gpus, key=lambda g: g["vram_gb"], default=None)
    tps = cpu_tokens_per_second(spec)
    single = SECONDS_SINGLE_AT_1TPS / tps if tps else None
    lite = SECONDS_LITE_AT_1TPS / tps if tps else None
    full = SECONDS_FULL_AT_1TPS / tps if tps else None
    fits = spec.ram_total_gb >= CPU_MIN_RAM_GB and spec.disk_free_gb >= DISK_CPU_GB and bool(lite)
    if spec.is_mac:
        tier, choice, reason = "mac", "网页版", "Mac 没有本地版，打开网页链接就能聊。"
    elif fits and lite <= CPU_FAST_SECONDS:
        tier, choice, reason = "fast", "本地版", f"内存和速度都够，每句约 {lite:.0f} 秒。"
    elif fits and lite <= CPU_OK_SECONDS:
        tier, choice, reason = "ok", "本地版", f"能跑，但每句要等约 {lite:.0f} 秒。"
    else:
        tier, choice, reason = "web", "网页版", "本地跑会比较吃力，直接打开网页链接聊天最省心。"
    return {
        "tier": tier,
        "choice": choice,
        "reason": reason,
        "best_gpu": best_gpu,
        "cpu_tokens_per_second": tps,
        "seconds_single": single,
        "seconds_lite": lite,
        "seconds_full": full,
    }


def logo() -> Text:
    text = Text(no_wrap=True, overflow="crop")
    for row, line in enumerate(TITLE_ART):
        for char in line:
            text.append(char, style=SHADE_COLOR if char == "▒" else f"bold {BANNER_COLORS[row % len(BANNER_COLORS)]}")
        text.append("\n")
    return text


def speak(console: Console, line: str) -> None:
    """嘉豪 delivers the verdict, typed out character by character."""
    from rich.live import Live

    shown = Text(style="bold #FFF3BF")
    with Live(console=console, refresh_per_second=40) as live:
        for char in line:
            shown.append(char)
            live.update(Panel(shown, title="[bold]嘉豪[/]", title_align="left", border_style=GOLD, padding=(1, 2)))
            time.sleep(0.045 if char in "，。！？…～" else 0.03)


def render(console: Console, spec: Spec, result: dict) -> dict:
    said = verdict(spec, result, AUTHOR, CPU_DOWNLOAD_URL)
    speak(console, said["line"])
    reasons = shortfalls(spec, result, CPU_MIN_RAM_GB, DISK_CPU_GB, CPU_OK_SECONDS)
    steps = Text()
    for number, step in enumerate(said["steps"], start=1):
        steps.append(f"{number}. ", style=f"bold {GOLD}")
        steps.append(step + "\n")
    if said["tier"] in {"cpu", "web"} and reasons:
        steps.append("\n为什么不是更快的版本：\n", style="dim")
        for reason in reasons:
            steps.append(f"  · {reason}\n", style="dim")
    steps.rstrip()
    console.print(Panel(steps, title=f"[bold]解决方案 · {said['title']}[/]", border_style="#69DB7C", padding=(1, 2)))

    table = Table(title="检测到的配置", border_style="dim", show_header=False, title_style="dim")
    table.add_column(style="dim")
    table.add_column(style="dim")
    table.add_row("系统", spec.os)
    table.add_row("CPU", f"{spec.cpu}（{spec.cores} 核 / {spec.threads} 线程）")
    table.add_row("内存", f"{spec.ram_total_gb} GB（可用 {spec.ram_free_gb} GB）")
    table.add_row("硬盘剩余", f"{spec.disk_free_gb} GB")
    gpus = [f"{g['name']}（{g['vram_gb']} GB 显存）" for g in spec.nvidia_gpus] or ["未检测到 NVIDIA 显卡"]
    table.add_row("NVIDIA 显卡", "\n".join(gpus))
    if spec.other_gpus:
        table.add_row("其他显卡", "\n".join(spec.other_gpus))
    if spec.bandwidth_gbps:
        table.add_row("内存带宽", f"{spec.bandwidth_gbps} GB/s（实测）")
    console.print(table)
    return said


def save_report(spec: Spec, result: dict, said: dict | None = None) -> Path:
    path = Path.cwd() / f"JLM配置检测-{datetime.now():%Y%m%d-%H%M%S}.txt"
    lines = [
        f"JLM 配置检测报告  {datetime.now():%Y-%m-%d %H:%M}",
        f"推荐：{result['choice']} —— {result['reason']}",
        f"嘉豪说：{said['line']}" if said else "",
        "",
        json.dumps({"spec": asdict(spec), "result": result}, ensure_ascii=False, indent=2),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    prepare_console()
    console = Console()
    console.print(
        Panel(
            Group(
                logo(),
                Text.assemble(
                    ("Jiahao Language Model · 配置检测", f"bold {GOLD}"),
                    ("   by ", "dim"),
                    (AUTHOR, "bold italic #FF922B"),
                ),
            ),
            border_style=GOLD,
            padding=(1, 2),
        )
    )
    with console.status(f"[{GOLD}]正在检测电脑配置……[/]", spinner="dots12", spinner_style=GOLD) as status:
        spec = collect(lambda message: status.update(f"[{GOLD}]{message}[/]"))
    result = recommend(spec)
    said = render(console, spec, result)
    report = save_report(spec, result, said)
    console.print(f"\n[dim]检测报告已保存：{report}\n可以把这个文件发给 {AUTHOR}。[/]")
    if not getattr(sys, "frozen", False):
        return
    if said.get("local") and CPU_DOWNLOAD_URL:
        if input("\n按回车打开下载页面（输入 n 跳过）……").strip().lower() != "n":
            import webbrowser

            webbrowser.open(CPU_DOWNLOAD_URL)
            print("已在浏览器打开下载页面。")
    input("\n按回车键退出……")


if __name__ == "__main__":
    main()
