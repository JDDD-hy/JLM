"""嘉豪-style verdict lines and concrete next steps for the config checker.

Only the CPU package (llama.cpp) is distributed; everything else is pointed at the web version.
"""

from __future__ import annotations

import random
import re

FAST_LINES = [
    "纳尼？！每句只要 {lite} 秒？……呵呵呵（黑化中）我竟然小瞧你了！？",
    "纳尼……{cpu}！？（推了推眼镜）有意思，这台机器配得上我。",
    "什么？{ram} GB 内存？呵呵呵……（嘴角上扬）看来你有资格和本豪平起平坐了。",
]
OK_LINES = [
    "哦？这配置也敢来见本豪？呵……内存 {ram} GB，勉强算你及格。",
    "嗯哼，有点骨气。本豪允许你慢慢来，每句等我 {lite} 秒。",
    "呵……（沉默三秒）罢了，看在你内存 {ram} GB 的份上，陪你玩。",
]
WEB_LINES = [
    "呀嘞呀嘞……就这？你还得练。",
    "呀嘞呀嘞，这配置连我的影子都追不上。没关系，本豪给你开个后门。",
    "呀嘞呀嘞～（摇头）你的电脑还在新手村，我在终点线等你。",
]
MAC_LINES = [
    "纳尼？{cpu}？……苹果电脑也想和本豪过招？呵呵呵，先用网页版陪你玩。",
    "呀嘞呀嘞，Mac 啊……本豪的本地版不伺候 Mac，你走网页版的后门吧。",
]


def short_cpu_name(name: str) -> str:
    """Strip vendor noise: 'AMD Ryzen 7 8845H w/ Radeon …' → 'Ryzen 7 8845H', '13th Gen Intel(R) Core(TM) i7-…'."""
    name = re.sub(r"\((R|TM)\)|CPU|Processor|\d+(st|nd|rd|th) Gen", "", name, flags=re.IGNORECASE)
    name = re.split(r"\s+(w/|with)\s+", name)[0]
    name = re.sub(r"^(AMD|Intel|Apple)\s+", "", name.strip())
    name = re.sub(r"^Core\s+(?=i\d|Ultra)", "", name)
    return " ".join(name.split()) or "这台电脑"


def shortfalls(spec, result: dict, cpu_min_ram: float, disk_cpu: float, ok_seconds: float) -> list[str]:
    reasons = []
    if spec.ram_total_gb < cpu_min_ram:
        reasons.append(f"内存 {spec.ram_total_gb} GB，本地版至少要 {cpu_min_ram:.0f} GB")
    if spec.disk_free_gb < disk_cpu:
        reasons.append(f"硬盘只剩 {spec.disk_free_gb} GB，本地版需要 {disk_cpu:.0f} GB")
    lite = result.get("seconds_lite")
    if lite and lite > ok_seconds:
        reasons.append(f"本地跑一句大约要 {lite:.0f} 秒，等得太久")
    if not lite:
        reasons.append("没测出内存速度，没法估算本地版快慢")
    return reasons


def local_steps(author: str, download_url: str, lite: str) -> list[str]:
    first = (
        "测完按回车会自动打开下载页（Google Drive 文件夹），下载里面的「JLM-CPU.7z」，约 2.5 GB。"
        "如果提示「无法扫描病毒」，点「仍然下载」。"
        if download_url
        else f"向 {author} 要「JLM-CPU.7z」的 Google Drive 下载链接，约 2.5 GB。"
    )
    return [
        first,
        "用 7-Zip / Bandizip / WinRAR 解压到硬盘，路径里最好不要有中文。",
        f"双击「和嘉豪聊天.bat」，加载约 15 秒，之后每句大约 {lite} 秒。",
        "输入 /help 看玩法，/topic 让嘉豪出题。",
        f"Google Drive 在国内需要科学上网；打不开就找 {author} 要网页链接。",
    ]


def web_steps(author: str, mac: bool = False) -> list[str]:
    return [
        f"向 {author} 要嘉豪的网页链接和访问口令。",
        "用 Safari 或 Chrome 打开链接就能聊，iPhone 也可以。" if mac else "用浏览器打开链接，电脑、手机都可以。",
        "不用下载任何东西，回复速度取决于服务器排队。",
    ]


def verdict(spec, result: dict, author: str, download_url: str = "", rng: random.Random | None = None) -> dict:
    rng = rng or random.Random()
    lite = result.get("seconds_lite")
    fill = {"cpu": short_cpu_name(spec.cpu), "ram": spec.ram_total_gb, "lite": f"{lite:.0f}" if lite else "十几"}
    if getattr(spec, "is_mac", False):
        return {
            "tier": "mac",
            "line": rng.choice(MAC_LINES).format(**fill),
            "title": "网页版 · 打开链接就能聊",
            "steps": web_steps(author, mac=True),
            "local": False,
        }
    tier = result.get("tier")
    if tier in {"fast", "ok"}:
        return {
            "tier": tier,
            "line": rng.choice(FAST_LINES if tier == "fast" else OK_LINES).format(**fill),
            "title": "本地版 · 离线就能聊" if tier == "fast" else "本地版 · 慢一点但能玩",
            "steps": local_steps(author, download_url, fill["lite"]),
            "local": True,
        }
    return {
        "tier": "web",
        "line": rng.choice(WEB_LINES),
        "title": "网页版 · 打开链接就能聊",
        "steps": web_steps(author),
        "local": False,
    }
