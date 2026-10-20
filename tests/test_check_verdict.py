from __future__ import annotations

import random

from src.app.check import CPU_MIN_RAM_GB, CPU_OK_SECONDS, DISK_CPU_GB, Spec, recommend
from src.app.verdict import short_cpu_name, shortfalls, verdict


def _say(spec: Spec, url: str = "") -> dict:
    return verdict(spec, recommend(spec), "jddd", url, random.Random(0))


def test_fast_machine_gets_local_with_nani_line():
    spec = Spec(cpu="AMD Ryzen 7 8845H w/ Radeon 780M Graphics", ram_total_gb=32, disk_free_gb=100, bandwidth_gbps=45)
    said = _say(spec)
    assert said["tier"] == "fast" and said["local"] and said["line"].startswith(("纳尼", "什么"))
    assert any("JLM-CPU.7z" in step for step in said["steps"])


def test_ok_and_web_tiers():
    ok = _say(Spec(cpu="Intel Core i5", ram_total_gb=16, disk_free_gb=50, bandwidth_gbps=20))
    assert ok["tier"] == "ok" and ok["local"]
    web = _say(Spec(ram_total_gb=4, disk_free_gb=50, bandwidth_gbps=12))
    assert web["tier"] == "web" and web["line"].startswith("呀嘞呀嘞") and not web["local"]
    reasons = shortfalls(
        Spec(ram_total_gb=4, disk_free_gb=2, bandwidth_gbps=5),
        recommend(Spec(ram_total_gb=4, disk_free_gb=2, bandwidth_gbps=5)),
        CPU_MIN_RAM_GB,
        DISK_CPU_GB,
        CPU_OK_SECONDS,
    )
    assert len(reasons) == 3


def test_mac_goes_to_web():
    said = _say(
        Spec(ram_total_gb=16, disk_free_gb=50, bandwidth_gbps=60, is_mac=True, apple_silicon=True, cpu="Apple M2")
    )
    assert said["tier"] == "mac" and not said["local"]


def test_download_link_changes_first_step():
    spec = Spec(ram_total_gb=32, disk_free_gb=100, bandwidth_gbps=45)
    assert "自动打开下载页" in _say(spec, "https://drive.google.com/x")["steps"][0]
    assert "要「JLM-CPU.7z」" in _say(spec)["steps"][0]


def test_short_cpu_names():
    assert short_cpu_name("AMD Ryzen 7 8845H w/ Radeon 780M Graphics") == "Ryzen 7 8845H"
    assert short_cpu_name("13th Gen Intel(R) Core(TM) i7-13700H") == "i7-13700H"


def test_calibrated_cpu_estimate():
    result = recommend(Spec(ram_total_gb=16, disk_free_gb=50, bandwidth_gbps=45))
    assert 5 <= result["seconds_lite"] <= 8
