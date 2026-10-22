import ctypes
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from src.app.process_lifetime import stop_process


def test_stop_process_waits_for_child_exit():
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])

    stop_process(process)

    assert process.poll() is not None


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object behavior")
def test_job_kills_child_after_forced_owner_exit():
    code = """
import os, subprocess, sys
from src.app.process_lifetime import KillOnCloseJob
job = KillOnCloseJob()
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
job.assign(child)
print(child.pid, flush=True)
os._exit(0)
"""
    owner = subprocess.Popen(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        stdout=subprocess.PIPE,
        text=True,
    )
    child_pid = int(owner.stdout.readline())
    assert owner.wait(timeout=5) == 0

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = ctypes.c_void_p
    for _ in range(20):
        handle = kernel32.OpenProcess(0x00100000, False, child_pid)  # SYNCHRONIZE
        if not handle:
            break
        exited = kernel32.WaitForSingleObject(handle, 0) == 0
        kernel32.CloseHandle(handle)
        if exited:
            break
        time.sleep(0.25)
    assert not handle or exited
