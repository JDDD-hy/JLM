"""Adapter to the installed Xiaohongshu skill CLI, never a separate crawler."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

READ_COMMANDS = {"check-login", "search-feeds", "get-feed-detail"}
SENSITIVE_KEYS = {
    "user",
    "userinfo",
    "username",
    "userid",
    "nickname",
    "avatar",
    "avatarurl",
    "xsectoken",
    "cookie",
    "cookies",
    "authorization",
    "token",
    "phone",
    "email",
}


def sanitize_response(value):
    if isinstance(value, dict):
        return {
            key: sanitize_response(item)
            for key, item in value.items()
            if key.replace("_", "").lower() not in SENSITIVE_KEYS
        }
    if isinstance(value, list):
        return [sanitize_response(item) for item in value]
    if isinstance(value, str) and value.startswith(("https://", "http://")):
        url = urlsplit(value)
        return urlunsplit((url.scheme, url.netloc.split("@")[-1], url.path, "", ""))
    return value


class SkillClient:
    def __init__(self, skill_root: str | Path):
        root = Path(skill_root).resolve()
        self.python = root / ".venv" / "Scripts" / "python.exe"
        self.cli = root / "scripts" / "cli.py"
        if not self.python.exists() or not self.cli.exists():
            raise ValueError("Existing skill CLI/runtime not found")

    def call(self, command: str, *arguments: str) -> dict:
        if command not in READ_COMMANDS:
            raise ValueError("Only read-only collection commands are permitted")
        try:
            completed = subprocess.run(
                [str(self.python), str(self.cli), command, *arguments],
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=300 if "--load-all-comments" in arguments else 100,
                env={**os.environ, "PYTHONUTF8": "1"},
            )
        except subprocess.TimeoutExpired:
            # Timeout exception messages contain the full command and paired access token.
            raise RuntimeError(f"{command}: timeout; collection stopped") from None
        if completed.returncode:
            try:
                payload = json.loads(completed.stdout)
            except json.JSONDecodeError:
                payload = {}
            error = str(payload.get("error", "")).lower() if isinstance(payload, dict) else ""
            # Only fixed hints escape this boundary; raw errors can contain credentials.
            hints = {
                "verification": ("captcha", "验证", "扫码"),
                "rate_limit": ("rate limit", "限流", "频率"),
                "access_denied": ("access denied", "forbidden", "访问拒绝"),
                "timeout": ("timeout", "timed out", "超时"),
                "bridge_connection": ("bridge", "websocket", "connection refused"),
            }
            hint = next((name for name, terms in hints.items() if any(term in error for term in terms)), "unknown")
            raise RuntimeError(f"{command}: exit {completed.returncode}; diagnostic_hint={hint}; collection stopped")
        try:
            response = json.loads(completed.stdout)
        except json.JSONDecodeError:
            raise RuntimeError(f"{command}: non-JSON response; collection stopped") from None
        if not isinstance(response, dict) or response.get("error"):
            raise RuntimeError(f"{command}: source error; collection stopped")
        return response
