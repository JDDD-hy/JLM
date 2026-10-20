"""Read-only discovery of the existing local xiaohongshu-mcp, without editing login state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx


def decode_response(response: httpx.Response) -> dict:
    response.raise_for_status()
    if "text/event-stream" in response.headers.get("content-type", ""):
        for line in response.text.splitlines():
            if line.startswith("data:"):
                value = json.loads(line[5:].strip())
                if "result" in value or "error" in value:
                    return value
        raise ValueError("MCP event stream contains no result")
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18060/mcp")
    args = parser.parse_args()
    report = {"endpoint": args.url, "status": "unavailable", "cookies_modified": False}
    try:
        with httpx.Client(timeout=90, headers={"Accept": "application/json, text/event-stream"}) as client:
            response = client.post(
                args.url,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "jlm-readonly-check", "version": "0.1"},
                    },
                },
            )
            initialized = decode_response(response)
            if "error" in initialized:
                raise ValueError(initialized["error"])
            session = response.headers.get("mcp-session-id")
            if session:
                client.headers["Mcp-Session-Id"] = session
            client.post(args.url, json={"jsonrpc": "2.0", "method": "notifications/initialized"})
            inventory = decode_response(client.post(args.url, json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"}))
            if "error" in inventory:
                raise ValueError(inventory["error"])
            read_names = {"search_feeds", "get_feed_detail", "check_login_status"}
            readable = [tool for tool in inventory.get("result", {}).get("tools", []) if tool["name"] in read_names]
            report.update(status="connected", read_only_tools=readable)
            status_tool = next((tool for tool in readable if tool["name"] == "check_login_status"), None)
            if status_tool and not status_tool.get("inputSchema", {}).get("required"):
                status = decode_response(
                    client.post(
                        args.url,
                        json={
                            "jsonrpc": "2.0",
                            "id": 3,
                            "method": "tools/call",
                            "params": {"name": "check_login_status", "arguments": {}},
                        },
                    )
                )
                report["login_check"] = status.get("result", status.get("error"))
                report["login_check_completed"] = "result" in status and not status["result"].get("isError", False)
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        report["error"] = str(exc)
    output = Path("logs/mcp_environment.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "connected" or report.get("error") or not report.get("login_check_completed"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
