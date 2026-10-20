"""HTTP client for the existing xiaohongshu-mcp, with a read-only allowlist."""

from __future__ import annotations

import json

import httpx

READ_TOOLS = {"check_login_status", "search_feeds", "get_feed_detail", "get_login_qrcode"}


def decode_rpc(response: httpx.Response) -> dict:
    response.raise_for_status()
    if "text/event-stream" in response.headers.get("content-type", ""):
        for line in response.text.splitlines():
            if line.startswith("data:"):
                result = json.loads(line[5:].strip())
                if "result" in result or "error" in result:
                    return result
        raise RuntimeError("MCP returned no result")
    return response.json()


class McpClient:
    def __init__(self, url="http://127.0.0.1:18060/mcp"):
        self.url = url
        self.sequence = 0
        self.http = httpx.Client(timeout=90, headers={"Accept": "application/json, text/event-stream"})
        response = self.http.post(
            self.url,
            json={
                "jsonrpc": "2.0",
                "id": 0,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "jlm-readonly", "version": "0.1"},
                },
            },
        )
        result = decode_rpc(response)
        if "error" in result:
            raise RuntimeError("MCP initialization failed")
        if response.headers.get("mcp-session-id"):
            self.http.headers["Mcp-Session-Id"] = response.headers["mcp-session-id"]
        self.http.post(self.url, json={"jsonrpc": "2.0", "method": "notifications/initialized"})
        self.schemas = {item["name"]: item for item in self.rpc("tools/list").get("tools", [])}

    def rpc(self, method, params=None):
        self.sequence += 1
        payload = {"jsonrpc": "2.0", "id": self.sequence, "method": method}
        if params is not None:
            payload["params"] = params
        result = decode_rpc(self.http.post(self.url, json=payload))
        if "error" in result:
            raise RuntimeError(f"MCP {method} failed")
        return result["result"]

    def tool(self, name, arguments):
        schema = self.schemas.get(name, {})
        if name not in READ_TOOLS or schema.get("annotations", {}).get("readOnlyHint") is not True:
            raise ValueError("MCP tool is not in the verified read-only allowlist")
        result = self.rpc("tools/call", {"name": name, "arguments": arguments})
        if result.get("isError"):
            raise RuntimeError(f"MCP {name} returned a source error; collection stopped")
        return result

    def close(self):
        self.http.close()
