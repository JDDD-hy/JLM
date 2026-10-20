"""Inspect the existing MCP's read-only QR endpoint, without copying credential files."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from src.collection.mcp_client import McpClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qrcode", action="store_true")
    args = parser.parse_args()
    client = McpClient()
    try:
        if not args.qrcode:
            print(json.dumps(client.schemas.get("get_login_qrcode", {}), ensure_ascii=False, indent=2))
            return
        result = client.tool("get_login_qrcode", {})
        text_parts = []
        image_data = None
        for block in result.get("content", []):
            if block.get("type") == "image":
                image_data = block["data"]
            elif block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        # Some versions return the image in a JSON text block instead.
        for text in text_parts:
            try:
                value = json.loads(text)
                if isinstance(value, dict) and isinstance(value.get("img"), str):
                    image_data = value["img"].split(",", 1)[-1]
            except json.JSONDecodeError:
                pass
        if image_data:
            target = Path(".tmp/xiaohongshu-mcp-login.png").resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(base64.b64decode(image_data))
            print(json.dumps({"qrcode_path": str(target), "status_text": text_parts}, ensure_ascii=False))
        else:
            print(json.dumps({"qrcode_available": False, "status_text": text_parts}, ensure_ascii=False))
    finally:
        client.close()


if __name__ == "__main__":
    main()
