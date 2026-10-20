"""Local-only JSON inference service using the Python standard library."""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from src.config import load_yaml
from src.inference.generate import generate_quotes, load_generation_model


def make_handler(model, tokenizer, adapter):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status, payload):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.send_json(
                200 if self.path == "/health" else 404,
                {"ready": True, "adapter": adapter} if self.path == "/health" else {"error": "not found"},
            )

        def do_POST(self):
            if self.path != "/generate":
                return self.send_json(404, {"error": "not found"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 8192:
                    raise ValueError("JSON body must be 1..8192 bytes")
                settings = json.loads(self.rfile.read(size))
                if not isinstance(settings, dict):
                    raise ValueError("Expected JSON object")
                if not 1 <= settings.get("num_return_sequences", 1) <= 8:
                    raise ValueError("num_return_sequences must be 1..8")
                if not 1 <= settings.get("max_new_tokens", 96) <= 128:
                    raise ValueError("max_new_tokens must be 1..128")
                if len(settings.get("prefix", "")) > 512:
                    raise ValueError("prefix too long")
                settings.setdefault("num_return_sequences", 1)
                with lock:
                    quotes = generate_quotes(model, tokenizer, **settings)
                self.send_json(200, {"quotes": quotes})
            except (ValueError, TypeError) as exc:
                self.send_json(400, {"error": str(exc)})

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    model, tokenizer = load_generation_model(load_yaml("configs/model.yaml"), args.adapter)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(model, tokenizer, args.adapter))
    print(f"JLM listening on http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
