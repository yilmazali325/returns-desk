#!/Users/ali/Desktop/Projects/retail-returns-desk/.venv/bin/python
"""Stand-in for llama-server (NOT llama.cpp): answers --version, /health, warm-up, and stalls on
real chat calls so an adapter run can be interrupted mid-case. Used only by the C4 orphan check."""
import json, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

if "--version" in sys.argv:
    print("version: 0.7.0 (build 11669, commit 620d2225c)")
    sys.exit(0)
port = int(sys.argv[sys.argv.index("--port") + 1])

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        if self.path.startswith("/health"): self._send(200, {"status": "ok"})
        elif self.path.startswith("/v1/models"): self._send(200, {"data": [{"id": "stub"}]})
        else: self._send(404, {})
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); body = json.loads(self.rfile.read(n) or b"{}")
        if body.get("max_tokens") == 1:
            self._send(200, {"choices": [{"message": {"content": "p"}}]}); return
        time.sleep(15)
        self._send(500, {"error": "stub"})

ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
