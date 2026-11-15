"""Local browser entry point for ad review and the organizer CSV."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
from uuid import uuid4

from .fmt import ALLOWED

PAGE = Path(__file__).with_name("final_demo.html")
DEMO_CSV = PAGE.parent / "examples" / "example.csv"
MAX_INPUT_BYTES = 4 * 1024 * 1024


def serve(port, out, workers, process, run_batch, classification_error):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, body, media_type, status=200, filename=None):
            self.send_response(status)
            self.send_header("Content-Type", media_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if filename:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(body)

        def _json(self, value, status=200):
            self._send(json.dumps(value, ensure_ascii=False).encode(), "application/json; charset=utf-8", status)

        def _local_host(self):
            port = self.server.server_port
            return self.headers.get("Host") in {f"127.0.0.1:{port}", f"localhost:{port}"}

        def do_GET(self):
            if not self._local_host():
                self.send_error(404)
            elif self.path in {"/", "/demo"}:
                self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/demo.csv":
                self._send(DEMO_CSV.read_bytes(), "text/csv; charset=utf-8")
            else:
                self.send_error(404)

        def do_POST(self):
            media = self.headers.get_content_type()
            expected = "/drafts" if media == "application/json" else "/batch" if media == "text/csv" else None
            if not self._local_host() or self.path != expected:
                self.send_error(404)
                return
            try:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= MAX_INPUT_BYTES:
                    self._json({"error": "Размер запроса должен быть от 1 байта до 4 МБ."}, 413)
                    return
                body = self.rfile.read(size)
                job = out / uuid4().hex
                if self.path == "/drafts":
                    result, metrics = process(json.loads(body), job / "audit")
                    self._json({**result, "metrics": metrics})
                else:
                    self._batch(body, job)
            except ValueError as exc:
                self._json({"error": str(exc)}, 400)
            except classification_error as exc:
                self._json({"error": str(exc)}, 502)
            except OSError:
                self._json({"error": "Не удалось сохранить результат. Проверьте доступ к папке запуска."}, 500)

        def _batch(self, body, job):
            source = int(self.headers.get("X-Default-McId", "101"))
            if source not in ALLOWED:
                raise ValueError("Укажите исходную микрокатегорию от 101 до 106.")
            with tempfile.TemporaryDirectory() as temp:
                input_file = Path(temp) / "input.csv"
                input_file.write_bytes(body)
                code = run_batch(input_file, job, source, workers, process)
            if code:
                self._json({"error": f"Выгрузка требует проверки. Результаты и диагностика сохранены в {job}."}, 422)
                return
            self._send((job / "submission.csv").read_bytes(), "text/csv; charset=utf-8", filename="submission.csv")

    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as server:
        print(f"Откройте http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
