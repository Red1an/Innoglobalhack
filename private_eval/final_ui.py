"""Local browser entry point for ad review and the organizer CSV."""
import csv
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

from draft_generator.core import CATEGORIES

from .fmt import ALLOWED, read_answer

PAGE = Path(__file__).with_name("final_demo.html")
REVIEW_PAGE = PAGE.with_name("draft_review.html")
DEMO_CSV = PAGE.parent / "examples" / "example.csv"
REVIEW_CSV = PAGE.parent.parent / "data" / "private-test-dataset.labeled.csv"
BASELINE_CSV = PAGE.parent / "input" / "private-test-dataset.predict.csv"
RECORDINGS = PAGE.parent / "examples" / "draft-review"
QUICK_CASES = {
    "case-113": "Длинное объявление ремонтной бригады",
    "case-018": "Стресс-тест · услуги в 72 тыс. знаков",
    "case-001": "Пять услуг в одном объявлении",
    "case-057": "Скидка только на обои",
    "case-077": "Сантехника с ограничениями",
    "case-120": "Цена за м² с условием",
    "case-176": "Два мастера, разные условия",
    "case-126": "Без дополнительных услуг",
}
MAX_INPUT_BYTES = 4 * 1024 * 1024


def load_review_cases(path):
    cases = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            testcase, source, labels = row["testcase"], int(row["mcId"]), json.loads(row["labels"])
            if (testcase in cases or source not in ALLOWED or not isinstance(labels, list)
                    or any(type(label) is not int or label not in ALLOWED - {source} for label in labels)
                    or len(labels) != len(set(labels))):
                raise ValueError(f"Некорректная разметка {testcase}")
            cases[testcase] = {
                "testcase": testcase, "mcId": source, "labels": labels, "description": row["description"],
                "needs_review": row["needs_review"].lower() == "true", "annotation": row["annotation_reason"],
            }
    return cases


def serve(port, out, workers, process, run_batch, classification_error, generate_labeled):
    cases = load_review_cases(REVIEW_CSV)
    live = process is not None
    baseline = {
        row["testcase"]: dict(zip(row["labels"], row["descriptions"]))
        for row in (read_answer(BASELINE_CSV) if BASELINE_CSV.is_file() else [])
    }
    review_dir = out / "review"
    index = {"categories": CATEGORIES, "live": live,
             "quick": {key: title for key, title in QUICK_CASES.items() if key in cases},
             "has_baseline": bool(baseline), "cases": [
                 {"testcase": key, "labels": item["labels"], "needs_review": item["needs_review"],
                  "preview": " ".join(item["description"].split())[:220]}
                 for key, item in cases.items()]}

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
            elif self.path == "/" and not live:
                self.send_response(303)
                self.send_header("Location", "/demo")
                self.end_headers()
            elif self.path in {"/", "/demo"}:
                page = PAGE.read_bytes()
                if not live:
                    page = page.replace(b"<body>", b'<body data-replay="true">', 1)
                self._send(page, "text/html; charset=utf-8")
            elif self.path == "/demo.csv":
                self._send(DEMO_CSV.read_bytes(), "text/csv; charset=utf-8")
            elif urlsplit(self.path).path == "/review":
                self._send(REVIEW_PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/review/index":
                self._json(index)
            elif urlsplit(self.path).path == "/review/case":
                testcase = parse_qs(urlsplit(self.path).query).get("testcase", [""])[0]
                if testcase not in cases:
                    self._json({"error": "Объявление отсутствует в наборе."}, 404)
                    return
                latest = review_dir / "latest" / f"{testcase}.json"
                if not live or not latest.is_file():
                    latest = RECORDINGS / f"{testcase}.json"
                self._json({**cases[testcase], "baseline": baseline.get(testcase, {}),
                            "run": json.loads(latest.read_text(encoding="utf-8")) if latest.is_file() else None})
            else:
                self.send_error(404)

        def do_POST(self):
            media = self.headers.get_content_type()
            allowed_paths = {"application/json": {"/drafts", "/review/generate"}, "text/csv": {"/batch"}}
            if not self._local_host() or self.path not in allowed_paths.get(media, ()):
                self.send_error(404)
                return
            if not live:
                self._json({"error": "Открыта запись. Для живого запроса запустите сервер с --serve."}, 503)
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
                elif self.path == "/review/generate":
                    self._generate_review(json.loads(body), job.name)
                else:
                    self._batch(body, job)
            except ValueError as exc:
                self._json({"error": str(exc)}, 400)
            except classification_error as exc:
                self._json({"error": str(exc)}, 502)
            except OSError:
                self._json({"error": "Не удалось сохранить результат. Проверьте доступ к папке запуска."}, 500)

        def _generate_review(self, payload, run_id):
            testcase = payload.get("testcase") if isinstance(payload, dict) else None
            if not isinstance(testcase, str) or testcase not in cases:
                self._json({"error": "Объявление отсутствует в наборе."}, 404)
                return
            item = cases[testcase]
            job = review_dir / "runs" / run_id
            result, metrics = generate_labeled(item["description"], item["labels"], job / "audit")
            has_draft = any(draft["title"] and draft["text"] for draft in result["drafts"])
            if item["labels"] and not has_draft:
                self._json({"error": "Генератор не вернул пригодных черновиков.", "metrics": metrics}, 502)
                return
            run = {**result, "id": run_id, "metrics": metrics}
            latest = review_dir / "latest" / f"{testcase}.json"
            latest.parent.mkdir(parents=True, exist_ok=True)
            pending = latest.with_suffix(f".{run_id}.tmp")
            pending.write_text(json.dumps(run, ensure_ascii=False), encoding="utf-8")
            pending.replace(latest)
            self._json(run)

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
