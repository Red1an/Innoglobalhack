import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def test_batch_preserves_original_text_and_rejects_incomplete_submission(self):
        description = "  Разводим трубы.\nЗаменяем проводку.  "
        generated = [
            {"mcId": 104, "supported": True, "title": "Сантехника",
             "text": "Разводим трубы.", "quotes": ["Разводим трубы."]},
            {"mcId": 105, "supported": True, "title": "Электрика",
             "text": "Заменяем проводку.", "quotes": ["Заменяем проводку."]},
        ]
        calls = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, value, status=200):
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(value).encode("utf-8"))

            def do_GET(self):
                self.reply({"data": []})

            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                payload = json.loads(request["messages"][1]["content"])
                calls.append((request["model"], payload))
                if request["model"] == "qwen-test":
                    content = {"labels": [105, 104, 104, 101, 999] if payload["mcId"] == 101 else [104]}
                elif request["model"] == "glm-test":
                    content = {"drafts": generated}
                else:
                    self.reply({"error": "test failure"}, 402)
                    return
                self.reply({"choices": [{"message": {"content": json.dumps(content)}, "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 300, "completion_tokens": 9, "cost": 0.012}})

        with tempfile.TemporaryDirectory() as temporary, ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            work = Path(temporary)
            original = work / "input.csv"
            with original.open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["testcase", "description", "mcId"])
                writer.writerow(["positive", description, 101])
                writer.writerow(["source-excluded", "Меняем смесители.", 104])
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            self.addCleanup(thread.join, 2)
            try:
                base_url = f"http://127.0.0.1:{server.server_port}/v1"
                env = {**os.environ, "QWEN_API_BASE_URL": base_url, "QWEN_MODEL": "qwen-test",
                       "QWEN_KEY_ENV": "PIPELINE_TEST_QWEN_KEY", "PIPELINE_TEST_QWEN_KEY": "test-qwen-secret",
                       "DRAFT_API_BASE_URL": base_url, "DRAFT_API_KEY": "test-draft-secret",
                       "DRAFT_PRICING_FILE": "", "DRAFT_TEMPERATURE": ""}
                command = [sys.executable, str(ROOT / "avito_pipeline.py"), "--csv", str(original), "--workers", "1"]

                finished = subprocess.run([*command, "--out", str(work / "valid")],
                                          env={**env,"DRAFT_MODEL": "glm-test"}, capture_output=True, text=True, timeout=10)
                self.assertEqual(finished.returncode, 0, finished.stderr)
                self.assertTrue((work / "valid" / "submission.csv").exists())
                rows = [json.loads(row) for row in (work / "valid" / "results.jsonl").read_text().splitlines()]
                self.assertEqual([d["mcId"] for d in rows[0]["drafts"]], [104, 105])
                self.assertEqual(rows[1]["drafts"], [])
                self.assertEqual(rows[0]["metrics"]["generation_input_tokens"], 300)
                self.assertEqual(rows[0]["metrics"]["classification_reported_cost"], 0.012)
                self.assertEqual([model for model, _ in calls], ["qwen-test", "glm-test", "qwen-test"])
                self.assertEqual(calls[1][1], {"description": description, "targets": [{"mcId": 104}, {"mcId": 105}]})

                failed = subprocess.run([*command, "--out", str(work / "invalid")],
                                        env={**env,"DRAFT_MODEL": "glm-error"}, capture_output=True, text=True, timeout=10)
                self.assertEqual(failed.returncode, 3, failed.stderr)
                self.assertFalse((work / "invalid" / "submission.csv").exists())
                self.assertIn("пустой черновик", (work / "invalid" / "validation.txt").read_text())
                row = json.loads((work / "invalid" / "results.jsonl").read_text().splitlines()[0])
                self.assertEqual(row["metrics"]["generation_error"]["message"], "HTTP 402")
                self.assertEqual(row["drafts"], [{"mcId": m, "title": None, "text": None} for m in (104, 105)])
                records = [json.loads(line) for line in (work / "invalid" / "results.jsonl").read_text().splitlines()]
                records[0]["drafts"][0]["text"] = "Разводим трубы."
                records[0]["drafts"][1]["text"] = "Заменяем проводку."
                reviewed = work / "reviewed.jsonl"
                reviewed.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records))
                calls_before_export = len(calls)
                exported = subprocess.run(
                    [sys.executable, str(ROOT / "avito_pipeline.py"), "--export-reviewed", str(reviewed),
                     "--input", str(original), "--out", str(work / "exported")],
                    env=env, capture_output=True, text=True, timeout=10)
                self.assertEqual(exported.returncode, 0, exported.stderr)
                self.assertEqual(len(calls), calls_before_export)
                with (work / "exported" / "submission.csv").open(encoding="utf-8-sig", newline="") as file:
                    output = list(csv.DictReader(file))
                self.assertEqual(json.loads(output[0]["descriptions"]), ["Разводим трубы.", "Заменяем проводку."])
            finally:
                server.shutdown()
