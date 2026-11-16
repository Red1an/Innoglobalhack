"""Веб-интерфейс к demo/ для показа экспертам

Отдельная надстройка: demo/ работает и без этой папки. Интерфейс читает прогоны из demo/output/,
повторный прогон и черновики для всего прогона запускает командами demo/predict.py и demo/drafts.py,
черновики одного объявления делает тем же генератором из demo/draft_generator/.

Запуск из codebase/: .venv/bin/python demo_ui/server.py, затем http://localhost:8770
"""
import csv
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
DEMO = HERE.parent / "demo"
sys.path.insert(0, str(DEMO))

import predict  # noqa: E402

OUTPUT = predict.OUTPUT
PRICES = HERE / "prices.json"
PORT = 8770
LOCK = threading.Lock()
JOBS = {}


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def cost(model, tokens_in, tokens_out):
    price = json.loads(PRICES.read_text(encoding="utf-8")).get(model)
    if not price:
        return None
    amount = (tokens_in * price["input_per_million"] + tokens_out * price["output_per_million"]) / 1_000_000
    return {"amount": round(amount, 4), "currency": price["currency"], "note": price.get("note", "")}


def gold_labels(path):
    """Эталон из колонки labels или microcats во входе или из соседнего <имя>.gold.csv."""
    for candidate in (path, path.with_name(path.stem + ".gold.csv")):
        if not candidate.exists():
            continue
        with candidate.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        col = next((c for c in ("labels", "microcats") if rows and c in rows[0]), None)
        if col:
            return {r["testcase"]: set(json.loads(r[col] or "[]")) - {int(r.get("mcId") or predict.SOURCE_MCID)}
                    for r in rows}
    return None


def micro_f1(pred, gold):
    tp = sum(len(set(pred[tc]) & g) for tc, g in gold.items() if tc in pred)
    fp = sum(len(set(pred[tc]) - g) for tc, g in gold.items() if tc in pred)
    fn = sum(len(g - set(pred[tc])) for tc, g in gold.items() if tc in pred)
    return round(2 * tp / (2 * tp + fp + fn), 4) if tp + fp + fn else 1.0


def source_path(header):
    path = Path(header.get("input") or "")
    return path if path.exists() else predict.INPUT / path.name


def run_files(run_id):
    if not re.fullmatch(r"[\w.\-]+", run_id or ""):
        raise ValueError("Неверный идентификатор прогона")
    log = OUTPUT / f"{run_id}.classification.jsonl"
    if not log.exists():
        raise ValueError(f"Нет прогона {run_id}")
    return log, OUTPUT / f"{run_id}.csv", OUTPUT / f"{run_id}.drafts.jsonl"


def drafts_summary(path):
    if not path.exists():
        return None
    header, *items = jsonl(path)
    items = [it for it in items if it.get("labels") and not it.get("error")]
    request_seconds = sum(it.get("seconds") or 0 for it in items)
    return {"ads": len(items), "input_tokens": sum(it.get("input_tokens") or 0 for it in items),
            "output_tokens": sum(it.get("output_tokens") or 0 for it in items),
            "seconds": header.get("seconds"), "request_seconds": round(request_seconds, 2)}


def run_summary(log):
    header, *rows = jsonl(log)
    rows = [r for r in rows if not r["error"]]
    run_id = log.name.removesuffix(".classification.jsonl")
    src = source_path(header)
    gold = gold_labels(src) if src.exists() else None
    tin = sum(r["input_tokens"] for r in rows)
    tout = sum(r["output_tokens"] for r in rows)
    seconds, estimated = header.get("seconds"), False
    if seconds is None:
        seconds, estimated = round(sum(r["seconds"] for r in rows) / 4, 2), True
    started = header.get("started_at") or time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(log.stat().st_mtime))
    model = next((name for name, full in predict.MODELS.items() if full == header["model"]), header["model"])
    return {"id": run_id, "model": model, "prep": header.get("prep", False), "input": src.name, "started_at": started,
            "rows": len(rows), "seconds": seconds, "seconds_estimated": estimated,
            "per_ad_seconds": round(seconds / max(len(rows), 1), 3), "input_tokens": tin, "output_tokens": tout,
            "cost": cost(header["model"], tin, tout),
            "micro_f1": micro_f1({r["testcase"]: r["labels"] for r in rows}, gold) if gold else None,
            "drafts": drafts_summary(OUTPUT / f"{run_id}.drafts.jsonl")}


def run_detail(run_id):
    log, _, drafts_path = run_files(run_id)
    header, *rows = jsonl(log)
    rows = [r for r in rows if not r["error"]]
    src = source_path(header)
    texts = {r["testcase"]: r for r in predict.read_input(src)} if src.exists() else {}
    drafts = {it["testcase"]: it for it in jsonl(drafts_path)
              if "testcase" in it and not it.get("error")} if drafts_path.exists() else {}
    out = []
    for r in rows:
        d = drafts.get(r["testcase"])
        out.append({"testcase": r["testcase"], "description": texts.get(r["testcase"], {}).get("description", ""),
                    "pred": r["labels"],
                    "draft": {k: d.get(k) for k in ("drafts", "quotes", "seconds", "input_tokens", "output_tokens")}
                    if d else None})
    return {"summary": run_summary(log), "rows": out}


def state():
    runs = [r for r in map(run_summary, OUTPUT.glob("*.classification.jsonl")) if r["rows"]]
    runs.sort(key=lambda r: r["started_at"], reverse=True)
    with LOCK:
        jobs = [public_job(j) for j in JOBS.values()]
    return {"models": list(predict.MODELS), "runs": runs, "jobs": jobs, "names": predict.NAMES}


def public_job(job):
    return {k: job[k] for k in ("id", "kind", "title", "status", "done", "total", "log", "run", "started", "finished")}


def start_job(kind, title, cmd, run, finish=None):
    job = {"id": uuid.uuid4().hex[:8], "kind": kind, "title": title, "status": "running", "done": 0, "total": 0,
           "log": "", "run": run, "started": time.time(), "finished": None}
    proc = subprocess.Popen(cmd, cwd=str(DEMO.parent), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            env={**os.environ, "PYTHONUNBUFFERED": "1"})
    with LOCK:
        for old in [k for k, j in JOBS.items() if j["kind"] == kind and j["status"] != "running"]:
            del JOBS[old]
        JOBS[job["id"]] = job

    def pump():
        buffer, lines = "", []
        for chunk in iter(lambda: proc.stdout.read1(4096), b""):
            buffer += chunk.decode("utf-8", "replace")
            *parts, buffer = re.split(r"[\r\n]", buffer)
            with LOCK:
                for part in parts:
                    bar = re.search(r"\]\s+(\d+)/(\d+)", part)
                    if bar:
                        job["done"], job["total"] = int(bar.group(1)), int(bar.group(2))
                    elif part.strip():
                        lines.append(part)
                job["log"] = "\n".join(lines[-20:])
        code = proc.wait()
        with LOCK:
            note = finish(code) if finish else None
            if note:
                lines.append(note)
                job["log"] = "\n".join(lines[-20:])
            job["finished"] = time.time()
            job["status"] = "done" if code == 0 and not note else "failed"

    threading.Thread(target=pump, daemon=True).start()
    return public_job(job)


def dismiss_job(body):
    with LOCK:
        job = JOBS.get(body.get("id"))
        if job and job["status"] != "running":
            del JOBS[body["id"]]
    return {"ok": True}


def classify_job(body):
    log, _, _ = run_files(body.get("run"))
    model = body.get("model")
    if model not in predict.MODELS:
        raise ValueError("Неверная модель")
    summary = run_summary(log)
    stem = Path(summary["input"]).stem
    out = f"{stem}-{model}{'-prep' if summary['prep'] else ''}-{time.strftime('%Y%m%d-%H%M%S')}.csv"
    cmd = [sys.executable, "-u", str(DEMO / "predict.py"), summary["input"], "--model", model, "-o", out]
    if summary["prep"]:
        cmd.append("--prep")
    run_id = out.removesuffix(".csv")

    def finish(code):
        log = OUTPUT / f"{run_id}.classification.jsonl"
        if log.exists() and not any(not r["error"] for r in jsonl(log)[1:]):
            log.unlink()
            (OUTPUT / out).unlink(missing_ok=True)
            return "Ни одного успешного ответа модели, прогон не сохранён"
        return None

    return start_job("classify", f"Прогон {summary['input']}, {model}", cmd, run_id, finish)


def drafts_all_job(body):
    run_id = body.get("run")
    log, answer, _ = run_files(run_id)
    cmd = [sys.executable, "-u", str(DEMO / "drafts.py"), str(answer), "--source", str(source_path(jsonl(log)[0]))]
    files = [OUTPUT / f"{run_id}.drafts.jsonl", OUTPUT / f"{run_id}.drafts.csv"]
    saved = {f: f.read_bytes() if f.exists() else None for f in files}

    def restore():
        for f, data in saved.items():
            if data is None:
                f.unlink(missing_ok=True)
            else:
                f.write_bytes(data)

    def finish(code):
        if code != 0:
            restore()
            return "Генерация не завершилась, прежние черновики сохранены без изменений"
        new_header, *new = jsonl(files[0])
        lines = (saved[files[0]] or b"").decode("utf-8").splitlines()[1:]
        old = {it["testcase"]: it for it in map(json.loads, filter(str.strip, lines)) if not it.get("error")}
        if not any(it.get("labels") and not it.get("error") for it in new):
            restore()
            return "Ни одного успешного ответа генератора, прежние черновики сохранены без изменений"
        kept = []
        for it in new:
            if not it.get("error"):
                kept.append(it)
            elif it["testcase"] in old:
                kept.append(old[it["testcase"]])
        write_jsonl(files[0], new_header, kept)
        return None

    return start_job("drafts", f"Черновики для прогона {run_id}", cmd, run_id, finish)


def write_jsonl(path, header, items):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        for it in (header, *items):
            stream.write(json.dumps(it, ensure_ascii=False) + "\n")
    tmp.replace(path)


def error_text(error):
    """Ошибка генератора, словарь {type, message} или строка, в виде одной понятной фразы."""
    if not error:
        return None
    message = error.get("message") or error.get("type") if isinstance(error, dict) else str(error)
    code = re.search(r"HTTP (\d+)", message or "")
    hints = {"401": "API отклонил ключ, проверьте CAILA_API_KEY в demo/.env",
             "403": "API закрыт для текущего IP: Caila пускает только российские адреса, отключите VPN",
             "429": "API ограничил частоту запросов, повторите через минуту"}
    if code:
        return f"{hints.get(code.group(1), 'API вернул ошибку')} (HTTP {code.group(1)})"
    if message in ("URLError", "TimeoutError", "timeout", "OSError"):
        return f"API недоступен, проверьте сеть ({message})"
    return f"Ошибка генерации: {message}"


def draft_one(body):
    import drafts
    from draft_generator.core import HTTPModel
    log, answer, drafts_path = run_files(body.get("run"))
    header, *rows = jsonl(log)
    tc = body.get("testcase")
    row = next((r for r in rows if r["testcase"] == tc), None)
    src = source_path(header)
    text = next((r for r in predict.read_input(src) if r["testcase"] == tc), None)
    if row is None or text is None:
        raise ValueError(f"Нет объявления {tc} в прогоне")
    labels = [c for c in row["labels"] if c != text["mcId"]]
    os.environ.update(DRAFT_API_BASE_URL=predict.API_URL, DRAFT_MODEL=drafts.MODEL, DRAFT_TEMPERATURE="0")
    try:
        os.environ.setdefault("DRAFT_API_KEY", predict.load_key())
        predict.check_access(os.environ["DRAFT_API_KEY"])
    except SystemExit as exc:
        raise ValueError(str(exc)) from None
    generator = HTTPModel()
    generator.parameters["max_tokens"] = drafts.MAX_TOKENS
    item = {"testcase": tc, "mcId": text["mcId"], "labels": labels,
            **drafts.run_one({"description": text["description"], "labels": labels}, generator)}
    if item["error"]:
        raise ValueError(error_text(item["error"]))
    with LOCK:
        existing = jsonl(drafts_path) if drafts_path.exists() else []
        head = existing[0] if existing and "testcase" not in existing[0] else {
            "input": predict.rel(answer), "source": predict.rel(src), "model": drafts.MODEL, "mode": "rewrite"}
        by_tc = {it["testcase"]: it for it in existing if "testcase" in it}
        by_tc[tc] = item
        tmp = drafts_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(head, ensure_ascii=False) + "\n")
            for r in rows:
                if r["testcase"] in by_tc:
                    stream.write(json.dumps(by_tc[r["testcase"]], ensure_ascii=False) + "\n")
        tmp.replace(drafts_path)
    return {k: item[k] for k in ("drafts", "quotes", "seconds", "input_tokens", "output_tokens")} | {"error": None}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def send(self, code, body, kind="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def guard(self, fn, *args):
        try:
            self.send(200, fn(*args))
        except ValueError as exc:
            self.send(400, {"error": str(exc)})
        except Exception as exc:
            self.send(500, {"error": f"{type(exc).__name__}: {exc}"})

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            self.send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif url.path == "/api/state":
            self.guard(state)
        elif url.path == "/api/run":
            self.guard(run_detail, parse_qs(url.query).get("id", [""])[0])
        else:
            self.send(404, {"error": "not found"})

    def do_POST(self):
        routes = {"/api/classify": classify_job, "/api/drafts/all": drafts_all_job, "/api/drafts/one": draft_one,
                  "/api/jobs/dismiss": dismiss_job}
        fn = routes.get(urlparse(self.path).path)
        if fn is None:
            return self.send(404, {"error": "not found"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        except ValueError:
            return self.send(400, {"error": "Неверный JSON"})
        self.guard(fn, body)


def main():
    print(f"Интерфейс demo: http://localhost:{PORT}  (прогоны из {predict.rel(OUTPUT)}, остановка Ctrl+C)")
    try:
        ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
