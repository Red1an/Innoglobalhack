"""Классификация объявлений

Вход берётся из demo/input/, ответ пишется в demo/output/<имя входа>-<модель>[-prep].csv.
В descriptions названия категорий, с --drafts тексты черновиков из demo/drafts.py.

Пример:
  python demo/predict.py example.csv
  python demo/predict.py private-test.csv --prep
  python demo/predict.py private-test.csv --model qwen
  python demo/predict.py private-test.csv --prep --drafts   классификация и тексты черновиков в том же ответе
"""
import argparse
import csv
import hashlib
import json
import os
import re
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
PROMPT = HERE / "prompt_v9.txt"
INPUT = HERE / "input"
OUTPUT = HERE / "output"
API_URL = "https://caila.io/api/adapters/openai"
MODELS = {"glm": "just-ai/openrouter-proxy/z-ai/glm-5.3-flash",
          "qwen": "just-ai/openrouter-proxy/qwen/qwen3-235b-a22b-2507"}
DEFAULT_MODEL = "glm"
MAX_TOKENS = 600
SOURCE_MCID = 101
NAMES = {101: "Ремонт под ключ", 102: "Штукатурные работы", 103: "Плиточные работы", 104: "Сантехника",
         105: "Электрика", 106: "Поклейка обоев и малярные работы"}
RETRIES = 3


def rel(path):
    try:
        return str(Path(path).resolve().relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def load_key():
    env = HERE / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            name, sep, value = line.strip().partition("=")
            if sep and not name.startswith("#"):
                os.environ.setdefault(name.strip(), value.strip())
    key = os.environ.get("CAILA_API_KEY", "")
    if not key:
        raise SystemExit("Нет ключа: задайте CAILA_API_KEY в окружении или строкой CAILA_API_KEY=... в demo/.env")
    return key


def check_access(key):
    host = urlsplit(API_URL).hostname
    try:
        address = socket.gethostbyname(host)
    except OSError:
        raise SystemExit(f"Не удаётся найти адрес {host}, проверьте сеть") from None
    hint = (f" Адрес {host} подменён VPN ({address}), отключите VPN или добавьте {host} в исключения."
            if address.startswith(("198.18.", "198.19.")) else "")
    try:
        with urlopen(Request(API_URL + "/models", headers={"Authorization": key}), timeout=30) as resp:
            json.load(resp)
    except HTTPError as exc:
        body = exc.read(300).decode("utf-8", "replace").lower()
        if exc.code == 403 and "<html" in body:
            raise SystemExit("API закрыт для текущего IP: Caila пускает только российские адреса." + hint) from None
        raise SystemExit(f"API ответил HTTP {exc.code} на проверку доступа, проверьте ключ") from None
    except (URLError, TimeoutError, socket.timeout, ValueError) as exc:
        raise SystemExit(f"API недоступен: {type(exc).__name__}." + hint) from None


def chat(key, model, system, user):
    payload = {"model": model, "temperature": 0, "max_tokens": MAX_TOKENS, "stream": False,
               "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    body = json.dumps(payload, ensure_ascii=False).encode()
    headers = {"Content-Type": "application/json", "Accept": "application/json", "Authorization": key}
    start, error = time.perf_counter(), None
    for attempt in range(RETRIES + 1):
        try:
            with urlopen(Request(API_URL + "/chat/completions", data=body, headers=headers), timeout=90) as resp:
                value = json.load(resp)
            content = ((value.get("choices") or [{}])[0].get("message") or {}).get("content")
            usage = value.get("usage") or {}
            return {"raw": content, "error": None if content else "empty_response",
                    "seconds": round(time.perf_counter() - start, 2),
                    "input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0)}
        except HTTPError as exc:
            error = f"HTTP {exc.code}"
            if exc.code not in (408, 409, 429, 500, 502, 503, 504):
                break
        except (URLError, TimeoutError, socket.timeout, ValueError) as exc:
            error = type(exc).__name__
        time.sleep(min(2 ** attempt, 10))
    return {"raw": None, "error": error, "seconds": round(time.perf_counter() - start, 2),
            "input_tokens": 0, "output_tokens": 0}


def is_offer(value):
    if isinstance(value, dict):
        quote = value.get("quote")
        return value.get("status") == "offer" and isinstance(quote, str) and bool(quote.strip())
    return isinstance(value, str) and bool(value.strip())


def parse_labels(content, allowed):
    """Ответ v9: {"refusal": ..., "evidence": {"102": {"quote", "status"}, ...}}, метка только при status offer."""
    if not isinstance(content, str):
        return None
    text = content.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.S | re.I)
    if fence:
        text = fence.group(1)
    if not text.startswith("{"):
        found = re.search(r"\{[^{}]*\"labels\"[^{}]*\}", text)
        outer = text[text.find("{"):text.rfind("}") + 1] if "{" in text else ""
        if '"evidence"' in outer:
            text = outer
        elif found:
            text = found.group(0)
        else:
            return None
    try:
        value = json.loads(text)
        if "labels" in value:
            labels = value["labels"]
        else:
            labels = sorted(int(k) for k, v in value["evidence"].items() if is_offer(v))
    except (ValueError, TypeError, KeyError, AttributeError):
        return None
    if not isinstance(labels, list) or any(type(x) is not int for x in labels):
        return None
    return sorted(set(labels) & set(allowed))


class Bar:
    def __init__(self, total, width=30):
        self.total, self.width, self.done, self.errors = total, width, 0, 0
        self.start, self.lock = time.perf_counter(), threading.Lock()
        self.draw()

    def step(self, error=False):
        with self.lock:
            self.done += 1
            self.errors += error
            self.draw()

    def draw(self):
        spent = time.perf_counter() - self.start
        left = spent / self.done * (self.total - self.done) if self.done else 0
        fill = int(self.width * self.done / max(self.total, 1))
        line = (f"\r[{'█' * fill}{'░' * (self.width - fill)}] {self.done}/{self.total} "
                f"{100 * self.done // max(self.total, 1):3d}%  {spent:4.0f} с, осталось ~{left:.0f} с")
        if self.errors:
            line += f", ошибок {self.errors}"
        print(line, end="\n" if self.done == self.total else "", flush=True)


def read_input(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = {"testcase", "description"} - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{path}: нет колонок {sorted(missing)}")
        rows = [{"testcase": r["testcase"], "description": r["description"] or "",
                 "mcId": int(r["mcId"]) if r.get("mcId") else SOURCE_MCID} for r in reader]
    if len({r["testcase"] for r in rows}) != len(rows):
        raise SystemExit(f"{path}: повторяются testcase")
    return rows


def write_answer(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        w = csv.writer(stream, lineterminator="\n")
        w.writerow(["testcase", "microcats", "descriptions"])
        for r in rows:
            w.writerow([r["testcase"], json.dumps(r["labels"], separators=(",", ":")),
                        json.dumps(r["drafts"], ensure_ascii=False, separators=(",", ":"))])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="имя CSV в demo/input/ или путь к нему, колонки testcase,description")
    ap.add_argument("-o", "--out", default=None, help="имя файла ответа в demo/output/ или полный путь")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS),
                    help=f"модель по названию, по умолчанию {DEFAULT_MODEL}")
    ap.add_argument("--prep", action="store_true", help="предобработка текста: инъекции, эмодзи, омоглифы")
    ap.add_argument("--workers", type=int, default=4, help="параллельные запросы к API, по умолчанию 4")
    ap.add_argument("--drafts", action="store_true", help="после классификации записать в descriptions тексты черновиков")
    ap.add_argument("--limit", type=int, default=None, help="только первые N строк, для пробы")
    args = ap.parse_args()

    src = Path(args.input)
    if not src.exists() and (INPUT / src).exists():
        src = INPUT / src
    if not src.exists():
        raise SystemExit(f"Нет файла {args.input} ни по этому пути, ни в {INPUT}")
    out = Path(args.out or f"{src.stem}-{args.model}{'-prep' if args.prep else ''}.csv")
    if out.parent == Path("."):
        out = OUTPUT / out.name
    args.input = str(src)

    rows = read_input(args.input)[:args.limit]
    prompt = PROMPT.read_text(encoding="utf-8-sig")
    model = MODELS[args.model]
    print(f"Вход: {rel(args.input)}, строк {len(rows)}")
    print(f"Модель: {args.model} ({model})")
    print(f"Промпт: {PROMPT.name}, sha256 {hashlib.sha256(prompt.encode()).hexdigest()[:12]}")
    if args.prep:
        from preprocess import preprocess_rows
        print("Предобработка:", ", ".join(f"{k} {v}" for k, v in preprocess_rows(rows).items()))
    else:
        print("Предобработка: выключена")
    key = load_key()
    check_access(key)

    bar = Bar(len(rows))

    def classify(r):
        allowed = [c for c in range(101, 107) if c != r["mcId"]]
        user = json.dumps({"mcId": r["mcId"], "description": r["description"]}, ensure_ascii=False)
        ans = chat(key, model, prompt, user)
        labels = parse_labels(ans["raw"], allowed) if ans["error"] is None else None
        ans["error"] = ans["error"] or (None if labels is not None else "invalid_response")
        bar.step(ans["error"] is not None)
        return {"testcase": r["testcase"], "labels": labels or [], **ans}

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(classify, rows))
    seconds = time.perf_counter() - start

    for r, res in zip(rows, results):
        res["drafts"] = [f"{NAMES[c]}." for c in res["labels"]]
    out.parent.mkdir(parents=True, exist_ok=True)
    write_answer(out, results)
    log = out.with_suffix(".classification.jsonl")
    with log.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps({"model": model, "prompt": PROMPT.name, "prep": args.prep,
                                 "input": str(Path(args.input).resolve())}, ensure_ascii=False) + "\n")
        for res in results:
            stream.write(json.dumps({k: res[k] for k in ("testcase", "labels", "error", "seconds", "input_tokens",
                                                       "output_tokens", "raw")}, ensure_ascii=False) + "\n")

    errors = [res["testcase"] for res in results if res["error"]]
    counts = {c: sum(c in res["labels"] for res in results) for c in range(102, 107)}
    print(f"Готово за {seconds:.0f} с, токены {sum(r['input_tokens'] for r in results)}"
          f"/{sum(r['output_tokens'] for r in results)}")
    print(f"Пустых ответов {sum(not r['labels'] for r in results)}/{len(results)}, метки "
          + ", ".join(f"{c}: {n}" for c, n in counts.items()))
    if errors:
        print(f"Ошибки API у {len(errors)} строк, им записан пустой список: {', '.join(errors[:10])}")
    print(f"Ответ: {rel(out)}")
    print(f"Сырые ответы модели: {rel(log)}")
    if args.drafts:
        from drafts import run
        print("\nЧерновики")
        run(str(out), args.input, args.workers, answer=out)
    else:
        print(f"Черновики отдельно: python demo/drafts.py {rel(out)} --source {rel(args.input)}")


if __name__ == "__main__":
    main()
