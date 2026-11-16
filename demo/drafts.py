"""Черновики объявлений по целевым микрокатегориям

Второй этап после predict.py, его же запускает predict.py --drafts. На входе файл, где у каждого
testcase уже есть целевые микрокатегории в колонке microcats или labels: ответ классификации или размеченный набор.
Если в файле нет колонки description, текст берётся из --source
(по умолчанию demo/input/<имя без -<модель>-prep>.csv).

Генератор коллеги из draft_generator/, режим rewrite: один запрос к GLM на объявление,
каждый черновик проверяется дословными цитатами из исходного текста.

Результат в demo/output/:
  <имя>.drafts.csv    формат организаторов, descriptions с текстами черновиков
                      (при predict.py --drafts тексты пишутся прямо в ответ классификации)
  <имя>.drafts.jsonl  сырые ответы генератора, цитаты и статус проверки по каждой категории

Пример:
  python demo/drafts.py private-test-glm-prep.csv
  python demo/drafts.py private-test-dataset.labeled.csv --limit 5
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from predict import API_URL, INPUT, MODELS, NAMES, OUTPUT, SOURCE_MCID, Bar, check_access, load_key, read_input, rel  # noqa: E402

MODEL = MODELS["glm"]
MAX_TOKENS = 4096
LABEL_COLUMNS = ("microcats", "labels")
STATUS = {"quotes_verified": "цитаты сверены", "invalid_source_quotes": "цитат нет в исходнике",
          "unsupported_category": "генератор не нашёл услугу в тексте", "model_failure": "ошибка модели",
          "missing_or_duplicate_category": "категории нет в ответе модели", "invalid_draft_text": "пустой текст"}
# Где проверка не прошла, в descriptions пишется название категории, чтобы формат оставался валидным.


def find(name, *bases):
    for path in (Path(name), *(base / name for base in bases)):
        if path.exists():
            return path
    raise SystemExit(f"Нет файла {name} ни по этому пути, ни в {', '.join(map(str, bases))}")


def read_targets(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        col = next((c for c in LABEL_COLUMNS if c in (reader.fieldnames or [])), None)
        if col is None or "testcase" not in (reader.fieldnames or []):
            raise SystemExit(f"{path}: нужны колонки testcase и microcats или labels, есть {reader.fieldnames}")
        rows = {}
        for r in reader:
            labels = json.loads(r[col] or "[]")
            if not isinstance(labels, list) or any(type(x) is not int or x not in NAMES for x in labels):
                raise SystemExit(f"{path}: {r['testcase']}: метки не из 101–106: {labels}")
            rows[r["testcase"]] = {"labels": sorted(set(labels)), "description": r.get("description"),
                                   "mcId": int(r["mcId"]) if r.get("mcId") else None}
    return rows


def default_source(path):
    stem = path.stem.removesuffix("-prep")
    for name in MODELS:
        stem = stem.removesuffix(f"-{name}")
    return INPUT / f"{stem}.csv"


def run_one(row, generator):
    from draft_generator import generate
    targets = [{"mcId": c} for c in row["labels"]]
    if not targets:
        return {"drafts": [], "validation": [], "quotes": {}, "error": None,
                "seconds": 0.0, "input_tokens": 0, "output_tokens": 0, "raw": None}
    with tempfile.TemporaryDirectory() as audit_dir:
        result = generate({"description": row["description"], "targets": targets}, "rewrite", generator, audit_dir)
        audit = json.loads(next(Path(audit_dir).glob("*.json")).read_text(encoding="utf-8"))
    usage = audit["usage"] or {}
    return {"drafts": result["drafts"], "validation": audit["validation"], "quotes": audit["verified_quotes"],
            "error": audit["error"], "seconds": round(audit["timing"]["request_seconds"], 2),
            "input_tokens": usage.get("prompt_tokens") or 0, "output_tokens": usage.get("completion_tokens") or 0,
            "raw": audit.get("model_content")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="файл с testcase и microcats/labels: из demo/output/, demo/input/ или путь")
    ap.add_argument("--source", default=None, help="CSV с testcase,description, если во входе нет текста")
    ap.add_argument("--workers", type=int, default=4, help="параллельные объявления, по умолчанию 4")
    ap.add_argument("--limit", type=int, default=None, help="только первые N строк, для пробы")
    args = ap.parse_args()
    run(args.input, args.source, args.workers, args.limit)


def run(input_name, source_name=None, workers=4, limit=None, answer=None):
    src = find(input_name, OUTPUT, INPUT)
    targets = read_targets(src)
    if any(t["description"] is None for t in targets.values()):
        source = find(source_name, INPUT) if source_name else default_source(src)
        if not source.exists():
            raise SystemExit(f"Во входе нет description, укажите --source (не найден {source})")
        texts = {r["testcase"]: r for r in read_input(source)}
        missing = set(targets) - set(texts)
        if missing:
            raise SystemExit(f"В {source} нет {len(missing)} testcase из входа, например {sorted(missing)[0]}")
        for tc, t in targets.items():
            t["description"], t["mcId"] = texts[tc]["description"], t["mcId"] or texts[tc]["mcId"]
    else:
        source = src
    rows = [{"testcase": tc, **t, "mcId": t["mcId"] or SOURCE_MCID} for tc, t in targets.items()][:limit]
    for r in rows:
        r["labels"] = [c for c in r["labels"] if c != r["mcId"]]

    os.environ.update(DRAFT_API_BASE_URL=API_URL, DRAFT_MODEL=MODEL, DRAFT_TEMPERATURE="0")
    from draft_generator.categories import guide_version
    from draft_generator.core import HTTPModel
    guide = guide_version()
    print(f"Вход: {rel(src)}, строк {len(rows)}, категорий {sum(len(r['labels']) for r in rows)}")
    print(f"Тексты объявлений: {rel(source)}")
    print(f"Генератор: draft_generator, режим rewrite, модель {MODEL}")
    print(f"Справочник: {guide['file']}, sha256 {guide['sha256'][:12]}")
    os.environ.setdefault("DRAFT_API_KEY", load_key())
    check_access(os.environ["DRAFT_API_KEY"])
    generator = HTTPModel()
    generator.parameters["max_tokens"] = MAX_TOKENS

    base = OUTPUT / f"{src.stem}.drafts"
    bar = Bar(len(rows))

    def one(r):
        res = run_one(r, generator)
        bar.step(res["error"] is not None)
        return {"testcase": r["testcase"], "mcId": r["mcId"], "labels": r["labels"],
                "description": r["description"], **res}

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        items = list(pool.map(one, rows))
    seconds = time.perf_counter() - start

    OUTPUT.mkdir(parents=True, exist_ok=True)
    answer = Path(answer or f"{base}.csv")
    with answer.open("w", encoding="utf-8-sig", newline="") as stream:
        w = csv.writer(stream, lineterminator="\n")
        w.writerow(["testcase", "microcats", "descriptions"])
        for it in items:
            texts = [d["text"] or f"{NAMES[d['mcId']]}." for d in it["drafts"]]
            w.writerow([it["testcase"], json.dumps(it["labels"], separators=(",", ":")),
                        json.dumps(texts, ensure_ascii=False, separators=(",", ":"))])
    header = {"input": rel(src), "source": rel(source), "model": MODEL, "mode": "rewrite",
              "guide_sha256": guide["sha256"],
              "generator_sha256": hashlib.sha256(Path(sys.modules["draft_generator.core"].__file__).read_bytes()).hexdigest()}
    with Path(f"{base}.jsonl").open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(header, ensure_ascii=False) + "\n")
        for it in items:
            stream.write(json.dumps({k: v for k, v in it.items() if k != "description"}, ensure_ascii=False) + "\n")

    statuses = {}
    for it in items:
        for v in it["validation"]:
            statuses[v["status"]] = statuses.get(v["status"], 0) + 1
    errors = [it["testcase"] for it in items if it["error"]]
    print(f"Готово за {seconds:.0f} с, токены {sum(i['input_tokens'] for i in items)}"
          f"/{sum(i['output_tokens'] for i in items)}")
    print("Черновики по категориям: " + (", ".join(f"{STATUS.get(k, k)} {v}" for k, v in sorted(statuses.items())) or "нет меток"))
    if errors:
        print(f"Ошибки генератора у {len(errors)} объявлений: {', '.join(errors[:10])}")
    print(f"Ответ для бота: {rel(answer)}")
    print(f"Сырые ответы генератора: {rel(base)}.jsonl")


if __name__ == "__main__":
    main()
