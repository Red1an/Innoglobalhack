"""Прогон проверочного набора организаторов и подготовка файла ответа для Telegram-бота.

Каждый прогон пишется в private_eval/runs/<имя входа>/<имя прогона>/:
  submission.csv  файл для загрузки (testcase,microcats,descriptions)
  labels.jsonl    решение по каждой строке: метки, Stage 1, ответ LLM, ошибки, откаты, время, токены
  drafts.jsonl    черновики с заголовками и источником (extract или LLM)
  meta.json       конфигурация, время, число вызовов, токены, распределение меток
  score.json      метрики, если рядом со входом лежит <имя>.gold.csv или задан --gold

Подкоманды: predict, regen, check, score, compare, list, export-set. Подробности: run.py <подкоманда> -h.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from private_eval import fmt  # noqa: E402

RUNS = HERE / "runs"
INPUT = HERE / "input"
CATS = list(range(101, 107))


class Progress:
    def __init__(self, total, label, every=10):
        self.total, self.label, self.every, self.done = total, label, every, 0
        self.start, self.lock = time.perf_counter(), threading.Lock()

    def __call__(self):
        with self.lock:
            self.done += 1
            if self.done % self.every == 0 or self.done == self.total:
                spent = time.perf_counter() - self.start
                eta = spent / self.done * (self.total - self.done)
                print(f"  {self.label}: {self.done}/{self.total}, {spent:.0f} с, осталось ~{eta:.0f} с", flush=True)


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def run_name(args):
    if args.name:
        return args.name
    suffix = "-prep" if getattr(args, "prep", False) else ""
    if args.method == "charlr":
        return "charlr" + suffix
    if args.method == "cascade":
        return f"cascade-{args.backend}-{args.model}{suffix}"
    return f"llm-{args.backend}-{args.model}-{Path(args.prompt).stem}{suffix}"


def load_rows(path, default_mcid, prep=False):
    rows = fmt.read_input(path)
    for r in rows:
        r["mcId"] = r["mcId"] if r["mcId"] is not None else default_mcid
    if prep:
        from private_eval.preprocess import preprocess_rows
        print("предобработка:", preprocess_rows(rows), flush=True)
    return rows


def write_jsonl(path, items):
    with Path(path).open("w", encoding="utf-8") as stream:
        for item in items:
            stream.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def add_llm_args(p):
    p.add_argument("--no-cache", action="store_true",
                   help="не брать ответы LLM из кэша: честное время, для API повторная оплата")
    p.add_argument("--base-url", default=None, help="OpenAI-совместимый endpoint, по умолчанию Caila (или $API_BASE_URL)")
    p.add_argument("--key-env", default="CAILA_API_KEY", help="переменная окружения с ключом")
    p.add_argument("--bearer", action="store_true", help="заголовок 'Bearer <ключ>' (OpenAI, OpenRouter); Caila без него")
    p.add_argument("--workers", type=int, default=4, help="параллельные запросы к API; Ollama всегда по одному")
    p.add_argument("--gen", default="extract", choices=["extract", "api", "local"],
                   help="черновики: extract без LLM за секунды, api или local через LLM")
    p.add_argument("--gen-model", default=None, help="модель для черновиков; по умолчанию qwen3-235b-instruct или avibe-q8")


def make_gen_client(args):
    from private_eval.classify import make_client
    if args.gen == "extract":
        return None
    default = "qwen3-235b-instruct" if args.gen == "api" else "avibe-q8"
    return make_client(args.gen, args.gen_model or default, args)


def build_drafts(rows, records, args):
    from private_eval.drafts import generate
    client = make_gen_client(args)
    n = sum(len(r["labels"]) for r in records)
    start = time.perf_counter()
    drafts = generate(rows, records, args.gen, client, args.workers, Progress(n, "черновики", every=25))
    stats = {"gen": args.gen, "gen_model": getattr(client, "tag", None), "drafts": n,
             "gen_seconds": round(time.perf_counter() - start, 1),
             "gen_fallbacks": sum(d["error"] is not None for ds in drafts for d in ds),
             "gen_input_tokens": sum(d.get("input_tokens", 0) for ds in drafts for d in ds),
             "gen_output_tokens": sum(d.get("output_tokens", 0) for ds in drafts for d in ds)}
    if client is not None and hasattr(client, "unload"):
        client.unload()
    return drafts, stats


def save_submission(out, rows, records, drafts):
    answer = [{"testcase": r["testcase"], "labels": rec["labels"], "descriptions": [d["text"] for d in ds]}
              for r, rec, ds in zip(rows, records, drafts)]
    fmt.write_answer(out / "submission.csv", answer)
    write_jsonl(out / "drafts.jsonl", [{"testcase": r["testcase"], "drafts": ds} for r, ds in zip(rows, drafts)])


def gold_for(input_path, explicit):
    if explicit:
        return Path(explicit)
    candidate = Path(input_path).with_name(Path(input_path).stem + ".gold.csv")
    return candidate if candidate.exists() else None


def cmd_predict(args):
    from private_eval.classify import (Stage1, classify_cascade, classify_charlr, classify_llm, make_client,
                                       prompt_text)
    rows = load_rows(args.input, args.default_mcid, args.prep)
    if args.limit:
        rows = rows[:args.limit]
    out = (Path(args.out) if args.out else RUNS / Path(args.input).stem) / run_name(args)
    print(f"{len(rows)} строк из {args.input} → {out}", flush=True)
    stage1 = Stage1()
    client = None if args.method == "charlr" else make_client(args.backend, args.model, args)
    out.mkdir(parents=True, exist_ok=True)
    prompt = None
    progress = Progress(len(rows), "классификация", every=max(1, min(10, len(rows) // 25)))
    start = time.perf_counter()
    if args.method == "charlr":
        records = classify_charlr(rows, stage1, progress)
    elif args.method == "llm":
        prompt = prompt_text(args.prompt)
        records = classify_llm(rows, client, prompt, stage1, args.workers, progress)
    else:
        records = classify_cascade(rows, client, "on", stage1, progress)
    cls_seconds = time.perf_counter() - start
    if client is not None and hasattr(client, "unload") and not (args.gen == "local" and args.backend == "local"):
        client.unload()
    write_jsonl(out / "labels.jsonl", [{"testcase": r["testcase"], "mcId": r["mcId"], **rec}
                                       for r, rec in zip(rows, records)])
    drafts, gen_stats = build_drafts(rows, records, args)
    save_submission(out, rows, records, drafts)
    counts = Counter(c for rec in records for c in rec["labels"])
    meta = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git": git_commit(),
            "input": str(Path(args.input).resolve()), "input_sha": sha(args.input), "rows": len(rows),
            "method": args.method, "backend": None if args.method == "charlr" else args.backend,
            "model": None if args.method == "charlr" else getattr(client, "tag", args.model),
            "prompt": args.prompt if args.method == "llm" else None,
            "prompt_sha": hashlib.sha256(prompt.encode()).hexdigest()[:12] if prompt else None,
            "stage1_t_mid": stage1.cfg.t_mid, "default_mcid": args.default_mcid, "no_cache": args.no_cache,
            "prep": args.prep,
            "classify_seconds": round(cls_seconds, 1),
            "llm_calls": sum(rec["llm_called"] for rec in records),
            "llm_cached": sum(bool(rec.get("cached")) for rec in records),
            "llm_errors": sum(rec["error"] is not None for rec in records),
            "fallbacks_to_stage1": sum(rec["fallback"] for rec in records),
            "input_tokens": sum(rec["input_tokens"] for rec in records),
            "output_tokens": sum(rec["output_tokens"] for rec in records),
            "empty_rows": sum(not rec["labels"] for rec in records),
            "label_counts": {str(c): counts.get(c, 0) for c in CATS}, **gen_stats}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    report(out, meta)
    gold = gold_for(args.input, args.gold)
    if gold:
        print_score(score(out / "submission.csv", gold), out)


def report(out, meta):
    problems = fmt.check_answer(out / "submission.csv", meta["input"], meta["default_mcid"])
    print(f"Классификация {meta['classify_seconds']} с, вызовов LLM {meta['llm_calls']} "
          f"(из кэша {meta['llm_cached']}), ошибок {meta['llm_errors']}, откатов на Stage 1 "
          f"{meta['fallbacks_to_stage1']}, токены {meta['input_tokens']}/{meta['output_tokens']}")
    print(f"Черновики {meta['gen']}: {meta['drafts']} за {meta['gen_seconds']} с, откатов на extract {meta['gen_fallbacks']}")
    print(f"Пустых строк {meta['empty_rows']}/{meta['rows']}, метки {meta['label_counts']}")
    print("Формат: OK" if not problems else "Формат: ПРОБЛЕМЫ\n  " + "\n  ".join(problems[:20]))
    print(f"Файл для загрузки: {out / 'submission.csv'}")


def cmd_regen(args):
    out = Path(args.run)
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    labels = read_jsonl(out / "labels.jsonl")
    rows = {r["testcase"]: r for r in load_rows(meta["input"], meta["default_mcid"], meta.get("prep", False))}
    rows = [rows[rec["testcase"]] for rec in labels]
    drafts, gen_stats = build_drafts(rows, labels, args)
    save_submission(out, rows, labels, drafts)
    meta.update(gen_stats)
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    report(out, meta)


def score(pred_path, gold_path):
    pred = {r["testcase"]: set(r["labels"]) for r in fmt.read_answer(pred_path)}
    gold = {r["testcase"]: set(r["labels"]) for r in fmt.read_answer(gold_path)}
    common = [t for t in gold if t in pred]
    tp = fp = fn = 0
    per = {c: [0, 0, 0] for c in CATS}
    for t in common:
        g, p = gold[t], pred[t]
        tp, fp, fn = tp + len(g & p), fp + len(p - g), fn + len(g - p)
        for c in CATS:
            per[c][0] += c in g and c in p
            per[c][1] += c in p and c not in g
            per[c][2] += c in g and c not in p
    f1 = lambda a, b, c: 2 * a / (2 * a + b + c) if (2 * a + b + c) else 1.0  # noqa: E731
    empty = [t for t in common if not gold[t]]
    return {"gold": str(gold_path), "rows": len(common), "missing_rows": len(gold) - len(common),
            "micro_f1": round(f1(tp, fp, fn), 4), "points": round(60 * f1(tp, fp, fn), 2),
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "recall": round(tp / (tp + fn), 4) if tp + fn else None, "tp": tp, "fp": fp, "fn": fn,
            "exact_match": round(sum(gold[t] == pred[t] for t in common) / max(len(common), 1), 4),
            "empty_fp_rows": f"{sum(bool(pred[t]) for t in empty)}/{len(empty)}",
            "per_category_f1": {str(c): round(f1(*v), 4) for c, v in per.items() if sum(v)},
            "macro_f1": round(sum(f1(*v) for v in per.values() if sum(v)) / max(sum(1 for v in per.values() if sum(v)), 1), 4),
            "errors": [{"testcase": t, "gold": sorted(gold[t]), "pred": sorted(pred[t])}
                       for t in common if gold[t] != pred[t]]}


def print_score(s, run_dir=None):
    print(f"micro-F1 {s['micro_f1']} (≈{s['points']} из 60), macro-F1 {s['macro_f1']}, P {s['precision']}, R {s['recall']}, "
          f"EM {s['exact_match']}, FP на пустых {s['empty_fp_rows']}, строк {s['rows']}")
    print("  по категориям: " + ", ".join(f"{c} {v}" for c, v in s["per_category_f1"].items()))
    if run_dir:
        Path(run_dir, "score.json").write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")


def cmd_check(args):
    problems = fmt.check_answer(args.file, args.input, args.default_mcid)
    rows = fmt.read_answer(args.file) if not problems or "не читается" not in problems[0] else []
    print(f"{len(rows)} строк")
    print("OK, файл можно загружать" if not problems else "\n".join(problems))
    sys.exit(1 if problems else 0)


def cmd_score(args):
    path = Path(args.pred)
    run_dir = path if path.is_dir() else None
    pred = path / "submission.csv" if run_dir else path
    gold = args.gold
    if gold is None and run_dir:
        gold = gold_for(json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))["input"], None)
    if gold is None:
        raise SystemExit("Укажите --gold: рядом со входом нет <имя>.gold.csv")
    s = score(pred, gold)
    print_score(s, run_dir)
    if args.errors:
        for e in s["errors"]:
            print(f"  {e['testcase']}: эталон {e['gold']}, ответ {e['pred']}")


def cmd_compare(args):
    runs = [Path(r) for r in args.runs]
    preds = {r.name: {x["testcase"]: set(x["labels"]) for x in fmt.read_answer(r / "submission.csv")} for r in runs}
    names = list(preds)
    print("Попарное согласие micro-F1 (строка как эталон):")
    print("  " + " | ".join(["", *names]))
    for a in names:
        cells = []
        for b in names:
            tp = fp = fn = 0
            for t, g in preds[a].items():
                p = preds[b].get(t, set())
                tp, fp, fn = tp + len(g & p), fp + len(p - g), fn + len(g - p)
            cells.append(f"{2 * tp / max(2 * tp + fp + fn, 1):.3f}")
        print("  " + " | ".join([a, *cells]))
    tcs = list(preds[names[0]])
    diff = [t for t in tcs if len({frozenset(preds[n].get(t, set())) for n in names}) > 1]
    print(f"Строк с расхождением: {len(diff)} из {len(tcs)}")
    out = Path(args.out) if args.out else runs[0].parent / "compare.csv"
    meta = json.loads((runs[0] / "meta.json").read_text(encoding="utf-8"))
    desc = {r["testcase"]: r["description"] for r in fmt.read_input(meta["input"])}
    import csv
    with out.open("w", encoding="utf-8-sig", newline="") as stream:
        w = csv.writer(stream)
        w.writerow(["testcase", *names, "description"])
        for t in diff:
            w.writerow([t, *[json.dumps(sorted(preds[n].get(t, set()))) for n in names], desc.get(t, "")])
    print(f"Расхождения: {out}")


def cmd_list(args):
    base = Path(args.dir) if args.dir else RUNS
    metas = sorted(base.glob("**/meta.json"))
    if not metas:
        print(f"В {base} нет прогонов")
        return
    print("| набор | прогон | строк | F1 | классиф., с | черновики | вызовов LLM | ошибок | токены |")
    print("| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |")
    for m in metas:
        meta = json.loads(m.read_text(encoding="utf-8"))
        s = m.with_name("score.json")
        f1 = json.loads(s.read_text(encoding="utf-8"))["micro_f1"] if s.exists() else "—"
        print(f"| {m.parent.parent.name} | {m.parent.name} | {meta['rows']} | {f1} | {meta['classify_seconds']} | "
              f"{meta['gen']} {meta['gen_seconds']} с | {meta['llm_calls']} | {meta['llm_errors']} | "
              f"{meta['input_tokens'] + meta.get('gen_input_tokens', 0)}/"
              f"{meta['output_tokens'] + meta.get('gen_output_tokens', 0)} |")


def cmd_export_set(args):
    """Internal labeled sets in the organizer format, for rehearsals with a known answer."""
    import pyarrow.parquet as pq
    base = ROOT / "data" / "test" / f"test_{args.guide}"
    if args.set == "gold":
        src = [{"testcase": f"gold-{r['row_id']}", "description": r["description"],
                "labels": [c for c in range(102, 107) if r.get(f"y_{c}") == 1]}
               for r in pq.read_table(base / "test_gold.parquet").to_pylist()]
    else:
        raw = read_jsonl(base / f"{args.set}.jsonl")
        src = [{"testcase": str(r["row_id"]), "description": r["description"],
                "labels": sorted(set(r["labels"]) - {101})} for r in raw if r["mcId"] == 101]
    INPUT.mkdir(parents=True, exist_ok=True)
    name = f"{args.guide}_{args.set}"
    import csv
    with (INPUT / f"{name}.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        w = csv.writer(stream, lineterminator="\n")
        w.writerow(fmt.HEADER_IN)
        for r in src:
            w.writerow([r["testcase"], r["description"]])
    fmt.write_answer(INPUT / f"{name}.gold.csv",
                     [{"testcase": r["testcase"], "labels": r["labels"], "descriptions": ["-"] * len(r["labels"])}
                      for r in src])
    print(f"{len(src)} строк: {INPUT / (name + '.csv')} и {name}.gold.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("predict", help="классификация, черновики и submission.csv")
    p.add_argument("input", help="CSV организаторов: testcase,description")
    p.add_argument("--method", default="llm", choices=["llm", "cascade", "charlr"])
    p.add_argument("--backend", default="api", choices=["api", "local"], help="где работает LLM")
    p.add_argument("--model", default=None, help="имя из api_models.json или ID модели; для local avibe-q8, t-lite-q8")
    p.add_argument("--prompt", default="classify_v3_seo_on", help="промпт из research/experiments/prompts или путь")
    p.add_argument("--default-mcid", type=int, default=101, help="исходная категория, если во входе нет mcId")
    p.add_argument("--name", default=None, help="имя прогона вместо автоматического")
    p.add_argument("--out", default=None, help="каталог набора вместо runs/<имя входа>")
    p.add_argument("--limit", type=int, default=None, help="только первые N строк, для пробы")
    p.add_argument("--prep", action="store_true",
                   help="предобработка текста перед классификацией и черновиками (private_eval/preprocess.py)")
    p.add_argument("--gold", default=None, help="эталон для подсчёта метрик")
    add_llm_args(p)
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("regen", help="пересобрать черновики существующего прогона без повторной классификации")
    p.add_argument("run")
    add_llm_args(p)
    p.set_defaults(func=cmd_regen)

    p = sub.add_parser("check", help="проверить файл ответа перед загрузкой")
    p.add_argument("file")
    p.add_argument("--input", default=None, help="входной CSV для сверки testcase")
    p.add_argument("--default-mcid", type=int, default=101)
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("score", help="micro-F1 прогона или файла против эталона")
    p.add_argument("pred", help="каталог прогона или CSV ответа")
    p.add_argument("--gold", default=None)
    p.add_argument("--errors", action="store_true", help="показать строки с ошибками")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("compare", help="согласие прогонов между собой и строки с расхождениями")
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("list", help="таблица всех прогонов")
    p.add_argument("--dir", default=None)
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("export-set", help="внутренний размеченный набор в формате организаторов")
    p.add_argument("set", choices=["gold", "hard", "stress_suite"])
    p.add_argument("--guide", default="guide_v3", choices=["guide_v1", "guide_v2", "guide_v3"])
    p.set_defaults(func=cmd_export_set)

    args = ap.parse_args()
    if getattr(args, "no_cache", False):
        from private_eval import llm
        llm.READ_CACHE = False
    if getattr(args, "cmd", None) == "predict":
        if args.model is None:
            args.model = "qwen3-235b-instruct" if args.backend == "api" else "avibe-q8"
    args.func(args)


if __name__ == "__main__":
    main()
