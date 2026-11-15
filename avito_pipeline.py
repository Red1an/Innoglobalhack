"""Classify ads and generate grounded drafts from the browser, JSON or CSV."""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

from private_eval import fmt
from private_eval.llm import ApiClient, load_env, parse_labels
from private_eval.preprocess import preprocess
from draft_generator.core import HTTPModel, generate

ROOT = Path(__file__).resolve().parent
PROMPTS = ROOT / "research" / "experiments" / "prompts"
RUNS = ROOT / "private_eval" / "runs" / "final"
QWEN = "qwen3-235b-instruct"
GLM = "just-ai/openrouter-proxy/z-ai/glm-5.3-flash"


class ClassificationError(Exception):
    """An unavailable classification must be visible to the caller."""


def process_ad(payload, classifier, generator, audit_dir, *, system=None, max_tokens=96):
    if not isinstance(payload, dict) or not isinstance(payload.get("description"), str):
        raise ValueError("description должен быть строкой")
    source = payload.get("mcId")
    if type(source) is not int or source not in fmt.ALLOWED:
        raise ValueError("mcId должен быть целым числом от 101 до 106")

    started = time.perf_counter()
    description = payload["description"]
    clean, _ = preprocess(description)
    metrics = {"classification_model": classifier.tag, "classification_calls": 0}
    labels = []
    if clean.strip():
        user = json.dumps({"mcId": source, "description": clean}, ensure_ascii=False)
        ans = classifier.chat(system, user, max_tokens=max_tokens)
        if ans["error"] is not None:
            raise ClassificationError(f"Не удалось получить классификацию ({ans['error']})")
        allowed = fmt.ALLOWED - {source}
        labels = parse_labels(ans["raw"], allowed)
        if ans.get("finish") not in (None, "stop") or labels is None:
            raise ClassificationError("Классификатор вернул неполный или некорректный JSON")
        metrics.update(classification_calls=1, classification_seconds=ans["seconds"],
                       classification_input_tokens=ans["input_tokens"],
                       classification_output_tokens=ans["output_tokens"],
                       classification_reported_cost=ans.get("reported_cost"))

    result, generation_metrics = generate_targets(description, labels, generator, audit_dir)
    metrics.update(generation_metrics)
    metrics["total_seconds"] = time.perf_counter() - started
    return result, metrics


def generate_targets(description, labels, generator, audit_dir, *, include_quotes=False):
    targets = [{"mcId": label} for label in labels]
    metrics = {"generation_calls": 0}
    if labels and generator is not None:
        # Citations are checked against the original ad, including punctuation and whitespace.
        Path(audit_dir).mkdir(parents=True, exist_ok=False)
        result = generate({"description": description, "targets": targets}, "rewrite", generator, audit_dir)
        path, = Path(audit_dir).glob("*.json")
        audit = json.loads(path.read_text(encoding="utf-8"))
        usage = audit["usage"] or {}
        metrics.update(generation_calls=audit["model_calls"], generation_audit=str(path),
                       generation_model=audit["configuration"]["model"],
                       generation_seconds=audit["timing"]["request_seconds"],
                       generation_input_tokens=usage.get("prompt_tokens"),
                       generation_output_tokens=usage.get("completion_tokens"),
                       generation_reported_cost=usage.get("cost"),
                       generation_error=audit["error"], generation_validation=audit["validation"])
        if include_quotes:
            result["evidence"] = audit["verified_quotes"]
    else:
        result = {"drafts": [{**target, "title": None, "text": None} for target in targets]}
    return result, metrics


def run_batch(csv_path, out, default_mcid, workers, process, *, classify_only=False):
    rows = fmt.read_input(csv_path)
    if not rows:
        raise ValueError("Во входном CSV нет объявлений")
    out.mkdir(parents=True, exist_ok=False)

    def one(item):
        index, row = item
        row["mcId"] = default_mcid if row["mcId"] is None else row["mcId"]
        try:
            result, metrics = process(row, out / "audit" / str(index))
        except (ValueError, ClassificationError) as exc:
            return {"testcase": row["testcase"], "error": str(exc)}
        return {"testcase": row["testcase"], **result, "metrics": metrics}

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(one, enumerate(rows, start=1)))
    with (out / "results.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            print(json.dumps(record, ensure_ascii=False), file=stream)
    elapsed = time.perf_counter() - started
    (out / "run.json").write_text(json.dumps({"rows": len(rows), "workers": workers, "seconds": elapsed}) + "\n",
                                  encoding="utf-8")
    print(f"{len(rows)} строк за {elapsed:.2f} с. {out / 'results.jsonl'}", file=sys.stderr)
    errors = [r for r in records if "error" in r]
    if errors:
        print(f"Строк с ошибкой классификации или входа: {len(errors)}. CSV не сформирован.", file=sys.stderr)
        return 2
    if classify_only:
        return 0
    return export_submission(records, csv_path, out, default_mcid)


def export_submission(records, csv_path, out, default_mcid):
    answers = []
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("testcase"), str):
            raise ValueError("Каждая запись результата должна содержать строку testcase")
        testcase = record["testcase"]
        drafts = record.get("drafts")
        if not isinstance(drafts, list) or any(not isinstance(d, dict) or "mcId" not in d for d in drafts):
            raise ValueError(f"{testcase}: для выгрузки нужен список черновиков с mcId")
        answers.append({"testcase": testcase, "labels": [d["mcId"] for d in drafts],
                        "descriptions": [d.get("text") for d in drafts]})
    candidate = out / "predictions.csv"
    fmt.write_answer(candidate, answers)
    problems = fmt.check_answer(candidate, csv_path, default_mcid)
    if problems:
        (out / "validation.txt").write_text("\n".join(problems) + "\n", encoding="utf-8")
        print(f"CSV не прошёл проверку формата бота. Диагностика в {out / 'validation.txt'}.", file=sys.stderr)
        return 3
    candidate.rename(out / "submission.csv")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--csv", type=Path, help="CSV организаторов; по умолчанию JSON из stdin")
    modes.add_argument("--serve", action="store_true", help="редактор черновиков и загрузка CSV в браузере")
    modes.add_argument("--replay", action="store_true", help="показать сохранённое демо и черновики без API")
    modes.add_argument("--export-reviewed", type=Path, help="выгрузить проверенный results.jsonl без вызовов API")
    parser.add_argument("--input", type=Path, help="исходный CSV для проверки выгрузки")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--classifier", choices=["qwen", "glm"], default="qwen",
                        help="qwen с v6 или glm с v9")
    parser.add_argument("--out", type=Path, help="новый каталог для результата и аудита")
    parser.add_argument("--default-mcid", type=int, choices=sorted(fmt.ALLOWED), default=101)
    parser.add_argument("--workers", type=int, default=4, help="параллельные объявления в CSV")
    parser.add_argument("--classify-only", action="store_true", help="JSON с классификацией и null вместо текстов")
    args = parser.parse_args()
    if args.csv and not args.out:
        parser.error("для CSV укажите --out")
    if args.export_reviewed and (not args.input or not args.out):
        parser.error("для --export-reviewed укажите --input и --out")
    if args.input and not args.export_reviewed:
        parser.error("--input используется с --export-reviewed")
    if args.classify_only and (args.serve or args.replay or args.export_reviewed):
        parser.error("--classify-only доступен для JSON из stdin и --csv")
    if args.workers < 1:
        parser.error("--workers должен быть положительным")

    if args.export_reviewed:
        try:
            with args.export_reviewed.open(encoding="utf-8-sig") as stream:
                records = [json.loads(line) for line in stream if line.strip()]
            if not fmt.read_input(args.input):
                raise ValueError("Во входном CSV нет объявлений")
            args.out.mkdir(parents=True, exist_ok=False)
            return export_submission(records, args.input, args.out, args.default_mcid)
        except (OSError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2

    if args.replay:
        from private_eval.final_ui import serve
        serve(args.port, args.out or RUNS / "browser", args.workers, None, None, ClassificationError, None)
        return 0

    load_env(ROOT / ".env")
    use_glm = args.classifier == "glm"
    system = (PROMPTS / ("classify_v9.txt" if use_glm else "classify_v6.txt")).read_text(encoding="utf-8-sig")
    classifier = ApiClient("glm-5.3-flash" if use_glm else QWEN,
                           base_url=os.environ.get("QWEN_API_BASE_URL") or os.environ.get("DRAFT_API_BASE_URL"),
                           key_env=os.environ.get("QWEN_KEY_ENV", "DRAFT_API_KEY"),
                           model_id=None if use_glm else os.environ.get("QWEN_MODEL"),
                           bearer=os.environ.get("QWEN_BEARER") == "1", retries=0, cache=False)
    generator = None
    if not args.classify_only:
        os.environ.setdefault("DRAFT_API_BASE_URL", classifier.base_url)
        os.environ.setdefault("DRAFT_MODEL", GLM)
        generator = HTTPModel()
        generator.parameters["max_tokens"] = 4096
        if not os.environ.get("DRAFT_API_KEY") or generator.config_error:
            parser.error(generator.config_error or "укажите DRAFT_API_KEY")
    def process(payload, audit_dir):
        return process_ad(payload, classifier, generator, audit_dir, system=system, max_tokens=2048 if use_glm else 96)

    try:
        if args.serve:
            from private_eval.final_ui import serve
            classifier.check()
            def generate_labeled(description, labels, audit_dir):
                return generate_targets(description, labels, generator, audit_dir, include_quotes=True)

            serve(args.port, args.out or RUNS / "browser", args.workers, process, run_batch,
                  ClassificationError, generate_labeled)
            return 0
        if args.csv:
            classifier.check()
            return run_batch(args.csv, args.out, args.default_mcid, args.workers, process,
                             classify_only=args.classify_only)
        payload = json.load(sys.stdin)
        out = args.out or RUNS / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        result, metrics = process(payload, out / "audit")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(json.dumps(metrics, ensure_ascii=False), file=sys.stderr)
        return 0
    except (OSError, ValueError, ClassificationError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
