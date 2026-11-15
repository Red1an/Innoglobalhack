"""Run a local Ollama model with a static system prompt on hard_v0 and test_v7."""
import argparse
import json
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/experiments"))
from bench import PROMPT  # noqa: E402
from export_hard_test import load_hard_test  # noqa: E402

MC = [102, 103, 104, 105, 106]
SCHEMA = {"type": "object", "required": ["labels"], "additionalProperties": False,
          "properties": {"labels": {"type": "array", "items": {"type": "integer", "enum": MC}}}}


def datasets():
    hard = load_hard_test(ROOT / "data/hard/hard_v0.xlsx")
    test = pd.read_parquet(ROOT / "data/test/test_v7.parquet")
    return {"hard_v0": [(r["row_id"], r["description"], r["labels"]) for r in hard],
            "test_v7": [(int(r.row_id), r.description, [mc for mc in MC if r[f"y_{mc}"] == 1])
                        for _, r in test.iterrows()]}


def guide_labels():
    folder = ROOT / "data/test/test_guide_v1"
    hard = [json.loads(line) for line in (folder / "hard.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    test = pd.read_parquet(folder / "test_gold.parquet")
    return {"hard_v0": {str(r["row_id"]): r["labels"] for r in hard},
            "test_v7": {str(int(r.row_id)): [mc for mc in MC if r[f"y_{mc}"] == 1] for _, r in test.iterrows()}}


def rescore(out):
    result, old, new = json.loads(out.read_text()), datasets(), guide_labels()
    for key, run in sorted(result.items()):
        name = key.split("/")[1]
        if len(run["predictions"]) != len(old[name]):
            continue
        preds = [run["predictions"][str(r[0])] for r in old[name]]
        guide = [(r[0], r[1], new[name][str(r[0])]) for r in old[name]]
        run["metrics"], run["metrics_guide_v1"] = score(old[name], preds), score(guide, preds)
        print(key, *(f"{label} F1 {m['micro_f1']} FP {m['fp']} FN {m['fn']} exact {m['exact']}/{m['rows']}"
                     for label, m in (("old", run["metrics"]), ("guide_v1", run["metrics_guide_v1"]))), sep=" | ")
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1))


def ask(tag, prompt, text, num_ctx):
    payload = {"model": tag, "stream": False, "format": SCHEMA, "keep_alive": "10m",
               "messages": [{"role": "system", "content": prompt},
                            {"role": "user", "content": json.dumps({"mcId": 101, "description": text}, ensure_ascii=False)}],
               "options": {"temperature": 0, "seed": 42, "num_ctx": num_ctx, "num_predict": 64}}
    request = Request("http://localhost:11434/api/chat", data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
    answer = json.load(urlopen(request, timeout=900))
    try:
        labels = sorted(set(json.loads(answer["message"]["content"])["labels"]))
    except (ValueError, KeyError):
        labels = None
    return labels, answer.get("prompt_eval_count")


def score(rows, preds):
    stats = {mc: [0, 0, 0] for mc in MC}
    exact = 0
    for (_, _, gold), pred in zip(rows, preds):
        pred = pred or []
        exact += set(pred) == set(gold)
        for mc in MC:
            stats[mc][0] += mc in pred and mc in gold
            stats[mc][1] += mc in pred and mc not in gold
            stats[mc][2] += mc not in pred and mc in gold
    tp, fp, fn = (sum(s[i] for s in stats.values()) for i in range(3))
    return {"micro_f1": round(2 * tp / (2 * tp + fp + fn), 4), "tp": tp, "fp": fp, "fn": fn,
            "exact": exact, "rows": len(rows), "invalid": sum(p is None for p in preds),
            "per_category": {mc: {"tp": s[0], "fp": s[1], "fn": s[2]} for mc, s in stats.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", type=Path, help="Prompt file; default is bench.PROMPT")
    parser.add_argument("--name", help="Run name inside the output file")
    parser.add_argument("--rescore", action="store_true", help="Recompute metrics of saved runs on old and guide_v1 labels")
    parser.add_argument("--model", default="hf.co/NightForger/avibe-GGUF:Q8_0")
    parser.add_argument("--data", nargs="+", default=["hard_v0", "test_v7"])
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--out", type=Path, default=ROOT / "research/experiments/results/static_prompt_local.json")
    args = parser.parse_args()
    if args.rescore:
        return rescore(args.out)
    if not args.name:
        parser.error("--name is required")
    prompt = PROMPT if args.prompt is None else args.prompt.read_text(encoding="utf-8")
    result = json.loads(args.out.read_text()) if args.out.exists() else {}
    for name in args.data:
        rows = datasets()[name]
        run = result.setdefault(f"{args.name}/{name}", {"model": args.model, "num_ctx": args.num_ctx,
                                                        "prompt": str(args.prompt or "bench.PROMPT"),
                                                        "predictions": {}, "seconds": []})
        for row_id, text, _ in rows:
            if str(row_id) in run["predictions"]:
                continue
            start = time.time()
            labels, tokens = ask(args.model, prompt, text, args.num_ctx)
            if tokens and tokens >= args.num_ctx - 64:
                raise SystemExit(f"{row_id}: context overflow, {tokens} tokens")
            run["predictions"][str(row_id)] = labels
            run["seconds"].append(round(time.time() - start, 2))
            args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
            print(name, row_id, labels, run["seconds"][-1], flush=True)
        run["metrics"] = score(rows, [run["predictions"][str(r[0])] for r in rows])
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
        print(name, json.dumps(run["metrics"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
