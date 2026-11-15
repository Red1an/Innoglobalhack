"""Plain local LLM with a static full prompt (research/experiments/prompts/*.txt) on the bench sets.

Same Ollama settings as research/experiments/bench.py (temperature 0, seed 42, num_ctx 8192), plus the
maxItems/uniqueItems schema limits of hybrid/stage2.py. Records are written in the hybrid/bench.py format,
so hybrid/sweep.py and hybrid/policies.py can use them as an arbiter (name static-<prompt>-<model>).
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_stress_suite import SLICE_ORDER, score_slice  # noqa: E402
from hybrid.cascade import ARTIFACTS  # noqa: E402
from hybrid.reference import bench_sets  # noqa: E402
from hybrid.stage2 import CACHE, OllamaArbiter, parse  # noqa: E402

PROMPTS = ROOT / "research" / "experiments" / "prompts"
BENCH = ARTIFACTS / "bench"


def ask(arb, prompt, row):
    allowed = [c for c in range(101, 107) if c != row["mcId"]]
    user = json.dumps({"mcId": row["mcId"], "description": row["description"]}, ensure_ascii=False)
    schema = {"type": "object", "required": ["labels"], "additionalProperties": False,
              "properties": {"labels": {"type": "array", "maxItems": len(allowed), "uniqueItems": True,
                                        "items": {"type": "integer", "enum": allowed}}}}
    options = {"temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": 64}
    digest = hashlib.sha256(json.dumps(["static", arb.tag, prompt, user, options], ensure_ascii=False).encode()).hexdigest()
    path = CACHE / "static" / f"{digest}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        rec = json.loads(path.read_text(encoding="utf-8"))
        return {**rec, "labels": parse(rec["raw"], allowed), "cached": True}
    start = time.perf_counter()
    try:
        resp = arb._post("/api/chat", {"model": arb.tag, "stream": False, "format": schema, "keep_alive": "30m",
                                       "options": options,
                                       "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": user}]})
    except (OSError, ValueError) as exc:
        return {"labels": None, "raw": None, "seconds": time.perf_counter() - start, "input_tokens": 0,
                "output_tokens": 0, "error": type(exc).__name__, "cached": False}
    rec = {"raw": resp.get("message", {}).get("content"), "seconds": time.perf_counter() - start,
           "input_tokens": resp.get("prompt_eval_count", 0), "output_tokens": resp.get("eval_count", 0), "error": None}
    if rec["input_tokens"] + 64 >= 8192:
        rec["error"] = "context_overflow"
    labels = parse(rec["raw"], allowed) if rec["error"] is None else None
    if labels is None and rec["error"] is None:
        rec["error"] = "invalid_response"
    if rec["error"] is None:
        path.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    return {**rec, "labels": labels, "cached": False}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt", default="classify_v2_seo_off")
    ap.add_argument("--model", default="avibe-q8")
    ap.add_argument("--sets", default="hard,test_gold,stress_suite")
    args = ap.parse_args()
    prompt = (PROMPTS / f"{args.prompt}.txt").read_text(encoding="utf-8")
    arb = OllamaArbiter(args.model)
    name = f"static-{args.prompt}-{args.model}"
    summary_path = BENCH / "static_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    for set_name, rows in bench_sets().items():
        if set_name not in args.sets.split(","):
            continue
        print(f"== {name} on {set_name} ({len(rows)} rows)", flush=True)
        records = []
        for i, row in enumerate(rows, 1):
            a = ask(arb, prompt, row)
            labels = a["labels"] if a["labels"] is not None else []
            records.append({"row_id": row["row_id"], "mcId": row["mcId"], "stage1": [], "llm": a["labels"],
                            "llm_error": a["error"], "labels": labels, "seconds": a["seconds"],
                            "input_tokens": a["input_tokens"], "output_tokens": a["output_tokens"], "cached": a["cached"]})
            if i % 25 == 0:
                print(f"  {i}/{len(rows)}", flush=True)
        with (BENCH / f"{name}__{set_name}.jsonl").open("w", encoding="utf-8") as stream:
            for rec in records:
                stream.write(json.dumps(rec, ensure_ascii=False) + "\n")
        m = score_slice([r["labels"] for r in rows], [r["labels"] for r in records], [r["mcId"] for r in rows])
        live = [r["seconds"] for r in records if not r["cached"]] or [0.0]
        res = {"rows": len(rows), "micro_f1": m["micro_f1"], "exact_match": m["exact_match"],
               "empty_rows": m["empty_rows"], "empty_fp_rows": m["empty_fp_rows"],
               "errors": sum(r["llm_error"] is not None for r in records),
               "latency_p50_s": float(np.median(live)), "latency_p95_s": float(np.quantile(live, 0.95))}
        if set_name == "stress_suite":
            res["slices"] = {}
            for s in SLICE_ORDER + ["mentor_borderline_switches"]:
                idx = [i for i, r in enumerate(rows) if r["slice"] == s]
                if idx:
                    ms = score_slice([rows[i]["labels"] for i in idx], [records[i]["labels"] for i in idx],
                                     [rows[i]["mcId"] for i in idx])
                    res["slices"][s] = {"rows": len(idx), "micro_f1": ms["micro_f1"], "exact_match": ms["exact_match"]}
        summary.setdefault(name, {})[set_name] = res
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in res.items() if k != "slices"}, flush=True)
    arb.unload()


if __name__ == "__main__":
    main()
