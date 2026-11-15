"""Score saved reference predictions (Qwen3-235B via API, plain local avibe) on the guide_v2 labels.

No inference. Reference rows are matched to guide_v2 rows by normalized description, so the scores use
exactly the rows and labels of hybrid/bench.py (hard, test_gold, stress_suite without gold_core and dev copies).
Reference runs are read from output/hybrid/reference/<name>/{test,predictions}.jsonl (ClearML artifacts)
and from research/experiments/results/static_prompt_local.json (plain avibe runs with static prompts).
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_stress_suite import SLICE_ORDER, score_slice  # noqa: E402
from hybrid.cascade import ARTIFACTS  # noqa: E402
from hybrid.data import eval_sets, read_any  # noqa: E402

REF = ARTIFACTS / "reference"
STATIC = ROOT / "research" / "experiments" / "results" / "static_prompt_local.json"


def key(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(text))).strip().lower()


def bench_sets():
    sets = eval_sets()
    sets["stress_suite"] = [r for r in sets["stress_suite"]
                            if r.get("origin_split") != "dev" and r["slice"] != "gold_core"]
    return sets


def reference_runs():
    """name -> {text key: labels}."""
    runs = {}
    for folder in sorted(p for p in REF.iterdir() if (p / "predictions.jsonl").exists()):
        test = [json.loads(line) for line in (folder / "test.jsonl").open(encoding="utf-8")]
        preds = [json.loads(line) for line in (folder / "predictions.jsonl").open(encoding="utf-8")]
        name = folder.name.rsplit("_", 1)[0]
        runs.setdefault(name, {}).update({key(t["description"]): p["labels"] if p.get("valid", True) else []
                                          for t, p in zip(test, preds)})
    static = json.loads(STATIC.read_text(encoding="utf-8"))
    import pandas as pd
    hard = {r["id"]: r["description"] for r in pd.read_excel(ROOT / "data" / "hard" / "hard_v0.xlsx").to_dict("records")}
    gold = {str(r["row_id"]): r["description"] for r in read_any(ROOT / "data" / "test" / "test_v7.parquet")}
    for run, value in static.items():
        name, dataset = run.split("/")
        texts = hard if dataset == "hard_v0" else gold
        runs.setdefault(f"avibe_static_{name.split('-', 1)[1]}", {}).update(
            {key(texts[i]): labels for i, labels in value["predictions"].items() if i in texts})
    return runs


def main():
    sets = bench_sets()
    out = {}
    for name, preds in reference_runs().items():
        for set_name, rows in sets.items():
            idx = [i for i, r in enumerate(rows) if key(r["description"]) in preds]
            if len(idx) < len(rows):
                continue
            sub = [rows[i] for i in idx]
            m = score_slice([r["labels"] for r in sub], [preds[key(r["description"])] for r in sub], [r["mcId"] for r in sub])
            res = {"rows": len(sub), "of": len(rows), "micro_f1": m["micro_f1"], "exact_match": m["exact_match"],
                   "empty_rows": m["empty_rows"], "empty_fp_rows": m["empty_fp_rows"]}
            if set_name == "stress_suite":
                res["slices"] = {}
                for s in SLICE_ORDER + ["mentor_borderline_switches"]:
                    si = [r for r in sub if r["slice"] == s]
                    if si:
                        ms = score_slice([r["labels"] for r in si], [preds[key(r["description"])] for r in si],
                                         [r["mcId"] for r in si])
                        res["slices"][s] = {"rows": len(si), "micro_f1": ms["micro_f1"], "exact_match": ms["exact_match"]}
            out.setdefault(name, {})[set_name] = res
            print(name, set_name, f"{len(sub)}/{len(rows)}", round(m["micro_f1"], 4), round(m["exact_match"], 3),
                  f"FP empty {m['empty_fp_rows']}/{m['empty_rows']}")
    (ARTIFACTS / "bench" / "reference_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
