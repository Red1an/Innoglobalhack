"""Compare pure Stage 1, pure local LLM and the Stage 1 → LLM cascade on test_gold, hard and stress_suite."""
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_stress_suite import SLICE_ORDER, evaluate_and_report, score_slice  # noqa: E402
from hybrid.cascade import ARTIFACTS, load, run  # noqa: E402
from hybrid.data import STRESS, eval_sets  # noqa: E402
from hybrid.stage2 import OllamaArbiter  # noqa: E402

OUT = ARTIFACTS / "bench"


def summarize(rows, records):
    m = score_slice([r["labels"] for r in rows], [rec["labels"] for rec in records], [r["mcId"] for r in rows])
    seconds = [rec["seconds"] for rec in records]
    calls = [rec for rec in records if rec["escalated"] and (rec["llm"] is not None or rec["llm_error"])]
    return {"rows": len(rows), "micro_f1": m["micro_f1"], "precision": m["precision"], "recall": m["recall"],
            "exact_match": m["exact_match"], "empty_rows": m["empty_rows"], "empty_fp_rows": m["empty_fp_rows"],
            "llm_share": len(calls) / max(len(rows), 1),
            "llm_errors": sum(rec["llm_error"] is not None for rec in records),
            "latency_p50_s": float(np.median(seconds)), "latency_p95_s": float(np.quantile(seconds, 0.95)),
            "input_tokens": int(sum(rec["input_tokens"] for rec in records)),
            "output_tokens": int(sum(rec["output_tokens"] for rec in records))}


def pct(x):
    return f"{100 * x:.2f}%"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", default="stage1,cascade,llm")
    ap.add_argument("--model", default="avibe-q8")
    ap.add_argument("--seo", default="off", choices=["off", "on"])
    ap.add_argument("--sets", default="hard,stress_suite", help="test_gold is optional, gold_core already copies it")
    ap.add_argument("--skip-slices", default="gold_core", help="stress slices to drop; '' keeps all")
    ap.add_argument("--llm-sets", default="hard,stress_suite", help="sets for the pure-LLM variant")
    ap.add_argument("--llm-stress-slices", default="context_traps_and_negations,hard_empty,non_101_source_mcid",
                    help="stress slices for the pure-LLM variant; 'all' for every slice")
    ap.add_argument("--keep-dev-copies", action="store_true",
                    help="keep the 14 stress rows copied from dev (they are used for calibration)")
    args = ap.parse_args()

    model, cfg = load()
    sets = {k: v for k, v in eval_sets().items() if k in args.sets.split(",")}
    if "stress_suite" in sets and not args.keep_dev_copies:
        sets["stress_suite"] = [r for r in sets["stress_suite"] if r.get("origin_split") != "dev"]
    if "stress_suite" in sets and args.skip_slices:
        sets["stress_suite"] = [r for r in sets["stress_suite"] if r["slice"] not in args.skip_slices.split(",")]
    OUT.mkdir(parents=True, exist_ok=True)
    summary_path = OUT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    arbiter = None
    for variant in args.variants.split(","):
        name = variant if variant == "stage1" else f"{variant}-{args.model}-seo_{args.seo}"
        if variant != "stage1" and arbiter is None:
            arbiter = OllamaArbiter(args.model, args.seo)
        for set_name, rows in sets.items():
            if variant == "llm":
                if set_name not in args.llm_sets.split(","):
                    continue
                if set_name == "stress_suite" and args.llm_stress_slices != "all":
                    rows = [r for r in rows if r["slice"] in args.llm_stress_slices.split(",")]
            print(f"== {name} on {set_name} ({len(rows)} rows)", flush=True)
            mode, run_cfg = variant, cfg
            if variant == "cascade_notrig":
                mode, run_cfg = "cascade", replace(cfg, use_triggers=False)
            records = run(rows, model, run_cfg, mode, arbiter if variant != "stage1" else None, log_every=25)
            with (OUT / f"{name}__{set_name}.jsonl").open("w", encoding="utf-8") as stream:
                for rec in records:
                    stream.write(json.dumps(rec, ensure_ascii=False) + "\n")
            result = summarize(rows, records)
            if set_name == "stress_suite":
                if len({r["slice"] for r in rows}) == 7:
                    evaluate_and_report(rows, [rec["labels"] for rec in records], name, OUT / f"{name}__stress_report.md", STRESS)
                result["slices"] = {}
                for s in SLICE_ORDER + ["mentor_borderline_switches"]:
                    idx = [i for i, r in enumerate(rows) if r["slice"] == s]
                    if not idx:
                        continue
                    result["slices"][s] = summarize([rows[i] for i in idx], [records[i] for i in idx])
            summary.setdefault(name, {})[set_name] = result
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            print({k: round(v, 4) if isinstance(v, float) else v for k, v in result.items() if k != "slices"}, flush=True)
    if arbiter is not None:
        arbiter.unload()
    write_markdown(summary)


def write_markdown(summary):
    lines = ["# Каскад Stage 1 → Router → локальная LLM", "",
             "| Вариант | Набор | Строк | Micro-F1 | Exact Match | FP на пустых | % LLM | p50, с | p95, с | Токены вход/выход |",
             "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |"]
    for name, sets in summary.items():
        for set_name, m in sets.items():
            empty = f"{m['empty_fp_rows']}/{m['empty_rows']}" if m["empty_rows"] else "—"
            lines.append(f"| {name} | {set_name} | {m['rows']} | {pct(m['micro_f1'])} | {pct(m['exact_match'])} | {empty} | "
                         f"{pct(m['llm_share'])} | {m['latency_p50_s']:.2f} | {m['latency_p95_s']:.2f} | "
                         f"{m['input_tokens']}/{m['output_tokens']} |")
    lines += ["", "## Срезы stress_suite (Micro-F1 / Exact Match / % LLM)", ""]
    names = [n for n in summary if "stress_suite" in summary[n]]
    lines.append("| Срез | " + " | ".join(names) + " |")
    lines.append("| --- |" + " ---: |" * len(names))
    for s in SLICE_ORDER + ["mentor_borderline_switches"]:
        cells = []
        for n in names:
            m = summary[n]["stress_suite"]["slices"].get(s)
            if m is None:
                cells.append("—")
                continue
            cells.append(f"{pct(m['micro_f1'])} / {pct(m['exact_match'])} / {pct(m['llm_share'])}")
        lines.append(f"| `{s}` | " + " | ".join(cells) + " |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
