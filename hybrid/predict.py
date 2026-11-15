"""Batch inference for the private file: Stage 1 → Router → local LLM arbiter.

Input: .parquet, .jsonl or .csv with a description column, optional mcId (default 101) and optional id column.
Output folder:
  predictions.csv   id, mcId, labels (comma separated), stage1 labels, p_102..p_106, routing info
  answers.jsonl     {"id": ..., "drafts": [{"mcId": ..., "title": null, "text": null}, ...]} per row
  disagreements.csv rows where the arbiter changed the Stage 1 decision, or fell back, for manual review
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hybrid.cascade import load, run  # noqa: E402
from hybrid.data import TARGETS, read_any  # noqa: E402
from hybrid.stage2 import OllamaArbiter  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--id-column", default=None, help="default: first of row_id, id, item_id; else row index")
    ap.add_argument("--default-mcid", type=int, default=101)
    ap.add_argument("--model", default="avibe-q8")
    ap.add_argument("--seo", default="off", choices=["off", "on"], help="guide_v2 is off; on is the reserve policy")
    ap.add_argument("--no-llm", action="store_true", help="Stage 1 only")
    args = ap.parse_args()

    raw = read_any(args.input)
    id_col = args.id_column or next((c for c in ("row_id", "id", "item_id") if raw and c in raw[0]), None)
    rows = []
    for i, r in enumerate(raw):
        mc = r.get("mcId")
        mc = args.default_mcid if mc is None or mc != mc else int(mc)
        rows.append({"row_id": r[id_col] if id_col else i, "mcId": mc, "description": str(r["description"] or "")})
    model, cfg = load()
    arbiter = None if args.no_llm else OllamaArbiter(args.model, args.seo)
    start = time.perf_counter()
    records = run(rows, model, cfg, "cascade", arbiter, log_every=25)
    if arbiter is not None:
        arbiter.unload()
    elapsed = time.perf_counter() - start

    args.out.mkdir(parents=True, exist_ok=True)
    fields = ["id", "mcId", "labels", "stage1", *[f"p_{c}" for c in TARGETS], "contested", "llm", "fallback", "triggers"]
    with (args.out / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=fields)
        w.writeheader()
        for rec in records:
            w.writerow({"id": rec["row_id"], "mcId": rec["mcId"], "labels": ",".join(map(str, rec["labels"])),
                        "stage1": ",".join(map(str, rec["stage1"])),
                        **{f"p_{c}": rec["probs"][str(c)] for c in TARGETS},
                        "contested": ",".join(map(str, rec["contested"])),
                        "llm": "" if rec["llm"] is None else ",".join(map(str, rec["llm"])),
                        "fallback": int(rec["fallback"]), "triggers": ";".join(rec["triggers"])})
    with (args.out / "answers.jsonl").open("w", encoding="utf-8") as stream:
        for rec in records:
            drafts = [{"mcId": c, "title": None, "text": None} for c in rec["labels"] if c != rec["mcId"]]
            stream.write(json.dumps({"id": rec["row_id"], "drafts": drafts}, ensure_ascii=False, default=str) + "\n")
    disputed = [(row, rec) for row, rec in zip(rows, records)
                if rec["fallback"] or (rec["llm"] is not None and set(rec["labels"]) != set(rec["stage1"]))]
    with (args.out / "disagreements.csv").open("w", encoding="utf-8", newline="") as stream:
        w = csv.writer(stream)
        w.writerow(["id", "mcId", "stage1", "final", "added_by_llm", "removed_by_llm", "contested", "probs",
                    "triggers", "fallback", "description"])
        for row, rec in disputed:
            s1, fin = set(rec["stage1"]), set(rec["labels"])
            w.writerow([rec["row_id"], rec["mcId"], ",".join(map(str, sorted(s1))), ",".join(map(str, sorted(fin))),
                        ",".join(map(str, sorted(fin - s1))), ",".join(map(str, sorted(s1 - fin))),
                        ",".join(map(str, rec["contested"])),
                        " ".join(f"{c}:{rec['probs'][str(c)]:.2f}" for c in rec["contested"] if str(c) in rec["probs"]),
                        ";".join(rec["triggers"]), int(rec["fallback"]), row["description"]])
    calls = sum(rec["escalated"] for rec in records) if arbiter else 0
    print(f"{len(rows)} rows in {elapsed:.1f}s; LLM calls {calls} ({100 * calls / max(len(rows), 1):.1f}%); "
          f"fallbacks {sum(rec['fallback'] for rec in records)}; disagreements {len(disputed)}")
    print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
