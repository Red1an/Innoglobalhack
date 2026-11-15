"""Combination policies of Stage 1 and a full LLM answer, simulated from saved answers (no new calls).

The cascade lets the LLM decide only uncertain categories. Here the LLM answers all categories and
Stage 1 acts as a safety net in the opposite direction:
  llm             LLM answer as is
  intersection    Stage 1 ∩ LLM
  union           Stage 1 ∪ LLM
  veto_lo         LLM, minus categories with Stage 1 p < x (Stage 1 is sure the category is absent)
  add_hi          LLM, plus categories with Stage 1 p > y and no blocking trigger
  veto_lo+add_hi  both
Thresholds x and y are reported as a grid; they are not tuned on dev, so the grid is diagnostic.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_stress_suite import score_slice  # noqa: E402
from hybrid.cascade import ARTIFACTS  # noqa: E402
from hybrid.normalize import normalize  # noqa: E402
from hybrid.reference import bench_sets, key, reference_runs  # noqa: E402
from hybrid.triggers import BLOCKING, detect  # noqa: E402

BENCH = ARTIFACTS / "bench"
TARGETS = [102, 103, 104, 105, 106]


def read(path):
    return {r["row_id"]: r for r in map(json.loads, path.open(encoding="utf-8"))}


def policy(name, llm, p, risky, mc, x, y):
    llm = set(llm) - {mc}
    if name == "llm":
        out = llm
    elif name == "intersection":
        out = llm & {c for c, v in zip(TARGETS, p) if v >= 0.35}
    elif name == "union":
        out = llm | {c for c, v in zip(TARGETS, p) if v >= 0.35}
    else:
        out = set(llm)
        if "veto_lo" in name:
            out -= {c for c, v in zip(TARGETS, p) if v < x}
        if "add_hi" in name:
            out |= {c for c, v in zip(TARGETS, p) if v > y and c not in risky}
    return sorted(out - {mc})


def main():
    sets, refs = bench_sets(), reference_runs()
    arbiters = sys.argv[1].split(",") if len(sys.argv) > 1 else ["llm-avibe-q8-seo_off", "avibe_static_v2", "qwen235_promptB"]
    out = {}
    for set_name, rows in sets.items():
        s1 = read(BENCH / f"stage1__{set_name}.jsonl")
        for arb in arbiters:
            path = BENCH / f"{arb}__{set_name}.jsonl"
            if path.exists():
                answers = {i: rec["labels"] for i, rec in read(path).items()}
            elif arb in refs and all(key(r["description"]) in refs[arb] for r in rows):
                answers = {r["row_id"]: refs[arb][key(r["description"])] for r in rows}
            else:
                continue
            sub = [r for r in rows if r["row_id"] in answers]
            risky = {r["row_id"]: set().union(*[c for n, c in detect(normalize(r["description"])).items() if n in BLOCKING] or [set()])
                     for r in sub}
            res = {}
            for name in ("llm", "intersection", "union", "veto_lo", "add_hi", "veto_lo+add_hi"):
                xs = [0.02, 0.05, 0.1] if "veto_lo" in name else [None]
                ys = [0.9, 0.95, 0.98] if "add_hi" in name else [None]
                for x in xs:
                    for y in ys:
                        preds = [policy(name, answers[r["row_id"]], [s1[r["row_id"]]["probs"][str(c)] for c in TARGETS],
                                        risky[r["row_id"]], r["mcId"], x, y) for r in sub]
                        m = score_slice([r["labels"] for r in sub], preds, [r["mcId"] for r in sub])
                        tag = name + (f" x={x}" if x else "") + (f" y={y}" if y else "")
                        res[tag] = {"f1": m["micro_f1"], "exact": m["exact_match"], "empty_fp": m["empty_fp_rows"],
                                    "empty": m["empty_rows"]}
            out.setdefault(arb, {})[set_name] = {"rows": len(sub), "policies": res}
            print(f"== {arb} / {set_name} ({len(sub)})")
            for tag, m in res.items():
                print(f"   {tag:28s} F1 {m['f1']:.4f} EM {m['exact']:.3f} FP empty {m['empty_fp']}/{m['empty']}")
    (BENCH / "policies.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
