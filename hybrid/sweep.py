"""Simulate router thresholds from saved answers, without new LLM calls.

For every row we need Stage 1 probabilities (output/hybrid/bench/stage1__<set>.jsonl) and an arbiter answer
over all categories (pure-LLM run of hybrid/bench.py, or a saved static-prompt run from hybrid/reference.py).
A narrower contested set reuses the arbiter's decision per category, as in hybrid/calibrate.py.
Grid results on test sets are diagnostics only: thresholds are chosen on dev (calibration.json).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_stress_suite import score_slice  # noqa: E402
from hybrid.cascade import ARTIFACTS, load  # noqa: E402
from hybrid.normalize import normalize  # noqa: E402
from hybrid.reference import bench_sets, key, reference_runs  # noqa: E402
from hybrid.router import RouterConfig, merge, route  # noqa: E402

BENCH = ARTIFACTS / "bench"
TARGETS = [102, 103, 104, 105, 106]


def read(path):
    return {r["row_id"]: r for r in map(json.loads, path.open(encoding="utf-8"))}


def simulate(rows, probs, answers, cfg):
    preds, esc = [], 0
    for r in rows:
        rt = route(probs[r["row_id"]], r["mcId"], normalize(r["description"]), cfg)
        esc += rt.escalate
        preds.append(merge(rt, answers[r["row_id"]]))
    m = score_slice([r["labels"] for r in rows], preds, [r["mcId"] for r in rows])
    return m["micro_f1"], m["exact_match"], esc / len(rows), m["empty_fp_rows"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arbiters", default="llm-avibe-q8-seo_off,avibe_static_v2")
    args = ap.parse_args()
    _, base = load()
    sets = bench_sets()
    refs = reference_runs()
    out = {}
    for set_name, rows in sets.items():
        s1 = read(BENCH / f"stage1__{set_name}.jsonl")
        probs = {i: [s1[i]["probs"][str(c)] for c in TARGETS] for i in s1}
        for arb in args.arbiters.split(","):
            path = BENCH / f"{arb}__{set_name}.jsonl"
            if path.exists():
                recs = read(path)
                answers = {i: (rec["labels"] if rec["llm"] is not None else rec["stage1"]) for i, rec in recs.items()}
            elif arb in refs:
                answers = {r["row_id"]: refs[arb][key(r["description"])] for r in rows if key(r["description"]) in refs[arb]}
            else:
                continue
            sub = [r for r in rows if r["row_id"] in answers and r["row_id"] in probs]
            if len(sub) < 10 or (not path.exists() and len(sub) < len(rows)):
                continue
            s1_m = score_slice([r["labels"] for r in sub], [s1[r["row_id"]]["labels"] for r in sub], [r["mcId"] for r in sub])
            arb_m = score_slice([r["labels"] for r in sub], [answers[r["row_id"]] for r in sub], [r["mcId"] for r in sub])
            grid = []
            for trig in (True, False):
                for lo in np.round(np.arange(0.0, 0.51, 0.05), 2):
                    for hi in np.round(np.arange(0.35, 1.001, 0.05), 2):
                        if hi < lo:
                            continue
                        cfg = RouterConfig(float(lo), float(hi), base.t_mid, use_triggers=trig)
                        f1, em, share, fp = simulate(sub, probs, answers, cfg)
                        grid.append({"t_low": float(lo), "t_high": float(hi), "triggers": trig,
                                     "f1": f1, "exact": em, "llm_share": share, "empty_fp": fp})
            calib = simulate(sub, probs, answers, base)
            res = {"rows": len(sub), "stage1_f1": s1_m["micro_f1"], "arbiter_alone_f1": arb_m["micro_f1"],
                   "calibrated": dict(zip(["f1", "exact", "llm_share", "empty_fp"], calib)),
                   "best_any": max(grid, key=lambda g: (round(g["f1"], 4), -g["llm_share"])),
                   "best_le_50pct": max([g for g in grid if g["llm_share"] <= 0.5], key=lambda g: (round(g["f1"], 4), -g["llm_share"])),
                   "best_le_25pct": max([g for g in grid if g["llm_share"] <= 0.25], key=lambda g: (round(g["f1"], 4), -g["llm_share"])),
                   "grid": grid}
            out.setdefault(arb, {})[set_name] = res
            print(f"{arb} / {set_name} ({len(sub)}): Stage1 {s1_m['micro_f1']:.4f}, arbiter alone {arb_m['micro_f1']:.4f}, "
                  f"calibrated {calib[0]:.4f} @{calib[2]:.0%}")
            for k in ("best_le_25pct", "best_le_50pct", "best_any"):
                g = res[k]
                print(f"   {k}: {g['t_low']}-{g['t_high']} trig={g['triggers']} F1 {g['f1']:.4f} EM {g['exact']:.3f} @{g['llm_share']:.0%}")
    (BENCH / "sweep.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
