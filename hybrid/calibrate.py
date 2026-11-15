"""Fit Stage 1 on train_fit and calibrate t_mid, t_low, t_high strictly on dev.

The arbiter is run once on dev rows escalated by the widest band. Narrower bands are simulated
from those answers: the arbiter's decision for a category in the narrower contested set is reused
from the wide run, which approximates the prompt that the narrower set would have produced.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hybrid.cascade import ARTIFACTS, run, save  # noqa: E402
from hybrid.data import TARGETS, reserved_for_eval, split_train  # noqa: E402
from hybrid.normalize import normalize  # noqa: E402
from hybrid.router import RouterConfig, merge, route  # noqa: E402
from hybrid.stage1 import Blend, ExternalScores, LinearStage1  # noqa: E402
from hybrid.stage2 import OllamaArbiter  # noqa: E402


def micro(rows, preds):
    tp = fp = fn = em = 0
    for r, p in zip(rows, preds):
        t, p = set(r["labels"]) - {r["mcId"]}, set(p) - {r["mcId"]}
        tp, fp, fn, em = tp + len(t & p), fp + len(p - t), fn + len(t - p), em + (t == p)
    return 2 * tp / max(2 * tp + fp + fn, 1), em / max(len(rows), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="avibe-q8")
    ap.add_argument("--seo", default="off", choices=["off", "on"])
    ap.add_argument("--wide-low", type=float, default=0.10)
    ap.add_argument("--wide-high", type=float, default=0.90)
    ap.add_argument("--budget", type=float, default=0.25, help="max share of rows sent to the LLM on dev")
    ap.add_argument("--no-llm", action="store_true", help="only fit Stage 1 and t_mid")
    ap.add_argument("--external-scores", type=Path,
                    help="JSONL of p_102..p_106 by text_hash (e.g. RuModernBERT) for dev, tests and the private file")
    ap.add_argument("--blend-weight", type=float, default=0.5, help="weight of the linear model in the blend")
    args = ap.parse_args()

    train_fit, dev = split_train(reserved_rows=reserved_for_eval())
    print(f"train_fit={len(train_fit)} dev={len(dev)}", flush=True)
    model = LinearStage1().fit(train_fit)
    if args.external_scores:
        model = Blend([model, ExternalScores(args.external_scores)], [args.blend_weight, 1 - args.blend_weight])
    probs = model.predict_proba([r["description"] for r in dev])
    grid_mid = {round(t, 2): micro(dev, [[c for c, v in zip(TARGETS, p) if v >= t] for p in probs])[0]
                for t in np.arange(0.30, 0.71, 0.05)}
    t_mid = max(grid_mid, key=lambda t: (grid_mid[t], -abs(t - 0.5)))
    print(f"dev Stage 1 F1 by t_mid: {grid_mid}; chosen {t_mid}", flush=True)
    base = RouterConfig(t_mid=t_mid)
    if args.no_llm:
        save(model, base, {"dev": {"rows": len(dev), "stage1_f1_by_t_mid": grid_mid}})
        return

    wide = RouterConfig(args.wide_low, args.wide_high, t_mid)
    arbiter = OllamaArbiter(args.model, args.seo)
    records = run(dev, model, wide, "cascade", arbiter, probs=probs, log_every=25)
    arbiter.unload()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with (ARTIFACTS / f"dev_wide_{args.model}.jsonl").open("w", encoding="utf-8") as stream:
        for rec in records:
            stream.write(json.dumps(rec, ensure_ascii=False) + "\n")

    texts = [normalize(r["description"]) for r in dev]
    grid = []
    lows = [x for x in np.round(np.arange(args.wide_low, t_mid + 1e-9, 0.05), 2)]
    highs = [x for x in np.round(np.arange(t_mid, args.wide_high + 1e-9, 0.05), 2)]
    for trig in (True, False):
        for lo in lows:
            for hi in highs:
                cfg = RouterConfig(float(lo), float(hi), t_mid, use_triggers=trig)
                preds, esc = [], 0
                for row, p, text, wrec in zip(dev, probs, texts, records):
                    rt = route(p, row["mcId"], text, cfg)
                    esc += rt.escalate
                    answer = wrec["llm"] if wrec["llm"] is not None else [c for c in rt.contested if c in rt.stage1]
                    preds.append(merge(rt, answer))
                f1, em = micro(dev, preds)
                grid.append({"t_low": float(lo), "t_high": float(hi), "triggers": trig,
                             "f1": f1, "exact": em, "llm_share": esc / len(dev)})
    s1_f1, s1_em = micro(dev, [[c for c, v in zip(TARGETS, p) if v >= t_mid] for p in probs])
    feasible = [g for g in grid if g["llm_share"] <= args.budget and g["triggers"]]
    best = max(feasible, key=lambda g: (round(g["f1"], 4), -g["llm_share"]))
    cfg = RouterConfig(best["t_low"], best["t_high"], t_mid, use_triggers=True)
    errors = sum(r["llm_error"] is not None for r in records if r["escalated"])
    save(model, cfg, {"model": args.model, "seo": args.seo, "budget": args.budget,
                      "dev": {"rows": len(dev), "stage1_f1": s1_f1, "stage1_exact": s1_em,
                              "stage1_f1_by_t_mid": grid_mid, "wide_band": [args.wide_low, args.wide_high],
                              "wide_llm_calls": sum(r["escalated"] for r in records), "wide_llm_errors": errors,
                              "chosen": best},
                      "grid": grid})
    print(f"dev Stage 1: F1={s1_f1:.4f} EM={s1_em:.3f}")
    print("chosen:", best)
    for g in sorted(grid, key=lambda g: -g["f1"])[:10]:
        print(g)


if __name__ == "__main__":
    main()
