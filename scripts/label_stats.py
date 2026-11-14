import json
from itertools import combinations
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "label_stats_2026-11-15"
MC = [102, 103, 104, 105, 106]

DATASETS = {
    "train_old": ROOT / "data/raw/public-train-annotated.jsonl",
    "train_new": ROOT / "data/processed/train_relabel_v1/train.parquet",
    "train_synth": ROOT / "data/processed/train_synth_v1/train.parquet",
    "test_old": ROOT / "data/test/test_v4.parquet",
    "test_new": ROOT / "data/test/test_v5.parquet",
}


def load(path):
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.open(encoding="utf-8")]
        labels = [[m["mcId"] for m in r["found_microcategories"]] for r in rows]
        return pd.DataFrame([[int(mc in ls) for mc in MC] for ls in labels], columns=MC)
    frame = pd.read_parquet(path)
    if "found_microcategories" in frame:
        labels = frame["found_microcategories"].map(lambda xs: [int(m["mcId"]) for m in xs])
        return pd.DataFrame([[int(mc in ls) for mc in MC] for ls in labels], columns=MC)
    return frame[[f"y_{mc}" for mc in MC]].set_axis(MC, axis=1).astype(int)


def stats(y):
    n = len(y)
    count = y.sum()
    cooc = y.T @ y
    conditional = cooc.div(count, axis=0)
    combos = y.apply(lambda r: "+".join(str(mc) for mc in MC if r[mc]) or "пусто", axis=1).value_counts()
    solo = {mc: int(((y[mc] == 1) & (y.sum(axis=1) == 1)).sum()) for mc in MC}
    pairs = []
    for a, b in combinations(MC, 2):
        both = int(cooc.at[a, b])
        pairs.append({
            "pair": f"{a}+{b}", "both": both,
            "a_without_b": int(count[a] - both), "b_without_a": int(count[b] - both),
            "p_b_given_a": round(both / count[a], 3) if count[a] else None,
            "p_a_given_b": round(both / count[b], 3) if count[b] else None,
            "lift": round(both * n / (count[a] * count[b]), 2) if count[a] and count[b] else None,
        })
    pairs = pd.DataFrame(pairs)
    summary = {
        "rows": n,
        "empty_share": round(float((y.sum(axis=1) == 0).mean()), 3),
        "mean_labels": round(float(y.sum(axis=1).mean()), 2),
        "labels_per_row": y.sum(axis=1).value_counts().sort_index().to_dict(),
        "label_share": (count / n).round(3).to_dict(),
        "solo": solo,
    }
    return summary, cooc, conditional.round(3), combos, pairs


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, path in DATASETS.items():
        summary, cooc, conditional, combos, pairs = stats(load(path))
        report[name] = summary
        cooc.to_csv(OUT / f"{name}_cooccurrence.csv")
        conditional.to_csv(OUT / f"{name}_conditional.csv")
        combos.rename_axis("combo").rename("rows").to_csv(OUT / f"{name}_combos.csv")
        pairs.to_csv(OUT / f"{name}_pairs.csv", index=False)
        print(f"\n===== {name}", json.dumps(summary, ensure_ascii=False, default=str))
        print("P(столбец | строка):\n", conditional.to_string())
        print(pairs.to_string(index=False))
        print("комбинации:\n", combos.head(15).to_string())
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
