import random

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from common import MC, ROOT, base_train, labels_of
from probe_charlr import char_lr

SEED = 42
PROBE_SHARE = 0.2
COUPLED = [(104, 105), (105, 104), (102, 106), (102, 103), (103, 106)]


def matrix(labels):
    return np.array([[int(mc in row) for mc in MC] for row in labels])


def scores(model, texts, gold):
    pred = model.predict(texts)
    truth = matrix(gold)
    result = {"micro_f1": round(f1_score(truth, pred, average="micro", zero_division=0), 3),
              "exact": round(float((pred == truth).all(axis=1).mean()), 3)}
    for a, b in COUPLED:
        rows = [i for i, row in enumerate(gold) if a in row and b not in row]
        if rows:
            result[f"лишняя {b} при {a}"] = f"{int(pred[rows, MC.index(b)].sum())}/{len(rows)}"
    return result


def main():
    synth_train = pd.read_parquet(ROOT / "data/processed/train_synth_v1/train.parquet")
    synth = synth_train[synth_train.origin == "synthetic"].copy()
    synth["labels"] = synth.found_microcategories.map(labels_of)
    sources = sorted(set(-synth.source_line))
    random.Random(SEED).shuffle(sources)
    probe_sources = set(sources[:int(len(sources) * PROBE_SHARE)])
    probe = synth[(-synth.source_line).isin(probe_sources)]
    synth_fit = synth[~(-synth.source_line).isin(probe_sources)]

    base = base_train()
    base = base[~base.item_id.isin(probe_sources)]
    real = synth_train[(synth_train.origin == "real") & ~synth_train.item_id.isin(probe_sources)].copy()
    real["labels"] = real.found_microcategories.map(labels_of)
    variants = {
        "база (гибрид)": base,
        "база + сокращение «все пять»": real,
        "сокращение + синтез": pd.concat([real, synth_fit]),
    }
    gold = pd.read_parquet(ROOT / "data/test/test_gold.parquet")
    gold_labels = [[mc for mc in MC if row[f"y_{mc}"]] for _, row in gold.iterrows()]
    rows = []
    for name, frame in variants.items():
        model = char_lr(frame.description.tolist(), frame.labels.tolist())
        rows.append({"вариант": name, "строк": len(frame), "набор": "test_gold", **scores(model, gold.description.tolist(), gold_labels)})
        rows.append({"вариант": name, "строк": len(frame), "набор": f"зонд синтеза ({len(probe)})", **scores(model, probe.description.tolist(), probe.labels.tolist())})
    table = pd.DataFrame(rows).fillna("")
    print(table.to_string(index=False))
    table.to_csv(ROOT / "output/label_stats_2026-11-15/synth_v1_char_lr_comparison.csv", index=False)


if __name__ == "__main__":
    main()
