"""guide_v2 datasets and the leak-free train_fit / dev split used by the cascade."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "experiments"))

from bench import canonical, digest, gold_overlap_key, key  # noqa: E402

TARGETS = [102, 103, 104, 105, 106]
ALL = [101, *TARGETS]
TRAIN = ROOT / "data/processed/train_guide_v2/train.parquet"
TEST_DIR = ROOT / "data/test/test_guide_v2"
GOLD = TEST_DIR / "test_gold.parquet"
HARD = TEST_DIR / "hard.jsonl"
STRESS = TEST_DIR / "stress_suite.jsonl"


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_any(path):
    path = Path(path)
    if path.suffix == ".parquet":
        return pq.read_table(path).to_pylist()
    if path.suffix == ".csv":
        import pandas as pd
        return pd.read_csv(path).to_dict("records")
    return read_jsonl(path)


def gold_rows(path=GOLD):
    rows = []
    for r in pq.read_table(path).to_pylist():
        labels = [c for c in ALL if r.get(f"y_{c}") == 1]
        rows.append({"row_id": f"gold_{r['row_id']}", "mcId": 101, "description": r["description"],
                     "labels": labels})
    return rows


def hard_rows(path=HARD):
    return [{"row_id": r["row_id"], "mcId": r["mcId"], "description": r["description"],
             "labels": sorted(r["labels"])} for r in read_jsonl(path)]


def stress_rows(path=STRESS):
    return read_jsonl(path)


def split_train(train_path=TRAIN, reserved_rows=()):
    """Same rule as scripts/eval_stress_suite.build_leak_free_train, but also returns dev.

    Exact and normalized overlaps with reserved test rows are dropped, label-conflicting
    duplicates are dropped, real rows go to dev when the sha256 bucket of their text is 0 (of 5),
    and synthetic rows derived from a dev source item are excluded from train_fit.
    """
    normalize = gold_overlap_key()
    reserved_keys = {key(r["description"]) for r in reserved_rows}
    reserved_norm = {n for r in reserved_rows if (n := normalize(r["description"]))}
    groups = defaultdict(list)
    for src in pq.read_table(train_path).to_pylist():
        row = canonical(src, 101)
        tk = key(row["description"])
        if tk in reserved_keys or normalize(src["description"]) in reserved_norm:
            continue
        groups[(tk, row["mcId"])].append((row, src))
    train_fit, dev, synthetic, dev_items = [], [], [], set()
    for (tk, _), records in sorted(groups.items()):
        if len({tuple(r["labels"]) for r, _ in records}) > 1:
            continue
        row, src = records[0]
        row = {**row, "origin": src.get("origin"), "item_id": src.get("item_id")}
        if src.get("origin") == "synthetic" and src.get("source_item_id") is not None:
            synthetic.append((row, src["source_item_id"]))
            continue
        if int(digest(tk)[:8], 16) % 5 == 0:
            dev.append(row)
            dev_items.update(raw["item_id"] for _, raw in records if raw.get("item_id") is not None)
        else:
            train_fit.append(row)
    train_fit.extend(row for row, sid in synthetic if sid not in dev_items)
    return train_fit, dev


def reserved_for_eval():
    """Every held-out text except stress rows that were deliberately copied from dev."""
    stress = [r for r in stress_rows() if r["origin_split"] != "dev"]
    return gold_rows() + hard_rows() + stress


def eval_sets():
    return {"test_gold": gold_rows(), "hard": hard_rows(), "stress_suite": stress_rows()}
