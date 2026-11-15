"""Evaluate a classifier or predictions file across the stress suite slices."""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "experiments"))

from bench import canonical, digest, gold_overlap_key, key  # noqa: E402

CLASSES = [101, 102, 103, 104, 105, 106]
SUITE_PATH = ROOT / "data" / "test" / "stress_suite_v1" / "stress_suite_v1.jsonl"
TRAIN_PATH = ROOT / "data" / "processed" / "train_guide_v1" / "train.parquet"
GOLD_PATH = ROOT / "data" / "test" / "test_guide_v1" / "test_gold.parquet"
DEFAULT_REPORT = ROOT / "output" / "stress_suite_v1_report.md"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def build_leak_free_train(suite_rows: list[dict], train_path: Path = TRAIN_PATH,
                          gold_path: Path = GOLD_PATH) -> list[dict]:
    """Build train_fit excluding test_gold, dev split, and any suite overlaps."""
    raw_train = pq.read_table(train_path).to_pylist()
    raw_gold = pq.read_table(gold_path).to_pylist()
    normalize = gold_overlap_key()

    reserved_keys = {key(r["description"]) for r in raw_gold}
    reserved_norm = {normalize(r["description"]) for r in raw_gold}
    for r in suite_rows:
        if r["origin_split"] != "dev":
            reserved_keys.add(key(r["description"]))
            if nk := normalize(r["description"]):
                reserved_norm.add(nk)

    groups = defaultdict(list)
    for src in raw_train:
        row = canonical(src, 101)
        tk = key(row["description"])
        if tk in reserved_keys or normalize(src["description"]) in reserved_norm:
            continue
        groups[(tk, row["mcId"])].append((row, src))

    train_fit, synthetic, dev_item_ids = [], [], set()
    for (tk, _), records in sorted(groups.items()):
        if len({tuple(r["labels"]) for r, _ in records}) > 1:
            continue
        row, src = records[0]
        if src.get("origin") == "synthetic" and src.get("source_item_id") is not None:
            synthetic.append((row, src["source_item_id"]))
            continue
        if int(digest(tk)[:8], 16) % 5 == 0:
            dev_item_ids.update(raw["item_id"] for _, raw in records if raw.get("item_id") is not None)
        else:
            train_fit.append(row)
    train_fit.extend(row for row, sid in synthetic if sid not in dev_item_ids)
    return train_fit


def predict_char_lr(train_fit: list[dict], suite_rows: list[dict], threshold: float = 0.5) -> list[list[int]]:
    """Fit Char-LR for the categories represented by positive training examples."""
    classes = sorted({label for row in train_fit for label in row["labels"]})
    if not classes:
        return [[] for _ in suite_rows]
    if len(classes) == 1:
        y_train = np.array([int(classes[0] in row["labels"]) for row in train_fit], dtype=np.int32)
    else:
        y_train = np.array([[int(c in r["labels"]) for c in classes] for r in train_fit], dtype=np.int32)
    pipe = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=3, max_features=50000, sublinear_tf=True, dtype=np.float32),
        OneVsRestClassifier(LogisticRegression(C=4, solver="liblinear", random_state=42)),
    )
    t0 = time.perf_counter()
    pipe.fit([r["description"] for r in train_fit], y_train)
    fit_s = time.perf_counter() - t0
    probs = pipe.predict_proba([r["description"] for r in suite_rows])
    if len(classes) == 1:
        probs = probs[:, 1:]
    print(f"char-LR trained on {len(train_fit)} rows in {fit_s:.2f}s (threshold={threshold:.2f})")
    preds = []
    for row, p_vec in zip(suite_rows, probs):
        preds.append([c for c, p in zip(classes, p_vec) if p >= threshold and c != row["mcId"]])
    return preds


def to_matrix(label_lists: list[list[int]]) -> np.ndarray:
    return np.array([[int(c in labs) for c in CLASSES] for labs in label_lists], dtype=np.int32)


def glue_rate(truth_lists: list[list[int]], pred_lists: list[list[int]], a: int, b: int) -> tuple[int, int]:
    """Count how often b is falsely predicted when a is in truth and b is NOT in truth."""
    eligible = [(t, p) for t, p in zip(truth_lists, pred_lists) if a in t and b not in t]
    if not eligible:
        return 0, 0
    glued = sum(1 for _, p in eligible if b in p)
    return glued, len(eligible)


def additional_labels(labels, source):
    labels = set(labels)
    if labels - set(CLASSES) or source not in CLASSES:
        raise ValueError(f"Unexpected labels or source category: {sorted(labels)}, {source}")
    return sorted(labels - {source})


def score_slice(truth_lists: list[list[int]], pred_lists: list[list[int]], sources=None) -> dict:
    if sources is not None:
        truth_lists = [additional_labels(labels, source) for labels, source in zip(truth_lists, sources, strict=True)]
        pred_lists = [additional_labels(labels, source) for labels, source in zip(pred_lists, sources, strict=True)]
    yt = to_matrix(truth_lists)
    yp = to_matrix(pred_lists)
    p, r, f1, _ = precision_recall_fscore_support(yt, yp, average="micro", zero_division=0)
    em = accuracy_score(yt, yp)

    empty_idx = [i for i, t in enumerate(truth_lists) if len(t) == 0]
    empty_fp_rows = sum(1 for i in empty_idx if len(pred_lists[i]) > 0)
    empty_fp_labels = sum(len(pred_lists[i]) for i in empty_idx)

    g104_105 = glue_rate(truth_lists, pred_lists, 104, 105)
    g105_104 = glue_rate(truth_lists, pred_lists, 105, 104)
    g102_106 = glue_rate(truth_lists, pred_lists, 102, 106)
    g_tot_num = g104_105[0] + g105_104[0] + g102_106[0]
    g_tot_den = g104_105[1] + g105_104[1] + g102_106[1]

    return {
        "rows": len(truth_lists),
        "micro_f1": float(f1),
        "precision": float(p),
        "recall": float(r),
        "exact_match": float(em),
        "empty_rows": len(empty_idx),
        "empty_fp_rows": empty_fp_rows,
        "empty_fp_labels": empty_fp_labels,
        "glue_104_105": g104_105,
        "glue_105_104": g105_104,
        "glue_102_106": g102_106,
        "glue_total": (g_tot_num, g_tot_den),
        "per_class": {
            str(cat): {
                "tp": int(np.sum((yt[:, i] == 1) & (yp[:, i] == 1))),
                "fn": int(np.sum((yt[:, i] == 1) & (yp[:, i] == 0))),
                "fp": int(np.sum((yt[:, i] == 0) & (yp[:, i] == 1))),
            }
            for i, cat in enumerate(CLASSES)
        },
    }


def fmt_pct(val: float) -> str:
    return f"{val * 100:.2f}%"


def fmt_empty(m: dict) -> str:
    if m["empty_rows"] == 0:
        return "—"
    return f"{m['empty_fp_rows']}/{m['empty_rows']} ({100 * m['empty_fp_rows'] / m['empty_rows']:.1f}%, {m['empty_fp_labels']} lbl)"


def fmt_glue(m: dict) -> str:
    num, den = m["glue_total"]
    if den == 0:
        return "—"
    parts = []
    for tag, key_name in [("104→105", "glue_104_105"), ("105→104", "glue_105_104"), ("102→106", "glue_102_106")]:
        a, b = m[key_name]
        if b > 0:
            parts.append(f"{tag}:{a}/{b}")
    return f"{100 * num / den:.1f}% ({', '.join(parts)})"


SLICE_ORDER = [
    "gold_core",
    "hard_empty",
    "decoupled_solo_and_pairs",
    "context_traps_and_negations",
    "dirty_text_and_homoglyphs",
    "non_101_source_mcid",
]


def evaluate_and_report(suite_rows: list[dict], preds: list[list[int]], model_name: str,
                        report_path: Path, suite_path: Path = SUITE_PATH) -> None:
    sources = [r["mcId"] for r in suite_rows]
    preds = [additional_labels(labels, source) for labels, source in zip(preds, sources, strict=True)]
    alternate_seo = all("labels_seo" in r for r in suite_rows)
    lines = [
        f"# Отчёт по стресс-бенчмарку `{suite_path.parent.name}/{suite_path.name}` — `{model_name}`",
        "",
        f"Всего строк: **{len(suite_rows)}** (7 диагностических срезов).",
        "",
        "| Срез | Строк | Micro-F1 | Precision | Recall | Exact Match | FP на пустых | Склейка пар (104↔105, 102→106) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]

    def add_row(label: str, idx_list: list[int], truth_key: str = "labels") -> dict:
        t = [suite_rows[i][truth_key] for i in idx_list]
        p = [preds[i] for i in idx_list]
        m = score_slice(t, p, [sources[i] for i in idx_list])
        lines.append(
            f"| {label} | {m['rows']} | {fmt_pct(m['micro_f1'])} | {fmt_pct(m['precision'])} | "
            f"{fmt_pct(m['recall'])} | {fmt_pct(m['exact_match'])} | {fmt_empty(m)} | {fmt_glue(m)} |"
        )
        return m

    for sname in SLICE_ORDER:
        idxs = [i for i, r in enumerate(suite_rows) if r["slice"] == sname]
        add_row(f"`{sname}`", idxs, "labels")

    mentor_idxs = [i for i, r in enumerate(suite_rows) if r["slice"] == "mentor_borderline_switches"]
    add_row("`mentor_borderline_switches`", mentor_idxs)
    if alternate_seo:
        add_row("`mentor_borderline` (with_seo_tags)", mentor_idxs, "labels_seo")

    stress_only_idxs = [i for i, r in enumerate(suite_rows) if r["slice"] != "gold_core"]
    add_row("**ИТОГО stress без gold_core**", stress_only_idxs)
    add_row("**ИТОГО весь suite**", list(range(len(suite_rows))))

    # Detailed subslice breakdown
    lines.extend(["", "## Разбивка по подсрезам (Exact Match / Micro-F1)", "", "| Срез / подсрез | Строк | Micro-F1 | Exact Match | FP на пустых |", "| --- | ---: | ---: | ---: | --- |"])
    seen_pairs = []
    for r in suite_rows:
        pair = (r["slice"], r["subslice"])
        if pair not in seen_pairs:
            seen_pairs.append(pair)
    for sname, sub in seen_pairs:
        idxs = [i for i, r in enumerate(suite_rows) if r["slice"] == sname and r["subslice"] == sub]
        t_key = "labels_seo" if alternate_seo and sub == "seo_tags_only" else "labels"
        m = score_slice([suite_rows[i][t_key] for i in idxs], [preds[i] for i in idxs], [sources[i] for i in idxs])
        tag = f"`{sname}/{sub}`" + (" (seo)" if t_key == "labels_seo" else "")
        lines.append(f"| {tag} | {m['rows']} | {fmt_pct(m['micro_f1'])} | {fmt_pct(m['exact_match'])} | {fmt_empty(m)} |")

    # Sample errors per slice
    lines.extend(["", "## Примеры ошибок по срезам (до 3 на срез)", ""])
    for sname in SLICE_ORDER + ["mentor_borderline_switches"]:
        errs = [
            (suite_rows[i], preds[i])
            for i, r in enumerate(suite_rows)
            if r["slice"] == sname and preds[i] != additional_labels(r["labels"], sources[i])
        ]
        lines.append(f"### `{sname}` (ошибок: {len(errs)})")
        for r, p in errs[:3]:
            snippet = r["description"].replace("\n", " ")[:140]
            lines.append(f"- `{r['row_id']}` (`mcId={r['mcId']}`, `{r['subslice']}`): gold={r['labels']} pred={p} — *{r['note']}* | «{snippet}...»")
        lines.append("")

    report_text = "\n".join(lines)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")
    print("\n".join(lines[:18]))
    print(f"\nFull report saved to: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="char-lr", choices=["char-lr"])
    parser.add_argument("--suite", type=Path, default=SUITE_PATH)
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    parser.add_argument("--gold", type=Path, default=GOLD_PATH)
    parser.add_argument("--predictions", help="Optional JSONL with predictions ({row_id, labels} or aligned list)")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    suite_rows = load_jsonl(args.suite)
    if args.predictions:
        raw_preds = load_jsonl(Path(args.predictions))
        if raw_preds and "row_id" in raw_preds[0]:
            by_id = {r["row_id"]: sorted(set(r["labels"])) for r in raw_preds}
            preds = [by_id[r["row_id"]] for r in suite_rows]
        else:
            assert len(raw_preds) == len(suite_rows)
            preds = [sorted(set(r["labels"])) for r in raw_preds]
        model_name = Path(args.predictions).stem
    else:
        train_fit = build_leak_free_train(suite_rows, args.train, args.gold)
        preds = predict_char_lr(train_fit, suite_rows, threshold=args.threshold)
        model_name = f"char-lr (p>={args.threshold:.2f}, leak-free train_fit={len(train_fit)})"

    default_report = (DEFAULT_REPORT if args.suite.resolve() == SUITE_PATH
                      else args.suite.with_name(f"{args.suite.stem}_report.md"))
    evaluate_and_report(suite_rows, preds, model_name, args.report or default_report, args.suite)


if __name__ == "__main__":
    main()
