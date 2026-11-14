import sys
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline

from common import MC, SYNTH, base_train, read_jsonl
from pipeline import generations


def char_lr(texts, labels):
    model = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=3, max_features=50000,
                                          sublinear_tf=True, dtype=np.float32),
                          OneVsRestClassifier(LogisticRegression(C=4, solver="liblinear", random_state=42)))
    model.fit(texts, np.array([[int(mc in row) for mc in MC] for row in labels]))
    return model


def main():
    names = sys.argv[1:]
    sources = {task["source_item_id"] for name in names for task in read_jsonl(SYNTH / "tasks" / f"{name}.jsonl")}
    train = base_train()
    train = train[~train.item_id.isin(sources)]
    model = char_lr(train.description.tolist(), train.labels.tolist())
    rows = []
    for name in names:
        texts = generations(name)
        for task in read_jsonl(SYNTH / "tasks" / f"{name}.jsonl"):
            if task["task_id"] in texts:
                rows.append((task, texts[task["task_id"]]))
    predictions = model.predict([text for _, text in rows])
    extra, missed, exact = Counter(), Counter(), 0
    trap_hits = Counter()
    for (task, _), pred in zip(rows, predictions):
        predicted = {mc for mc, yes in zip(MC, pred) if yes}
        target = set(task["target_labels"])
        exact += predicted == target
        for mc in predicted - target:
            extra[mc] += 1
            if mc == task["trap_label"]:
                trap_hits[mc] += 1
        for mc in target - predicted:
            missed[mc] += 1
    traps = Counter(task["trap_label"] for task, _ in rows if task["trap_label"])
    print("текстов", len(rows), "точное совпадение", exact)
    print("лишние метки", dict(sorted(extra.items())))
    print("пропуски", dict(sorted(missed.items())))
    print("сработал на ловушку", {mc: f"{trap_hits[mc]} из {traps[mc]}" for mc in sorted(traps)})


if __name__ == "__main__":
    main()
