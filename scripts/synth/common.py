import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[2]
SYNTH = ROOT / "data" / "synth" / "synth_v1"
MC = [102, 103, 104, 105, 106]
LABEL_NAMES = {102: "Штукатурные работы", 103: "Плиточные работы", 104: "Сантехника", 105: "Электрика",
               106: "Поклейка обоев и малярные работы"}
TEST_FILES = [ROOT / "data/test/test_v4.parquet", ROOT / "data/raw/public-test-dataset.parquet"]


def labels_of(entries):
    return sorted(int(entry["mcId"]) for entry in entries)


def entries_of(labels):
    return [{"mcId": int(mc), "label": LABEL_NAMES[int(mc)]} for mc in sorted(labels)]


def base_train():
    labels_102 = pd.read_parquet(ROOT / "data/processed/train_labels_102_v1/train.parquet")
    relabel = pd.read_parquet(ROOT / "data/processed/train_relabel_v1/train.parquet")
    assert (labels_102.item_id.values == relabel.item_id.values).all()
    rows = []
    for left, right in zip(labels_102.itertuples(), relabel.itertuples()):
        labels = [mc for mc in labels_of(right.found_microcategories) if mc != 102]
        if 102 in labels_of(left.found_microcategories):
            labels.append(102)
        rows.append(sorted(labels))
    frame = relabel.copy()
    frame["labels"] = rows
    return frame


def test_texts():
    texts = []
    for path in TEST_FILES:
        texts += pd.read_parquet(path)["description"].fillna("").tolist()
    return texts


class Similarity:
    def __init__(self, reference):
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)
        self.reference = self.vectorizer.fit_transform([text.lower() for text in reference])

    def max_to_reference(self, texts):
        matrix = self.vectorizer.transform([text.lower() for text in texts])
        return cosine_similarity(matrix, self.reference).max(axis=1)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
