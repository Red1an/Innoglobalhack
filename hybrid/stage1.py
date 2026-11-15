"""Stage 1: locally trained classifier that returns per-category probabilities p_102..p_106."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .data import TARGETS
from .normalize import normalize


def text_hash(text):
    return hashlib.sha256(normalize(text).encode()).hexdigest()[:16]


class LinearStage1:
    """One binary LogisticRegression per category over char_wb and optional word TF-IDF features."""

    def __init__(self, features="char+word", C=4.0, normalize_text=True):
        self.features, self.C, self.normalize_text = features, C, normalize_text

    def _prep(self, texts):
        return [normalize(t) if self.normalize_text else t for t in texts]

    def _x(self, texts, fit=False):
        parts = []
        for vec in self.vectorizers:
            parts.append(vec.fit_transform(texts) if fit else vec.transform(texts))
        return hstack(parts).tocsr()

    def fit(self, rows):
        texts = self._prep([r["description"] for r in rows])
        self.vectorizers = [TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=120000,
                                            sublinear_tf=True, dtype=np.float32)]
        if self.features == "char-legacy":
            self.vectorizers = [TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=3, max_features=50000,
                                                sublinear_tf=True, dtype=np.float32)]
        if self.features == "char+word":
            self.vectorizers.append(TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=60000,
                                                    sublinear_tf=True, dtype=np.float32,
                                                    token_pattern=r"(?u)\b\w+\b"))
        x = self._x(texts, fit=True)
        self.models = {}
        for c in TARGETS:
            y = np.array([int(c in r["labels"]) for r in rows])
            self.models[c] = LogisticRegression(C=self.C, solver="liblinear", random_state=42).fit(x, y)
        return self

    def predict_proba(self, texts):
        x = self._x(self._prep(texts))
        return np.stack([self.models[c].predict_proba(x)[:, 1] for c in TARGETS], axis=1)


class ExternalScores:
    """Probabilities computed elsewhere (e.g. fine-tuned RuModernBERT in Colab).

    JSONL lines: {"text_hash": text_hash(description), "p_102": ..., ..., "p_106": ...}.
    """

    def __init__(self, path):
        self.table = {}
        with Path(path).open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    r = json.loads(line)
                    self.table[r["text_hash"]] = [float(r[f"p_{c}"]) for c in TARGETS]

    def predict_proba(self, texts):
        missing = [t for t in texts if text_hash(t) not in self.table]
        if missing:
            raise KeyError(f"{len(missing)} texts have no external scores; recompute them for this file")
        return np.array([self.table[text_hash(t)] for t in texts])


class Blend:
    def __init__(self, members, weights):
        self.members, self.weights = members, np.asarray(weights, dtype=float) / np.sum(weights)

    def predict_proba(self, texts):
        return sum(w * m.predict_proba(texts) for m, w in zip(self.members, self.weights))


def dump_scores(model, rows, path):
    probs = model.predict_proba([r["description"] for r in rows])
    with Path(path).open("w", encoding="utf-8") as stream:
        for r, p in zip(rows, probs):
            stream.write(json.dumps({"text_hash": text_hash(r["description"]),
                                     **{f"p_{c}": float(v) for c, v in zip(TARGETS, p)}}) + "\n")
