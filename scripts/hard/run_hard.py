import json
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/experiments"))
sys.path.insert(0, str(ROOT / "scripts/synth"))
from bench import PROMPT  # noqa: E402
from common import base_train, labels_of  # noqa: E402
from probe_charlr import char_lr  # noqa: E402

MC = [102, 103, 104, 105, 106]
OUT = ROOT / "output" / "hard_v0_predictions.json"
sheet = load_workbook(ROOT / "data/hard/hard_v0.xlsx")["разметка"]
rows = [(r[0].value, r[1].value, [mc for mc, v in zip(MC, r[2:7]) if v.value == 1]) for r in sheet.iter_rows(min_row=2)]
texts = [t for _, t, _ in rows]
preds = json.loads(OUT.read_text()) if OUT.exists() else {}


def to_labels(matrix):
    return [[mc for mc, yes in zip(MC, row) if yes] for row in matrix]


base = base_train()
preds["char-lr base"] = to_labels(char_lr(base.description.tolist(), base.labels.tolist()).predict(texts))
synth = pd.read_parquet(ROOT / "data/processed/train_synth_v1/train.parquet")
preds["char-lr synth_v1"] = to_labels(char_lr(synth.description.tolist(),
                                              [labels_of(x) for x in synth.found_microcategories]).predict(texts))

schema = {"type": "object", "required": ["labels"], "additionalProperties": False,
          "properties": {"labels": {"type": "array", "items": {"type": "integer", "enum": MC}}}}
for name, tag in [("avibe-q8", "hf.co/NightForger/avibe-GGUF:Q8_0"), ("t-lite-q8", "t-lite-it-2.1:q8_0")]:
    if name in preds:
        continue
    out = []
    for text in texts:
        payload = {"model": tag, "stream": False, "format": schema, "keep_alive": "10m",
                   "messages": [{"role": "system", "content": PROMPT},
                                {"role": "user", "content": json.dumps({"mcId": 101, "description": text}, ensure_ascii=False)}],
                   "options": {"temperature": 0, "seed": 42, "num_ctx": 4096, "num_predict": 64}}
        start = time.time()
        request = Request("http://localhost:11434/api/chat", data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
        answer = json.load(urlopen(request, timeout=600))["message"]["content"]
        try:
            out.append(sorted(set(json.loads(answer)["labels"])))
        except (ValueError, KeyError):
            out.append(["invalid"])
        print(name, len(out), out[-1], round(time.time() - start, 1), flush=True)
    preds[name] = out
    urlopen(Request("http://localhost:11434/api/generate", data=json.dumps({"model": tag, "keep_alive": 0}).encode()))
    OUT.write_text(json.dumps(preds, ensure_ascii=False))

OUT.write_text(json.dumps(preds, ensure_ascii=False))
gold = [g for _, _, g in rows]
for name, pred in preds.items():
    tp = sum(len(set(p) & set(g)) for p, g in zip(pred, gold))
    fp = sum(len(set(p) - set(g)) for p, g in zip(pred, gold))
    fn = sum(len(set(g) - set(p)) for p, g in zip(pred, gold))
    f1 = 2 * tp / (2 * tp + fp + fn)
    exact = sum(set(p) == set(g) for p, g in zip(pred, gold))
    print(f"{name}: micro F1 {f1:.3f}, точно {exact}/{len(gold)}, FP {fp}, FN {fn}")
print()
for i, (item, _, g) in enumerate(rows):
    print(item, "разметка", g, " | ".join(f"{n} {preds[n][i]}" for n in preds))
