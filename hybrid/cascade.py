"""Stage 1 → Router → Stage 2 cascade over a list of rows {description, mcId}."""
import json
import pickle
import time
from dataclasses import asdict
from pathlib import Path

from .data import TARGETS
from .normalize import normalize
from .router import Route, RouterConfig, merge, route
from .triggers import detect

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "output" / "hybrid"
MODEL_PATH = ARTIFACTS / "stage1.pkl"
CONFIG_PATH = ARTIFACTS / "calibration.json"


def save(model, cfg: RouterConfig, extra=None):
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_bytes(pickle.dumps(model))
    CONFIG_PATH.write_text(json.dumps({"router": asdict(cfg), **(extra or {})}, ensure_ascii=False, indent=2), encoding="utf-8")


def load():
    model = pickle.loads(MODEL_PATH.read_bytes())
    cfg = RouterConfig(**json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["router"])
    return model, cfg


def stage1_scores(model, rows):
    start = time.perf_counter()
    probs = model.predict_proba([r["description"] for r in rows]) if rows else []
    per_row = (time.perf_counter() - start) / max(len(rows), 1)
    return probs, per_row


def run(rows, model, cfg: RouterConfig, mode="cascade", arbiter=None, probs=None, log_every=0):
    """mode: stage1 | llm | cascade. Returns one record per row."""
    if probs is None:
        probs, s1_seconds = stage1_scores(model, rows)
    else:
        s1_seconds = 0.0
    records = []
    calls = 0
    for i, (row, p) in enumerate(zip(rows, probs)):
        text = normalize(row["description"])
        mc = row.get("mcId", 101)
        r = route(p, mc, text, cfg)
        if mode == "llm":
            contested = ([101] if mc != 101 else []) + [c for c in TARGETS if c != mc]
            r = Route(frozen={}, contested=contested, triggers={k: sorted(v) for k, v in detect(text).items()},
                      stage1=r.stage1)
        elif mode == "stage1":
            r = Route(frozen={c: int(c in r.stage1) for c in TARGETS if c != mc}, contested=[], stage1=r.stage1)
        rec = {"row_id": row.get("row_id", i), "mcId": mc, "stage1": r.stage1,
               "probs": {str(c): round(float(v), 4) for c, v in zip(TARGETS, p)},
               "contested": r.contested, "triggers": r.triggers, "escalated": r.escalate,
               "llm": None, "llm_error": None, "seconds": s1_seconds, "input_tokens": 0, "output_tokens": 0}
        if r.escalate and arbiter is not None:
            calls += 1
            ans = arbiter.ask(text, mc, r.contested, list(r.triggers))
            rec.update(llm=ans["labels"], llm_error=ans.get("error"), cached=ans["cached"],
                       input_tokens=ans["input_tokens"], output_tokens=ans["output_tokens"])
            rec["seconds"] += ans["seconds"]
        if rec["llm"] is None:
            fallback = [c for c in r.contested if c in r.stage1]
            rec["labels"] = merge(r, fallback)
            rec["fallback"] = r.escalate
        else:
            rec["labels"] = merge(r, rec["llm"])
            rec["fallback"] = False
        records.append(rec)
        if log_every and calls and calls % log_every == 0 and rec["escalated"]:
            print(f"  {i + 1}/{len(rows)} rows, {calls} LLM calls", flush=True)
    return records
