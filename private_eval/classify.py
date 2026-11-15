"""Classification methods: charlr (Stage 1), llm (static prompt, API or Ollama), cascade (Stage 1 → LLM arbiter)."""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .llm import ROOT, ApiClient, OllamaClient, label_schema, parse_labels

sys.path.insert(0, str(ROOT))

from hybrid.cascade import load as load_stage1, run as run_cascade  # noqa: E402
from hybrid.stage2 import build_prompt  # noqa: E402

PROMPTS = ROOT / "research" / "experiments" / "prompts"
TARGETS = [102, 103, 104, 105, 106]


def prompt_text(name):
    path = Path(name) if Path(name).suffix else PROMPTS / f"{name}.txt"
    return path.read_text(encoding="utf-8-sig")


def make_client(backend, model, args):
    if backend == "api":
        client = ApiClient(model, base_url=args.base_url, key_env=args.key_env, bearer=args.bearer)
        client.check()
        return client
    client = OllamaClient(model)
    client.check()
    return client


class Stage1:
    def __init__(self):
        self.model, self.cfg = load_stage1()

    def probs(self, texts):
        return self.model.predict_proba(texts)

    def labels(self, p, mc):
        return sorted(c for c, v in zip(TARGETS, p) if v >= self.cfg.t_mid and c != mc)


def classify_charlr(rows, stage1, progress):
    probs = stage1.probs([r["description"] for r in rows])
    out = []
    for r, p in zip(rows, probs):
        out.append({"labels": stage1.labels(p, r["mcId"]), "stage1": stage1.labels(p, r["mcId"]),
                    "llm_called": False, "error": None, "fallback": False, "seconds": 0.0,
                    "input_tokens": 0, "output_tokens": 0, "cached": False})
        progress()
    return out


def classify_llm(rows, client, prompt, stage1, workers, progress):
    probs = stage1.probs([r["description"] for r in rows])

    def one(item):
        r, p = item
        allowed = [c for c in range(101, 107) if c != r["mcId"]]
        user = json.dumps({"mcId": r["mcId"], "description": r["description"]}, ensure_ascii=False)
        if isinstance(client, OllamaClient):
            ans = client.chat(prompt, user, max_tokens=64, schema=label_schema(allowed))
        else:
            ans = client.chat(prompt, user, max_tokens=96)
        labels = parse_labels(ans["raw"], allowed) if ans["error"] is None else None
        error = ans["error"] or (None if labels is not None else "invalid_response")
        s1 = stage1.labels(p, r["mcId"])
        progress()
        return {"labels": s1 if labels is None else labels, "stage1": s1, "llm": labels, "llm_called": True,
                "error": error, "fallback": labels is None, "raw": ans["raw"], "seconds": ans["seconds"],
                "input_tokens": ans["input_tokens"], "output_tokens": ans["output_tokens"], "cached": ans["cached"]}

    with ThreadPoolExecutor(max_workers=workers if isinstance(client, ApiClient) else 1) as pool:
        return list(pool.map(one, zip(rows, probs)))


class ApiArbiter:
    """Same interface as hybrid.stage2.OllamaArbiter, but over the API client."""

    def __init__(self, client, seo):
        self.client, self.seo = client, seo

    def ask(self, text, mc_id, contested, triggers):
        prompt = build_prompt(contested, triggers, self.seo)
        user = json.dumps({"mcId": mc_id, "description": text}, ensure_ascii=False)
        ans = self.client.chat(prompt, user, max_tokens=64)
        labels = parse_labels(ans["raw"], contested) if ans["error"] is None else None
        return {**ans, "labels": labels, "error": ans["error"] or (None if labels is not None else "invalid_response")}


class OllamaArbiter(ApiArbiter):
    def ask(self, text, mc_id, contested, triggers):
        prompt = build_prompt(contested, triggers, self.seo)
        user = json.dumps({"mcId": mc_id, "description": text}, ensure_ascii=False)
        ans = self.client.chat(prompt, user, max_tokens=64, schema=label_schema(contested))
        labels = parse_labels(ans["raw"], contested) if ans["error"] is None else None
        return {**ans, "labels": labels, "error": ans["error"] or (None if labels is not None else "invalid_response")}


def classify_cascade(rows, client, seo, stage1, progress):
    arbiter = (OllamaArbiter if isinstance(client, OllamaClient) else ApiArbiter)(client, seo)
    items = [{"row_id": r["testcase"], "mcId": r["mcId"], "description": r["description"]} for r in rows]
    recs = run_cascade(items, stage1.model, stage1.cfg, "cascade", arbiter, log_every=0)
    out = []
    for rec in recs:
        out.append({"labels": rec["labels"], "stage1": rec["stage1"], "llm": rec["llm"],
                    "llm_called": bool(rec["escalated"]), "contested": rec["contested"],
                    "triggers": list(rec["triggers"]), "probs": rec["probs"],
                    "error": rec["llm_error"], "fallback": bool(rec["fallback"]),
                    "seconds": rec["seconds"], "input_tokens": rec["input_tokens"],
                    "output_tokens": rec["output_tokens"], "cached": rec.get("cached", False)})
        progress()
    return out

