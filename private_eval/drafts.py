"""Drafts for the descriptions column: extract (sentences of the source, no LLM) or an LLM with generate_v1."""
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .llm import ROOT, ApiClient

sys.path.insert(0, str(ROOT))

from hybrid.triggers import CLASS_RE  # noqa: E402

NAMES = {101: "Ремонт под ключ", 102: "Штукатурные работы", 103: "Плиточные работы", 104: "Сантехника",
         105: "Электрика", 106: "Поклейка обоев и малярные работы"}
PROMPT = Path(__file__).resolve().parent / "prompts" / "generate_v1.txt"

CONTACT = re.compile(r"\+?\d[\d\s()-]{8,}\d|https?://|www\.|@\w|t\.me|whatsapp|телеграм|вотсап|звоните", re.I)
SEO = re.compile(r"#\w|также ищут|ключевые слова|теги|по запросам", re.I)
VAGUE = re.compile(r"этих видов|из перечисленн|любой из|все виды|под ключ|комплексн", re.I)
SPLIT = re.compile(r"(?<=[.!?…])\s+|\n+")


def sentences(text):
    return [s.strip(" •-–—*\t") for s in SPLIT.split(text) if len(s.strip(" •-–—*\t")) > 2]


def extract(description, target, max_general=3):
    """Topical sentences for the target plus up to max_general neutral sentences, without contacts or SEO tails."""
    topical, general = [], []
    for s in sentences(description):
        if CONTACT.search(s) or SEO.search(s):
            continue
        cats = {c for c in range(102, 107) if CLASS_RE[c].search(s)}
        if target in cats:
            topical.append(s)
        elif not cats and not VAGUE.search(s) and len(s) <= 200:
            general.append(s)
    if not topical:
        topical = [f"{NAMES[target]}."]
    text = " ".join(_end(s) for s in topical + general[:max_general])
    return text[:1200]


def _end(s):
    return s if s[-1] in ".!?…" else s + "."


def parse_draft(raw):
    if not isinstance(raw, str) or not raw.strip():
        return None, None
    title = re.search(r"Заголовок:\s*(.+)", raw)
    body = re.search(r"Текст:\s*(.+)", raw, re.S)
    text = (body.group(1) if body else raw).strip()
    return (title.group(1).strip() if title else None), (text or None)


def generate(rows, records, mode, client, workers, progress):
    """Returns per row a list of {mcId, title, text, source, error} aligned with record labels."""
    system = PROMPT.read_text(encoding="utf-8")
    tasks = [(i, c) for i, rec in enumerate(records) for c in rec["labels"]]

    def one(task):
        i, c = task
        desc = rows[i]["description"]
        fallback = extract(desc, c) if c != 101 else f"{NAMES[c]}."
        if mode == "extract":
            progress()
            return {"mcId": c, "title": NAMES[c], "text": fallback, "source": "extract", "error": None}
        user = f"Целевая микрокатегория: {c} {NAMES[c]}\n\nИсходное объявление:\n{desc}"
        ans = client.chat(system, user, max_tokens=700)
        title, text = parse_draft(ans["raw"]) if ans["error"] is None else (None, None)
        progress()
        if text is None:
            return {"mcId": c, "title": NAMES[c], "text": fallback, "source": "extract",
                    "error": ans["error"] or "empty_draft"}
        return {"mcId": c, "title": title, "text": text, "source": mode, "error": None,
                "seconds": ans["seconds"], "cached": ans["cached"],
                "input_tokens": ans["input_tokens"], "output_tokens": ans["output_tokens"]}

    parallel = workers if isinstance(client, ApiClient) else 1
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        results = list(pool.map(one, tasks))
    out = [[] for _ in records]
    for (i, _), d in zip(tasks, results):
        out[i].append(d)
    return out
