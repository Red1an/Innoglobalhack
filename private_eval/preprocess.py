"""Внутренний шаг прогона run.py predict --prep: очистка description перед классификацией и черновиками.

Входной файл не меняется, submission.csv имеет те же поля, что и без предобработки.
"""
import re
import sys

from .llm import ROOT

sys.path.insert(0, str(ROOT / "dataset_review"))
from train_text_cleaning import (  # noqa: E402
    normalize_emoji, normalize_scripts, normalize_unicode, normalize_whitespace, remove_controls,
)

INJECTION_BLOCK = re.compile(
    r"<(tool_call|system|assistant|user|instructions?)\b[^>]*>.*?</\1\s*>|<!--.*?-->|<meta\b[^>]*>",
    re.I | re.S,
)
INJECTION_LINE = re.compile(r"""(?ix)
    языков\w*\s+модел | для\s+модел | команд\w*\s+(?:модел|классификатор) | классификатор[уа]\b
  | \bLLM\b | \bprompt | промпт
  | игнорируй | ignore\s+(?:all\s+)?(?:previous|prior|the\s+above) | previous\s+instructions
  | ^\s*\[?(?:system|user|assistant)\]?\s*: | ^\s*\[(?:system|user|assistant)\]
  | "?microcats"?\s*[:=] | correct_answer | gold_labels | (?:это|вот)\s+готовый\s+ответ
  | скопируй\s+(?:его|ответ) | вместо\s+(?:анализа|текста)
  | служебн\w+\s+сообщени | начисли | полный\s+балл | правил\w*\s+классификации | output\s+category
""")
MAX_INJECTION_LINE = 300
LEFTOVER_PICTOGRAPH = re.compile(r"[⠀\U0001F000-\U0001FAFF☀-➿⬀-⯿️‍]")


def is_injection(line):
    return len(line) <= MAX_INJECTION_LINE and bool(INJECTION_LINE.search(line))


def remove_injections(text):
    text, blocks = INJECTION_BLOCK.subn("\n", text)
    lines = text.split("\n")
    kept = [line for line in lines if not is_injection(line)]
    return "\n".join(kept), blocks + len(lines) - len(kept)


def strip_leftover_pictographs(text):
    return LEFTOVER_PICTOGRAPH.sub("", text)


def preprocess(text):
    stats = {}
    out = normalize_unicode(text)
    out = remove_controls(out)
    out, stats["injections"] = remove_injections(out)
    before = out
    out = strip_leftover_pictographs(normalize_emoji(out))
    stats["emoji"] = before != out
    before = out
    out = normalize_scripts(normalize_whitespace(out))
    stats["homoglyphs"] = sum(a != b for a, b in zip(re.findall(r"\w+", before), re.findall(r"\w+", out)))
    return out, stats


def preprocess_rows(rows):
    totals = {"rows": len(rows), "changed": 0, "injection_rows": 0, "injections": 0,
              "emoji_rows": 0, "homoglyph_rows": 0, "homoglyph_words": 0}
    for r in rows:
        clean, stats = preprocess(r["description"])
        if not clean.strip():
            clean = normalize_whitespace(r["description"])
        totals["changed"] += clean != r["description"]
        totals["injection_rows"] += stats["injections"] > 0
        totals["injections"] += stats["injections"]
        totals["emoji_rows"] += stats["emoji"]
        totals["homoglyph_rows"] += stats["homoglyphs"] > 0
        totals["homoglyph_words"] += stats["homoglyphs"]
        r["description"] = clean
    return totals

