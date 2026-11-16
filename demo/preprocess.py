"""Предобработка description по флагу --prep: Unicode, управляющие символы, инъекции, эмодзи, омоглифы."""
import re
import unicodedata
from functools import lru_cache
from pathlib import Path


def emoji_ranges():
    ranges = []
    for line in Path(__file__).with_name("emoji_ranges.txt").read_text(encoding="utf-8").splitlines():
        entry = line.partition("#")[0].strip()
        if entry:
            start, _, end = entry.partition("..")
            ranges.append(chr(int(start, 16)) + ("-" + chr(int(end, 16)) if end else ""))
    return "".join(ranges)


HOMOGLYPHS = str.maketrans("aceopxyëABCEHKMOPTXYË", "асеорхуёАВСЕНКМОРТХУЁ")
REVIEWED_SPELLINGS = {"whitе": "white", "bох": "box", "WhаtsАрр": "WhatsApp", "Whаtsарр": "WhatsApp",
                      "Теlеgrаm": "Telegram", "SЕО": "SEO"}
PRESERVED = frozenset({"АкимовPRO", "PROквадраты", "SmartРемонт", "капиталкаMNN",
                       "84G5Y74YGТ4НIUЕВНGВ", "234F524f2G54Н4", "Аргумент32473DSFDGDG"})

TOKEN = re.compile(r"[\ẁ-ͯ]+")
LINE_ENDINGS = re.compile(r"\r\n?|[\u0085  ]|\\u202[89]")
KEYCAP = re.compile(r"([0-9#*])[︎️]?⃣")
UNDER_KEY = re.compile(r"\b([пП][оoОO][дД])[ \t\xa0]*🔑[︎️]?")
SPECIAL_PUNCTUATION = re.compile(r"([‼⁉])[︎️]?")
PUNCTUATION = {"‼": "!!", "⁉": "!?"}
FLAG = "[\U0001F1E6-\U0001F1FF]{2}"
PICTOGRAM = rf"(?:[{emoji_ranges()}]|[©®™]️)[︎️]?[\U0001F3FB-\U0001F3FF]*[\U000E0020-\U000E007F]*"
EMOJI = re.compile(rf"(?:(?:{FLAG}|{PICTOGRAM})(?:‍{PICTOGRAM})*)+")
DISCRETIONARY_SPACES = str.maketrans({"​": " ", "": "•"})
EMOJI_COMPONENTS = re.compile(r"[︎️⃣]")
LETTER = re.compile(r"[^\W\d_]")
LEFTOVER_PICTOGRAPH = re.compile(r"[⠀\U0001F000-\U0001FAFF☀-➿⬀-⯿️‍]")

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


@lru_cache(maxsize=65536)
def has_mixed_scripts(token):
    scripts = {unicodedata.name(char, "").split(" ", 1)[0] for char in token if char.isalpha()}
    return "LATIN" in scripts and "CYRILLIC" in scripts


@lru_cache(maxsize=65536)
def normalize_token(token):
    if token in REVIEWED_SPELLINGS:
        return REVIEWED_SPELLINGS[token]
    if not has_mixed_scripts(token) or not token.isalpha() or token in PRESERVED:
        return token
    corrected = token.translate(HOMOGLYPHS)
    return token if has_mixed_scripts(corrected) else corrected


def strip_emoji(match):
    left, right = match.span()
    text = match.string
    if left > 0 and right < len(text) and text[left - 1].isalnum() and text[right].isalnum():
        return " "
    return ""


def normalize_emoji(text):
    text = UNDER_KEY.sub(lambda m: m[1] + (" КЛЮЧ" if m[1].isupper() else " ключ"), text)

    def replace_keycap(match):
        left, right = match.span()
        before = " " if left and LETTER.fullmatch(text[left - 1]) else ""
        after = " " if LETTER.match(text, right) else ""
        return before + match[1] + after

    text = KEYCAP.sub(replace_keycap, text)
    text = SPECIAL_PUNCTUATION.sub(lambda m: PUNCTUATION[m[1]], text)
    return EMOJI_COMPONENTS.sub("", EMOJI.sub(strip_emoji, text))


def remove_controls(text):
    text = LINE_ENDINGS.sub("\n", text).translate(DISCRETIONARY_SPACES)
    clean = []
    for char in text:
        if char in "\n\t":
            clean.append(char)
            continue
        category = unicodedata.category(char)
        if category != "Cf":
            clean.append(" " if category == "Cc" else char)
    return "".join(clean)


def normalize_whitespace(text):
    lines = [" ".join(line.split()) for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines).strip())


def normalize_scripts(text):
    return TOKEN.sub(lambda m: normalize_token(m[0]), text)


def remove_injections(text):
    text, blocks = INJECTION_BLOCK.subn("\n", text)
    lines = text.split("\n")
    kept = [line for line in lines if not (len(line) <= MAX_INJECTION_LINE and INJECTION_LINE.search(line))]
    return "\n".join(kept), blocks + len(lines) - len(kept)


def preprocess(text):
    out = remove_controls(unicodedata.normalize("NFC", text))
    out, injections = remove_injections(out)
    out = LEFTOVER_PICTOGRAPH.sub("", normalize_emoji(out))
    out = normalize_scripts(normalize_whitespace(out))
    if not out.strip():
        out = normalize_whitespace(text)
    return out, injections


def preprocess_rows(rows):
    stats = {"изменено строк": 0, "строк с инъекциями": 0, "удалено инъекций": 0}
    for r in rows:
        clean, injections = preprocess(r["description"])
        stats["изменено строк"] += clean != r["description"]
        stats["строк с инъекциями"] += injections > 0
        stats["удалено инъекций"] += injections
        r["description"] = clean
    return stats
