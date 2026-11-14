"""Общие правила базовой очистки текста объявлений."""

import re
import unicodedata
from functools import lru_cache, partial
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
REVIEWED_SPELLINGS = {
    "whitе": "white",
    "bох": "box",
    "WhаtsАрр": "WhatsApp",
    "Whаtsарр": "WhatsApp",
    "Теlеgrаm": "Telegram",
    "SЕО": "SEO",
}
PRESERVED_NAMES_AND_CODES = frozenset({
    "АкимовPRO", "PROквадраты", "SmartРемонт", "капиталкаMNN",
    "84G5Y74YGТ4НIUЕВНGВ", "234F524f2G54Н4", "Аргумент32473DSFDGDG",
})

TOKEN = re.compile(r"[\w\u0300-\u036f]+")
LINE_ENDINGS = re.compile(r"\r\n?|[\u0085\u2028\u2029]|\\u202[89]")
KEYCAP = re.compile(r"([0-9#*])[\ufe0e\ufe0f]?\u20e3")
UNDER_KEY = re.compile(r"\b([пП][оoОO][дД])[ \t\xa0]*🔑[\ufe0e\ufe0f]?")
SPECIAL_PUNCTUATION = re.compile(r"([‼⁉])[\ufe0e\ufe0f]?")
PUNCTUATION = {"‼": "!!", "⁉": "!?"}
FLAG = "[\U0001F1E6-\U0001F1FF]{2}"
PICTOGRAM = rf"(?:[{emoji_ranges()}]|[©®™]\ufe0f)[\ufe0e\ufe0f]?[\U0001F3FB-\U0001F3FF]*[\U000E0020-\U000E007F]*"
EMOJI = re.compile(rf"(?:(?:{FLAG}|{PICTOGRAM})(?:\u200d{PICTOGRAM})*)+")
DISCRETIONARY_SPACES = str.maketrans({"\u200b": " ", "\uf0b7": "•"})
EMOJI_COMPONENTS = re.compile(r"[\ufe0e\ufe0f\u20e3]")
LETTER = re.compile(r"[^\W\d_]")
WORD_JOIN = re.compile(r"(?<=[а-яё])(?=[А-ЯЁ][а-яё])")
WORDS_OUTSIDE_HASHTAGS = re.compile(r"#\w+|[^#\s]+")
NUMBERED_LINE = re.compile(r"^([1-9])(?=[а-яёА-ЯЁ])", re.M)
NUMBER_UNIT = re.compile(r"(?<!\w)(\d+)(?=(?:год(?:а|ов)?|лет|человек(?:а)?)\b)", re.I)
NUMBERED_HEADING = re.compile(
    r"^(\d+[.)]\s+[^.!?\n]{3,65}[а-яё])(?=(?:Осматриваем|Составляем|До начала|Оплата|От черновых)\b)"
)
SEPARATOR = re.compile(r"_{4,}|[-–—=]{4,}")


@lru_cache(maxsize=65536)
def has_mixed_scripts(token):
    scripts = {unicodedata.name(char, "").split(" ", 1)[0] for char in token if char.isalpha()}
    return "LATIN" in scripts and "CYRILLIC" in scripts


def mixed_tokens(text):
    return (match[0] for match in TOKEN.finditer(text) if has_mixed_scripts(match[0]))


@lru_cache(maxsize=65536)
def normalize_token(token):
    if token in REVIEWED_SPELLINGS:
        return REVIEWED_SPELLINGS[token]
    if not has_mixed_scripts(token) or not token.isalpha() or token in PRESERVED_NAMES_AND_CODES:
        return token
    corrected = token.translate(HOMOGLYPHS)
    return token if has_mixed_scripts(corrected) else corrected


def strip_emoji(match):
    left, right = match.span()
    text = match.string
    if left > 0 and right < len(text) and text[left - 1].isalnum() and text[right].isalnum():
        return " "
    return ""


def normalize_emoji(text, *, separate_keycaps=True):
    text = UNDER_KEY.sub(lambda match: match[1] + (" КЛЮЧ" if match[1].isupper() else " ключ"), text)
    if separate_keycaps:
        def replace_keycap(match):
            left, right = match.span()
            before = " " if left and LETTER.fullmatch(text[left - 1]) else ""
            after = " " if LETTER.match(text, right) else ""
            return before + match[1] + after

        text = KEYCAP.sub(replace_keycap, text)
    else:
        text = KEYCAP.sub(r"\1", text)
    text = SPECIAL_PUNCTUATION.sub(lambda match: PUNCTUATION[match[1]], text)
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


def normalize_unicode(text):
    return unicodedata.normalize("NFC", text)


def normalize_scripts(text):
    return TOKEN.sub(lambda match: normalize_token(match[0]), text)


def separate_joined_list(text):
    lines = []
    for line in text.split("\n"):
        # Several joins in the same line distinguish pasted lists from names such as СтройЛидер.
        joins = sum(len(WORD_JOIN.findall(word)) for word in WORDS_OUTSIDE_HASHTAGS.findall(line)
                    if not word.startswith("#"))
        if joins >= 3:
            line = WORDS_OUTSIDE_HASHTAGS.sub(
                lambda match: match[0] if match[0].startswith("#") else WORD_JOIN.sub(" ", match[0]), line
            )
        lines.append(NUMBERED_HEADING.sub(r"\1\n", line))
    return "\n".join(lines)


def separate_joined_numbers(text):
    if len(set(NUMBERED_LINE.findall(text))) >= 3:
        text = NUMBERED_LINE.sub(r"\1 ", text)
    return NUMBER_UNIT.sub(r"\1 ", text)


def remove_decorative_separators(text):
    return SEPARATOR.sub("\n", text)


STEPS = (
    ("unicode_nfc", normalize_unicode, "Приведение Unicode к NFC"),
    ("emoji", normalize_emoji, "Удаление эмодзи с сохранением цифр, пунктуации и фразы «под ключ»"),
    ("controls", remove_controls, "Нормализация переносов и невидимых символов"),
    ("joined_list", separate_joined_list, "Восстановление пробелов в склеенных перечнях"),
    ("joined_numbers", separate_joined_numbers, "Восстановление пробелов у номеров списка, возраста и размера бригады"),
    ("decorative_separators", remove_decorative_separators, "Замена декоративных разделителей переносом строки"),
    ("whitespace", normalize_whitespace, "Нормализация пробелов с сохранением строк и абзацев"),
    ("mixed_scripts", normalize_scripts, "Исправление смешанных букв и разобранных написаний"),
)
STEPS_V1 = tuple(
    (name, partial(transform, separate_keycaps=False) if name == "emoji" else transform, description)
    for name, transform, description in STEPS if name not in {"joined_list", "joined_numbers", "decorative_separators"}
)


def clean_text(text):
    for _, transform, _ in STEPS:
        text = transform(text)
    return text
