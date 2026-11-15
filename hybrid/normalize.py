"""Inference normalization shared by Stage 1 training, Stage 1 inference and the Stage 2 prompt."""
import re
import unicodedata

INVISIBLE = re.compile("[\\u00ad\\u200b-\\u200f\\u2060-\\u2064\\ufeff]")
LATIN_TO_CYRILLIC = str.maketrans("aceopxyKkMHTBAECOPX", "асеорхуКкМНТВАЕСОРХ")
CYRILLIC = re.compile("[а-яё]", re.I)
LATIN = re.compile("[a-z]", re.I)
KEYCAP = re.compile("[0-9#*]\\ufe0f?\\u20e3")
EMOJI = re.compile("[\\U0001f000-\\U0001faff\\u2600-\\u27bf\\u2b00-\\u2bff\\ufe0f\\u200d]+")
CAMEL = re.compile("(?<=[а-яё])(?=[А-ЯЁ][а-яё])")
LETTER_DIGIT = re.compile("(?<=[а-яё]{3})(?=\\d)|(?<=\\d)(?=[а-яёА-ЯЁ]{3})")


def fix_homoglyphs(token: str) -> str:
    if CYRILLIC.search(token) and LATIN.search(token):
        return token.translate(LATIN_TO_CYRILLIC)
    return token


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = INVISIBLE.sub("", text)
    text = KEYCAP.sub(" ; ", text)
    text = EMOJI.sub(" ", text)
    text = "".join(fix_homoglyphs(part) for part in re.split(r"(\s+)", text))
    text = CAMEL.sub(" ", text)
    text = LETTER_DIGIT.sub(" ", text)
    return "\n".join(re.sub(r"[^\S\n]+", " ", line).strip() for line in text.split("\n")).strip()
