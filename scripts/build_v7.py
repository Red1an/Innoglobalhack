import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "test"

LATIN_LOOKALIKES = "aceopxyABCEHKMOPTXY"
CYRILLIC_PAIRS = "асеорхуАВСЕНКМОРТХУ"
TO_CYRILLIC = str.maketrans(LATIN_LOOKALIKES, CYRILLIC_PAIRS)
WORD = re.compile(r"\w+")
CYRILLIC = re.compile(r"[а-яё]", re.IGNORECASE)
LATIN = re.compile(r"[a-z]", re.IGNORECASE)


def only_lookalikes(word):
    return all(char in LATIN_LOOKALIKES for char in word if LATIN.match(char))


def has_cyrillic(word):
    return bool(word) and bool(CYRILLIC.search(word))


def fix_word(word, previous_word, next_word):
    if not LATIN.search(word) or not only_lookalikes(word):
        return word
    if has_cyrillic(word):
        return word.translate(TO_CYRILLIC)
    if len(word) == 1 and has_cyrillic(previous_word) and has_cyrillic(next_word):
        return word.translate(TO_CYRILLIC)
    return word


def fix_latin(text):
    matches = list(WORD.finditer(text))
    parts, position = [], 0
    for index, match in enumerate(matches):
        previous_word = matches[index - 1].group(0) if index > 0 else None
        next_word = matches[index + 1].group(0) if index + 1 < len(matches) else None
        parts.append(text[position:match.start()])
        parts.append(fix_word(match.group(0), previous_word, next_word))
        position = match.end()
    parts.append(text[position:])
    return "".join(parts)


def changed_words(before, after):
    pairs = zip(WORD.findall(before), WORD.findall(after))
    return ", ".join(f"{old}>{new}" for old, new in pairs if old != new)


def main():
    v6 = pd.read_parquet(OUT / "test_v6.parquet")
    v7 = v6.copy()
    v7["description"] = v6["description"].map(fix_latin)

    changed = v6["description"] != v7["description"]
    log = pd.DataFrame({
        "row_id": v6.loc[changed, "row_id"],
        "words": [changed_words(b, a) for b, a in zip(v6.loc[changed, "description"], v7.loc[changed, "description"])],
        "action": "edited",
        "reason": "латинские буквы, похожие на кириллические, заменены кириллическими",
    })

    assert (v6["description"].str.len() == v7["description"].str.len()).all()
    assert not v7["description"].duplicated().any()

    v7.to_parquet(OUT / "test_v7.parquet", index=False)
    v7.to_parquet(OUT / "test_gold.parquet", index=False)
    log.sort_values("row_id").to_csv(OUT / "test_v7_changes.csv", index=False)

    remaining = sum(
        bool(CYRILLIC.search(word)) and bool(LATIN.search(word))
        for text in v7["description"]
        for word in WORD.findall(text)
    )
    ys = [c for c in v7 if c.startswith("y_")]
    print("v7", v7.shape, "изменено строк", int(changed.sum()))
    print("исправлено слов", sum(len(words.split(", ")) for words in log["words"]))
    print("осталось слов со смешанными алфавитами", remaining)
    print("метки не изменились", bool((v6[ys] == v7[ys]).all().all()))


if __name__ == "__main__":
    main()
