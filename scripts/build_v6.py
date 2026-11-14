import re
from pathlib import Path

import emoji
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "test"

EMOTICON = re.compile(r"(?<!\w)[:;=]-?[)(DРрP]+|\){2,}|\({2,}")
LEFTOVER = re.compile("[️︎⃣‍]")
MARK = "\x00"
MARKED_RUN = re.compile(rf"[ \t]*(?:{MARK}[ \t]*)+")


def strip_emoji(text):
    marked = emoji.replace_emoji(text, replace=MARK)
    marked = EMOTICON.sub(MARK, marked)
    marked = LEFTOVER.sub(MARK, marked)
    if MARK not in marked:
        return text
    lines = [MARKED_RUN.sub(" ", line).strip() if MARK in line else line for line in marked.split("\n")]
    return "\n".join(lines).strip()


def removed_symbols(text):
    found = [item["emoji"] for item in emoji.emoji_list(text)]
    found += EMOTICON.findall(text)
    return "".join(found)


def main():
    v5 = pd.read_parquet(OUT / "test_v5.parquet")
    v6 = v5.copy()
    v6["description"] = v5["description"].map(strip_emoji)

    changed = v5["description"] != v6["description"]
    log = pd.DataFrame({
        "row_id": v5.loc[changed, "row_id"],
        "removed": v5.loc[changed, "description"].map(removed_symbols),
        "chars_before": v5.loc[changed, "description"].str.len(),
        "chars_after": v6.loc[changed, "description"].str.len(),
        "action": "edited",
        "reason": "удалены эмодзи и смайлики",
    })

    assert (v6["description"].str.len() > 0).all()
    assert not v6["description"].duplicated().any()

    v6.to_parquet(OUT / "test_v6.parquet", index=False)
    v6.to_parquet(OUT / "test_gold.parquet", index=False)
    log.sort_values("row_id").to_csv(OUT / "test_v6_changes.csv", index=False)

    ys = [c for c in v6 if c.startswith("y_")]
    print("v6", v6.shape, "изменено строк", int(changed.sum()))
    print("осталось эмодзи", sum(emoji.emoji_count(text) for text in v6["description"]))
    print("метки не изменились", bool((v5[ys] == v6[ys]).all().all()))


if __name__ == "__main__":
    main()
