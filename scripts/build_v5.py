import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "test"

TARGET_EMPTY_SHARE = 0.25
SEED = 42

STEMS = {
    102: r"штукатур|выравнива|шпакл|шпатл",
    103: r"плитк|кафел|керамогранит|мозаик",
    104: r"сантех|труб|смесител|унитаз|канализац|водоснаб|отоплен",
    105: r"электр|проводк|розет|освещени|щиток",
    106: r"обо[ий]|покрас|маляр|окраш",
}


def category_mentions(text):
    text = text.lower()
    return sum(bool(re.search(pattern, text)) for pattern in STEMS.values())


def main():
    v4 = pd.read_parquet(OUT / "test_v4.parquet")
    ys = [c for c in v4 if c.startswith("y_")]
    empty_mask = v4[ys].sum(axis=1) == 0
    labeled = v4[~empty_mask]
    empty = v4[empty_mask].assign(mentions=v4.loc[empty_mask, "description"].map(category_mentions))

    keep_count = round(TARGET_EMPTY_SHARE * len(labeled) / (1 - TARGET_EMPTY_SHARE))
    hard = empty[empty.mentions > 0]
    easy = empty[empty.mentions == 0]
    assert len(hard) <= keep_count, (len(hard), keep_count)
    kept_easy = easy.sample(n=keep_count - len(hard), random_state=SEED)
    kept = pd.concat([hard, kept_easy])
    dropped = empty.drop(kept.index)

    v5 = v4.drop(dropped.index).sort_values("row_id").reset_index(drop=True)
    v5.to_parquet(OUT / "test_v5.parquet", index=False)
    v5.to_parquet(OUT / "test_gold.parquet", index=False)

    log = dropped[["row_id", "mentions"]].assign(
        action="removed",
        reason="пустой ответ без упоминаний работ 102–106, сокращение доли пустых до 25%",
    )
    log.sort_values("row_id").to_csv(OUT / "test_v5_changes.csv", index=False)

    share = (v5[ys].sum(axis=1) == 0).mean()
    print("v4", v4.shape, "пустых", int(empty_mask.sum()))
    print("оставлено пустых: трудных", len(hard), "лёгких", len(kept_easy), "удалено", len(dropped))
    print("v5", v5.shape, "пустых", int((v5[ys].sum(axis=1) == 0).sum()), f"доля {share:.1%}")
    print("метки v5", v5[ys].sum().to_dict())


if __name__ == "__main__":
    main()
