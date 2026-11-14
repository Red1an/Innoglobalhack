import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "test"

DESIGN = "электрика только в бесплатном проекте, монтажа нет"
PREPARATION = "только шпаклёвка или подготовка под покраску и обои, самой покраски и поклейки нет"
INCIDENTAL = "упомянуто только в примере объекта или в технадзоре"

CHANGES = [
    (26, 105, DESIGN), (113, 105, DESIGN),
    (6, 106, PREPARATION), (38, 106, PREPARATION), (43, 106, PREPARATION), (68, 106, PREPARATION), (134, 106, PREPARATION),
    (30, 102, INCIDENTAL), (30, 105, INCIDENTAL), (67, 102, INCIDENTAL),
]


def main():
    v3 = pd.read_parquet(OUT / "test_v3.parquet")
    v4 = v3.set_index("row_id")
    log = []
    for row_id, mc, reason in CHANGES:
        old = int(v4.at[row_id, f"y_{mc}"])
        assert old == 1, (row_id, mc)
        v4.at[row_id, f"y_{mc}"] = 0
        log.append({"row_id": row_id, "mcId": mc, "old": 1, "new": 0, "reason": reason})
    v4 = v4.reset_index()
    v4.to_parquet(OUT / "test_v4.parquet", index=False)
    v4.to_parquet(OUT / "test_gold.parquet", index=False)
    pd.DataFrame(log).to_csv(OUT / "test_v4_changes.csv", index=False)
    ys = [c for c in v4 if c.startswith("y_")]
    print("v4", v4.shape, "изменённых пар", len(log))
    print("метки v3", v3[ys].sum().to_dict(), int(v3[ys].values.sum()))
    print("метки v4", v4[ys].sum().to_dict(), int(v4[ys].values.sum()), "пустых", int((v4[ys].sum(1) == 0).sum()))


if __name__ == "__main__":
    main()
