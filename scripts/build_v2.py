import re
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "test"
PLASTER = re.compile(r"штукатур|оштукатур|маяк", re.IGNORECASE)
NEGATION = re.compile(r"(никак\w*|\bне\b|\bбез\b|вместо|кроме|забудьте)[^.!?\n/]{0,25}$", re.IGNORECASE)


def explicit_plaster(description):
    text = description
    return any(not NEGATION.search(text[max(0, m.start() - 40):m.start()]) for m in PLASTER.finditer(text))


def main():
    v1 = pd.read_parquet(OUT / "test_v1.parquet")
    v2 = v1.copy()
    changes = []
    for i, row in v2.iterrows():
        if row.y_102 == 1 and not explicit_plaster(row.description):
            v2.at[i, "y_102"] = 0
            changes.append({"row_id": row.row_id, "mcId": 102, "old": 1, "new": 0, "reason": "выравнивание без явной штукатурки не считается 102"})

    v2.to_parquet(OUT / "test_v2.parquet", index=False)
    pd.DataFrame(changes).to_csv(OUT / "test_v2_changes.csv", index=False)
    print("v2", v2.shape, "изменённых пар", len(changes), "102 было", int(v1.y_102.sum()), "стало", int(v2.y_102.sum()))


if __name__ == "__main__":
    main()
