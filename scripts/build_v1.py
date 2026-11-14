import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "test"
IDS = [102, 103, 104, 105, 106]

test = pd.read_parquet(ROOT / "data" / "processed" / "test_clean.parquet")
dup = pd.read_excel(ROOT / "review" / "test_duplicate_conflicts.xlsx")
team = {(int(r.row_id), int(r.mcId)): int(r.decision) for r in dup.itertuples()}

v0 = test[["row_id", "description"] + [f"y_{m}" for m in IDS]].copy()
v0.to_parquet(OUT / "test_v0.parquet", index=False)

rows, changes = [], []
for _, group in v0.groupby("description", sort=False):
    first = group.iloc[0]
    record = {"row_id": int(first.row_id), "row_ids": [int(x) for x in group.row_id], "description": first.description}
    for m in IDS:
        values = set(group[f"y_{m}"])
        decided = {team[(int(r), m)] for r in group.row_id if (int(r), m) in team}
        if decided:
            assert len(decided) == 1
            new = decided.pop()
        else:
            assert len(values) == 1, (first.row_id, m, values)
            new = values.pop()
        record[f"y_{m}"] = new
        if set(group[f"y_{m}"]) != {new}:
            changes.append({"row_id": record["row_id"], "mcId": m, "old": ", ".join(f"{int(r)}:{int(v)}" for r, v in zip(group.row_id, group[f"y_{m}"])), "new": new, "reason": "решение по дубликатам"})
    rows.append(record)

v1 = pd.DataFrame(rows)
v1.to_parquet(OUT / "test_v1.parquet", index=False)
pd.DataFrame(changes).to_csv(OUT / "test_v1_changes.csv", index=False)
print("v0", v0.shape, "v1", v1.shape, "изменённых пар", len(changes))
print(v1[[f"y_{m}" for m in IDS]].sum().to_dict(), "пустых", int((v1[[f"y_{m}" for m in IDS]].sum(1) == 0).sum()))
