import sys
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / "dataset_review" / "review"
sys.path.insert(0, str(ROOT / "data" / "test_versions"))
from build_v2 import explicit_plaster

VERSION = sys.argv[1] if len(sys.argv) > 1 else "v2"
IDS = [102, 103, 104, 105, 106]
NAMES = {102: "Штукатурные работы", 103: "Плиточные работы", 104: "Сантехника", 105: "Электрика", 106: "Поклейка обоев и малярные работы"}

test = pd.read_parquet(ROOT / "data" / "test_versions" / f"test_{VERSION}.parquet")
uid_map = pd.read_json(REVIEW / "agent_input" / "uid_map.jsonl", lines=True)
row_to_uid = {r: u.uid for u in uid_map.itertuples() for r in u.row_ids}
agent = pd.concat([pd.read_json(REVIEW / f"mc_{m}.jsonl", lines=True) for m in IDS]).set_index(["uid", "mcId"])

records = []
for row in test.itertuples():
    uid = row_to_uid[row.row_id]
    plaster = explicit_plaster(row.description)
    for m in IDS:
        a = agent.loc[(uid, m)]
        verdict, comment = int(a.present), a.comment
        if m == 102 and verdict == 1 and not plaster:
            verdict, comment = 0, "по правилу команды: выравнивание без явной штукатурки не считается 102"
        records.append({
            "row_id": row.row_id, "row_ids": ", ".join(map(str, row.row_ids)), "uid": uid, "mcId": m, "category": NAMES[m],
            "current_label": int(getattr(row, f"y_{m}")), "agent_label": verdict, "doubt": bool(a.doubt) and verdict == int(a.present),
            "where": a["where"], "quote": a.quote, "agent_comment": comment,
            "description": row.description.replace("\n", " / "), "decision": None,
        })
full = pd.DataFrame(records)
full.to_parquet(REVIEW / f"agent_vs_labels_{VERSION}.parquet", index=False)


def reason(r):
    if r.agent_label == 1 and r.current_label == 0:
        return "агент нашёл, метки нет"
    if r.agent_label == 0 and r.current_label == 1:
        return "метка есть, агент не нашёл"
    return "сомнение агента"


diff = full[(full.current_label != full.agent_label) | full.doubt].copy()
diff.insert(5, "reason", diff.apply(reason, axis=1))
diff = diff.sort_values(["reason", "mcId", "row_id"])
out = REVIEW / f"test_agent_disagreements_{VERSION}.xlsx"
with pd.ExcelWriter(out, engine="openpyxl") as writer:
    diff.to_excel(writer, index=False, sheet_name="review")
    sheet = writer.sheets["review"]
    for column, width in {"E": 28, "F": 26, "K": 60, "L": 40, "M": 100}.items():
        sheet.column_dimensions[column].width = width
    sheet.freeze_panes = "A2"

print(VERSION, "пар:", len(full), "совпадает:", round((full.current_label == full.agent_label).mean(), 3), "строк на проверку:", len(diff))
print(diff.groupby(["mcId", "reason"]).size().unstack(fill_value=0))
