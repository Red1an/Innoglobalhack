import sys
import pandas as pd
from pathlib import Path
from openpyxl.styles import Alignment, PatternFill

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "test_versions"
AGENT = ROOT / "dataset_review" / "review"
IDS = [102, 103, 104, 105, 106]
NAMES = {102: "Штукатурные работы", 103: "Плиточные работы", 104: "Сантехника", 105: "Электрика", 106: "Поклейка обоев и малярные работы"}
VERSION = sys.argv[1] if len(sys.argv) > 1 else "v3"

test = pd.read_parquet(OUT / f"test_{VERSION}.parquet")
uid_map = pd.read_json(AGENT / "agent_input" / "uid_map.jsonl", lines=True)
row_to_uid = {r: u.uid for u in uid_map.itertuples() for r in u.row_ids}
agent = pd.concat([pd.read_json(AGENT / f"mc_{m}.jsonl", lines=True) for m in IDS]).set_index(["uid", "mcId"])
manual = pd.read_csv(OUT / f"test_{VERSION}_manual.csv")
changed = pd.concat([pd.read_csv(OUT / f"test_v{n}_changes.csv").assign(version=f"v{n}") for n in (2, 3)])

notes = {}
for r in manual.itertuples():
    notes.setdefault(r.row_id, []).append(f"{r.mcId}: {r.note}")
for r in changed.itertuples():
    if r.row_id in set(test.row_id):
        notes.setdefault(r.row_id, []).append(f"{r.mcId}: {r.old}→{r.new} в {r.version}, {r.reason}")

rows = []
for t in test.itertuples():
    uid = row_to_uid[t.row_id]
    record = {"row_id": t.row_id, "копии в исходном test": ", ".join(map(str, t.row_ids)), "description": t.description}
    for m in IDS:
        record[f"{m} {NAMES[m]}"] = int(getattr(t, f"y_{m}"))
    record["на что обратить внимание"] = "\n".join(notes.get(t.row_id, []))
    for m in IDS:
        a = agent.loc[(uid, m)]
        record[f"цитата {m}"] = a.quote if a.quote else ""
    rows.append(record)

frame = pd.DataFrame(rows)
path = OUT / f"test_{VERSION}_review.xlsx"
with pd.ExcelWriter(path, engine="openpyxl") as writer:
    frame.to_excel(writer, index=False, sheet_name="review")
    sheet = writer.sheets["review"]
    widths = [8, 12, 90, 11, 11, 11, 11, 11, 45, 35, 35, 35, 35, 35]
    for index, width in enumerate(widths):
        sheet.column_dimensions[chr(ord("A") + index)].width = width
    wrap = Alignment(wrap_text=True, vertical="top")
    flag = PatternFill("solid", fgColor="FFF2CC")
    for row in sheet.iter_rows(min_row=1):
        for cell in row:
            cell.alignment = wrap
        if row[8].value and row[0].row > 1:
            row[8].fill = flag
    sheet.freeze_panes = "D2"
print(path, frame.shape, "строк с пометками", int((frame["на что обратить внимание"] != "").sum()))
