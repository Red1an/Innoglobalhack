import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "test"

AGENT_FOUND = "агент нашёл услугу в тексте"
BRIGADE = "правило 1: специалист назван в составе бригады"
TAIL = "правило 3: услуга есть в SEO-хвосте"
DESIGN = "правило 5: проектирование без монтажа считается"
WRONG = "агент: упоминание не является услугой"

CHANGES = [
    (6, 106, 1, AGENT_FOUND), (11, 104, 1, AGENT_FOUND), (18, 106, 1, AGENT_FOUND), (21, 103, 1, AGENT_FOUND),
    (24, 103, 1, AGENT_FOUND), (27, 104, 1, AGENT_FOUND), (27, 105, 1, AGENT_FOUND),
    (30, 102, 1, AGENT_FOUND), (30, 103, 1, AGENT_FOUND), (30, 104, 1, AGENT_FOUND), (30, 105, 1, AGENT_FOUND), (30, 106, 1, AGENT_FOUND),
    (32, 102, 1, AGENT_FOUND), (32, 103, 1, AGENT_FOUND), (32, 104, 1, AGENT_FOUND), (32, 105, 1, AGENT_FOUND), (32, 106, 1, AGENT_FOUND),
    (34, 103, 1, AGENT_FOUND), (34, 104, 1, AGENT_FOUND), (34, 105, 1, AGENT_FOUND), (43, 106, 1, AGENT_FOUND),
    (56, 104, 1, AGENT_FOUND), (56, 105, 1, AGENT_FOUND), (103, 104, 1, AGENT_FOUND), (103, 105, 1, AGENT_FOUND),
    (134, 106, 1, AGENT_FOUND), (136, 104, 1, AGENT_FOUND), (136, 105, 1, AGENT_FOUND),
    (20, 102, 1, BRIGADE), (20, 103, 1, BRIGADE), (20, 104, 1, BRIGADE), (20, 105, 1, BRIGADE), (20, 106, 1, BRIGADE),
    (21, 102, 1, TAIL), (150, 102, 1, TAIL), (188, 102, 1, TAIL),
    (26, 105, 1, DESIGN),
    (41, 103, 0, WRONG), (114, 106, 0, WRONG),
]

MANUAL = {
    (75, 104): "правило 2: монтаж упомянут мимоходом, в объяснении про мебель",
    (75, 105): "правило 2: монтаж упомянут мимоходом, в объяснении про мебель",
    (96, 103): "правило 2: плитка упомянута только в описании фотоотчёта",
    (96, 105): "правило 2: проводка упомянута только в описании фотоотчёта",
    (103, 102): "правило 2: штукатурка упомянута только в технадзоре, в работах есть выравнивание",
    (103, 103): "правило 2: плитка упомянута только в технадзоре",
    (189, 106): "правило 2: обои упомянуты только в характеристиках примера объекта",
    (76, 105): "не покрыто правилами: тёплый пол без указания, электрический ли он",
}


def main():
    v2 = pd.read_parquet(OUT / "test_v2.parquet")
    v3 = v2.set_index("row_id")
    log = []
    for row_id, mc, new, reason in CHANGES:
        old = int(v3.at[row_id, f"y_{mc}"])
        assert old != new, (row_id, mc)
        v3.at[row_id, f"y_{mc}"] = new
        log.append({"row_id": row_id, "mcId": mc, "old": old, "new": new, "reason": reason})
    v3 = v3.reset_index()
    v3.to_parquet(OUT / "test_v3.parquet", index=False)
    pd.DataFrame(log).to_csv(OUT / "test_v3_changes.csv", index=False)
    pd.DataFrame([{"row_id": r, "mcId": m, "note": n} for (r, m), n in MANUAL.items()]).to_csv(OUT / "test_v3_manual.csv", index=False)
    ys = [c for c in v3 if c.startswith("y_")]
    print("v3", v3.shape, "изменённых пар", len(log), "на ручную проверку", len(MANUAL))
    print("метки v2", v2[ys].sum().to_dict())
    print("метки v3", v3[ys].sum().to_dict(), "пустых", int((v3[ys].sum(1) == 0).sum()))


if __name__ == "__main__":
    main()
