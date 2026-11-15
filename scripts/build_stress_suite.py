"""Build the diagnostic stress_suite_v1 benchmark in data/test/stress_suite_v1/."""
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "experiments"))
sys.path.insert(0, str(ROOT / "scripts"))

from bench import clean, digest, gold_overlap_key, key  # noqa: E402
from stress_suite_data import HARD_EMPTY_NEW  # noqa: E402
from stress_suite_data_decoupled import (  # noqa: E402
    PAIR_102_104,
    PAIR_102_105,
    PAIR_103_105,
    SOLO_102_NEW,
    SOLO_104_NEW,
    SOLO_105_NEW,
)
from stress_suite_data_dirty_and_mcid import (  # noqa: E402
    DIRTY_ANTIFRAUD_TAILS,
    DIRTY_GLUED_CAPS,
    DIRTY_HOMOGLYPHS,
    MENTOR_BRIGADE_AND_EXTERIOR,
    MENTOR_SEO_TAGS,
    NON_101_SOURCE_MCID,
    apply_homoglyphs,
)
from stress_suite_data_traps import (  # noqa: E402
    TRAPS_EXPLICIT_NEGATION,
    TRAPS_LEVELING_NO_PLASTER,
    TRAPS_PREP_ONLY,
    TRAPS_THIRD_PARTY,
    TRAPS_WARM_FLOOR_AMBIG,
)

OUT_DIR = ROOT / "data" / "test" / "stress_suite_v1"
GOLD_PATH = ROOT / "data" / "test" / "test_guide_v1" / "test_gold.parquet"
TRAIN_PATH = ROOT / "data" / "processed" / "train_guide_v1" / "train.parquet"

# 15 empty rows from test_gold free of шпаклёвка/106 ambiguity under microcategories-guide.md
HARD_GOLD_EMPTY_IDS = [1, 2, 17, 25, 26, 41, 66, 75, 96, 113, 114, 131, 166, 186, 189]

# Safe Avito ad intro/outro blocks with ZERO trigger words for 102-106
SAFE_INTROS = [
    "Здравствуйте! Меня зовут Алексей, работаю с напарником напрямую без посредников и диспетчеров.",
    "Доброго времени суток! Частный мастер с опытом более 10 лет, работаю аккуратно и на совесть.",
    "Приветствую! Своя слаженная команда мастеров, берём объекты в черте города и ближайшем пригороде.",
    "Здравствуйте! Работаем официально по договору с физлицами и организациями, возможен безналичный расчёт.",
    "Добрый день! Выезд на замер и подробная консультация — бесплатно в удобный для вас день.",
    "Приветствуем вас! Помогаем сэкономить время и бюджет: точный расчёт сметы до начала работ.",
    "Здравствуйте! Выполняем заказы любой сложности — от небольших бытовых задач до крупных объектов.",
    "Добрый день! Работаем без выходных с 8:00 до 21:00, всегда на связи в чате и по телефону.",
]

SAFE_OUTROS = [
    "Преимущества работы с нами:\n— фиксированная стоимость в смете без неожиданных доплат в процессе;\n— поэтапная оплата строго по факту выполненных и принятых вами этапов;\n— собственный профессиональный инструмент и порядок на объекте после каждого рабочего дня.",
    "Почему выбирают нас:\n1. Соблюдаем оговорённые сроки и прописываем их в договоре.\n2. Помогаем с закупкой и доставкой качественных расходников со скидкой поставщиков.\n3. Даём письменную гарантию на выполненные работы до 24 месяцев.",
    "Организационные моменты:\n• Выезжаем на осмотр в день обращения или на следующее утро.\n• Фото- и видеоотчёты каждый день в мессенджер, если вы находитесь в другом городе.\n• Бережное отношение к вашему имуществу и уважение к режиму тишины в доме.",
    "Условия сотрудничества:\n- Работаем без предоплаты за воздух — расчёт после проверки результата.\n- Убираем за собой весь рабочий мусор в мешки.\n- Возможен срочный выезд при необходимости, звоните или пишите в сообщения!",
    "Дополнительно: составляем подробный график выполнения задач, заранее предупреждаем о завозе материалов, предоставляем чеки из строительных гипермаркетов без наценок.",
    "Все мастера с опытом от 7 лет, без вредных привычек, ответственные и пунктуальные. Добавляйте объявление в избранное, чтобы не потерять контакт!",
    "Оплата наличными или переводом на карту после приёмки этапа. Звоните с 09:00 до 21:00 — с радостью отвечу на все вопросы и сориентирую по ориентировочной стоимости.",
    "Работаем в новостройках и во вторичном жилье, берём объекты в любом районе города. Ценим свою репутацию и всегда доводим дело до аккуратного результата.",
]


def enrich_text(text: str, rng: random.Random, target_min: int = 160, force_long: bool = False) -> str:
    """Naturally vary length (150..2500 chars) while keeping SEO/antifraud tails at the end."""
    text = text.strip()
    tail = ""
    for marker in ("\n\n#", "\n\nПоисковые", "\n\nТеги", "\n\nКлючевые", "\n\nЗапросы", "\n\nДля поиска", "\n\nХарактеристика:"):
        if marker in text:
            idx = text.index(marker)
            text, tail = text[:idx], text[idx:]
            break
    if len(text + tail) < target_min or force_long:
        parts = []
        if rng.random() < 0.65 or force_long:
            parts.append(rng.choice(SAFE_INTROS))
        parts.append(text)
        if len("\n\n".join(parts) + tail) < target_min or force_long or rng.random() < 0.45:
            outros = rng.sample(SAFE_OUTROS, k=2 if force_long else 1)
            parts.extend(outros)
        text = "\n\n".join(parts)
    result = (text + tail).strip()
    if len(result) < 150:
        result = f"{result}\n\n{rng.choice(SAFE_OUTROS)}"
    return result


def make_row(
    row_id: str,
    mc_id: int,
    description: str,
    labels: list[int],
    slice_name: str,
    subslice: str,
    note: str,
    origin_split: str,
    labels_strict: list[int] | None = None,
    labels_seo: list[int] | None = None,
) -> dict:
    labs = sorted(set(int(x) for x in labels))
    assert mc_id not in labs, f"mcId {mc_id} in labels {labs} for {row_id}"
    l_strict = sorted(set(int(x) for x in (labels if labels_strict is None else labels_strict)))
    l_seo = sorted(set(int(x) for x in (labels if labels_seo is None else labels_seo)))
    row = {
        "row_id": row_id,
        "mcId": int(mc_id),
        "description": clean(description),
        "labels": labs,
        "labels_strict": l_strict,
        "labels_seo": l_seo,
        "slice": slice_name,
        "subslice": subslice,
        "note": note,
        "origin_split": origin_split,
    }
    for c in range(102, 107):
        row[f"y_{c}"] = 1 if c in labs else 0
    return row


DEV_BAD_RE = {
    104: re.compile(r"шпакл|шпатл|плитк|кафел|керамогранит|штукатур|электр|проводк|розетк|обои|обоев|покраск|окраск|маляр|штроб", re.I),
    105: re.compile(r"шпакл|шпатл|плитк|кафел|керамогранит|фартук|штукатур|сантехн|водопровод|канализац|воды|труб|обои|обоев|покраск|окраск|маляр", re.I),
    102: re.compile(r"шпакл|шпатл|стеклохолст|плитк|кафел|керамогранит|сантехн|водопровод|канализац|электр|проводк|розетк|обои|обоев|покраск|окраск|маляр", re.I),
}


def load_dev_solos(train_rows: list[dict], target_label: int, limit: int) -> list[dict]:
    """Extract unique real dev rows having exact solo label [target_label] clean under guide."""
    seen = set()
    selected = []
    bad_pat = DEV_BAD_RE[target_label]
    for r in train_rows:
        if r.get("origin") == "synthetic" and r.get("source_item_id") is not None:
            continue
        tk = key(r["description"])
        if tk in seen or int(digest(tk)[:8], 16) % 5 != 0:
            continue
        labs = sorted(int(e["mcId"]) for e in r["found_microcategories"])
        if labs == [target_label] and not bad_pat.search(r["description"]):
            seen.add(tk)
            selected.append(r)
            if len(selected) >= limit:
                break
    if len(selected) < limit:
        raise RuntimeError(f"Expected {limit} dev solo {target_label} rows, found {len(selected)}")
    return selected


def build_slices(gold_rows: list[dict], train_rows: list[dict]) -> list[dict]:
    rng = random.Random(42)
    rows: list[dict] = []

    # 1. gold_core (80)
    for i, r in enumerate(gold_rows):
        labs = [c for c in range(102, 107) if r[f"y_{c}"] == 1]
        rows.append(make_row(
            f"gold_core_{i:03d}", 101, r["description"], labs,
            "gold_core", "test_gold", f"test_gold row_id={r['row_id']}", "test_gold",
        ))

    # 2. hard_empty (40 = 15 test_gold + 25 new)
    gold_by_id = {r["row_id"]: r for r in gold_rows}
    for i, gid in enumerate(HARD_GOLD_EMPTY_IDS):
        r = gold_by_id[gid]
        assert sum(r[f"y_{c}"] for c in range(102, 107)) == 0
        rows.append(make_row(
            f"hard_empty_{i:03d}", 101, r["description"], [],
            "hard_empty", "test_gold_empty", f"test_gold empty row_id={gid}", "test_gold",
        ))
    for j, item in enumerate(HARD_EMPTY_NEW):
        desc = enrich_text(item["description"], rng, force_long=(j % 5 == 0))
        rows.append(make_row(
            f"hard_empty_{15 + j:03d}", 101, desc, item["labels"],
            "hard_empty", item["subslice"], item["note"], "synthetic_stress",
        ))

    # 3. decoupled_solo_and_pairs (100 = 50 solo + 50 atypical pairs)
    idx = 0
    for target, dev_n, new_list in [
        (104, 5, SOLO_104_NEW[:15]),
        (105, 7, SOLO_105_NEW[:13]),
        (102, 2, SOLO_102_NEW[:8]),
    ]:
        for r in load_dev_solos(train_rows, target, dev_n):
            rows.append(make_row(
                f"decoupled_{idx:03d}", 101, r["description"], [target],
                "decoupled_solo_and_pairs", f"solo_{target}", f"dev item_id={r['item_id']}", "dev",
            ))
            idx += 1
        for text, note in new_list:
            desc = enrich_text(text, rng, force_long=(idx % 6 == 0))
            rows.append(make_row(
                f"decoupled_{idx:03d}", 101, desc, [target],
                "decoupled_solo_and_pairs", f"solo_{target}", note, "synthetic_stress",
            ))
            idx += 1

    for labs, sub, pair_list in [
        ([102, 104], "pair_102_104", PAIR_102_104),
        ([103, 105], "pair_103_105", PAIR_103_105),
        ([102, 105], "pair_102_105", PAIR_102_105),
    ]:
        for text, note in pair_list:
            desc = enrich_text(text, rng, force_long=(idx % 7 == 0))
            rows.append(make_row(
                f"decoupled_{idx:03d}", 101, desc, labs,
                "decoupled_solo_and_pairs", sub, note, "synthetic_stress",
            ))
            idx += 1

    # 4. context_traps_and_negations (80)
    t_idx = 0
    for sub, trap_list in [
        ("explicit_negation", TRAPS_EXPLICIT_NEGATION),
        ("third_party", TRAPS_THIRD_PARTY),
        ("prep_only", TRAPS_PREP_ONLY),
        ("leveling_no_plaster", TRAPS_LEVELING_NO_PLASTER),
        ("warm_floor_ambig", TRAPS_WARM_FLOOR_AMBIG),
    ]:
        for labs, text, note in trap_list:
            desc = enrich_text(text, rng, force_long=(t_idx % 8 == 0))
            rows.append(make_row(
                f"traps_{t_idx:03d}", 101, desc, labs,
                "context_traps_and_negations", sub, note, "synthetic_stress",
            ))
            t_idx += 1
    return rows


def build_remaining_slices(rng: random.Random) -> list[dict]:
    rows: list[dict] = []

    # 5. dirty_text_and_homoglyphs (50 = 20 homoglyphs + 15 glued_caps + 15 antifraud)
    d_idx = 0
    for labs, words, text in DIRTY_HOMOGLYPHS:
        desc = apply_homoglyphs(enrich_text(text, rng, force_long=(d_idx % 5 == 0)), words)
        rows.append(make_row(
            f"dirty_{d_idx:03d}", 101, desc, labs,
            "dirty_text_and_homoglyphs", "homoglyphs", f"latin homoglyphs in {words}", "synthetic_stress",
        ))
        d_idx += 1
    for labs, sub, note, text in DIRTY_GLUED_CAPS:
        desc = enrich_text(text, rng, force_long=False)
        rows.append(make_row(
            f"dirty_{d_idx:03d}", 101, desc, labs,
            "dirty_text_and_homoglyphs", sub, note, "synthetic_stress",
        ))
        d_idx += 1
    for labs, sub, note, text in DIRTY_ANTIFRAUD_TAILS:
        desc = enrich_text(text, rng, force_long=(d_idx % 4 == 0))
        rows.append(make_row(
            f"dirty_{d_idx:03d}", 101, desc, labs,
            "dirty_text_and_homoglyphs", sub, note, "synthetic_stress",
        ))
        d_idx += 1

    # 6. non_101_source_mcid (50)
    for i, (mc_id, labs, text, note) in enumerate(NON_101_SOURCE_MCID):
        desc = enrich_text(text, rng, force_long=(i % 7 == 0))
        rows.append(make_row(
            f"non101_{i:03d}", mc_id, desc, labs,
            "non_101_source_mcid", f"mcid_{mc_id}", note, "synthetic_stress",
        ))

    # 7. mentor_borderline_switches (40 = 20 seo_tags_only + 20 brigade_and_exterior)
    m_idx = 0
    for l_strict, l_seo, text, note in MENTOR_SEO_TAGS:
        desc = enrich_text(text, rng, force_long=(m_idx % 6 == 0))
        rows.append(make_row(
            f"mentor_{m_idx:03d}", 101, desc, l_strict,
            "mentor_borderline_switches", "seo_tags_only", note, "synthetic_stress",
            labels_strict=l_strict, labels_seo=l_seo,
        ))
        m_idx += 1
    for labs, text, note in MENTOR_BRIGADE_AND_EXTERIOR:
        desc = enrich_text(text, rng, force_long=(m_idx % 6 == 0))
        rows.append(make_row(
            f"mentor_{m_idx:03d}", 101, desc, labs,
            "mentor_borderline_switches", "brigade_and_exterior", note, "synthetic_stress",
            labels_strict=labs, labels_seo=labs,
        ))
        m_idx += 1

    return rows


EXPECTED_COUNTS = {
    "gold_core": 80,
    "hard_empty": 40,
    "decoupled_solo_and_pairs": 100,
    "context_traps_and_negations": 80,
    "dirty_text_and_homoglyphs": 50,
    "non_101_source_mcid": 50,
    "mentor_borderline_switches": 40,
}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    gold_rows = pq.read_table(GOLD_PATH).to_pylist()
    train_rows = pq.read_table(TRAIN_PATH).to_pylist()

    rng = random.Random(42)
    rows = build_slices(gold_rows, train_rows) + build_remaining_slices(rng)

    counts = Counter(r["slice"] for r in rows)
    assert dict(counts) == EXPECTED_COUNTS, f"Slice count mismatch: {dict(counts)} != {EXPECTED_COUNTS}"

    # Verify zero overlap with train_relabel_v2_synth_v1 for all synthetic_stress rows
    normalize = gold_overlap_key()
    train_keys = {key(r["description"]) for r in train_rows}
    train_norm_keys = {normalize(r["description"]) for r in train_rows}

    seen_new_keys: set[str] = set()
    for r in rows:
        if r["origin_split"] == "synthetic_stress":
            tk = key(r["description"])
            nk = normalize(r["description"])
            if tk in train_keys or nk in train_norm_keys:
                raise AssertionError(f"Overlap with train detected on {r['row_id']}")
            if tk in seen_new_keys:
                raise AssertionError(f"Duplicate synthetic_stress text on {r['row_id']}")
            seen_new_keys.add(tk)
            n_chars = len(r["description"])
            assert 150 <= n_chars <= 2500, f"Length out of [150, 2500] on {r['row_id']}: {n_chars}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, OUT_DIR / "stress_suite_v1.parquet")
    write_jsonl(OUT_DIR / "stress_suite_v1.jsonl", rows)

    by_slice: dict[str, list[dict]] = {}
    for r in rows:
        by_slice.setdefault(r["slice"], []).append(r)
    for slice_name, slice_rows in by_slice.items():
        pq.write_table(pa.Table.from_pylist(slice_rows), OUT_DIR / f"{slice_name}.parquet")
        write_jsonl(OUT_DIR / f"{slice_name}.jsonl", slice_rows)

    manifest = {
        "total_rows": len(rows),
        "gold_source": str(GOLD_PATH.relative_to(ROOT)),
        "train_source": str(TRAIN_PATH.relative_to(ROOT)),
        "slices": {k: len(v) for k, v in by_slice.items()},
        "origin_splits": dict(Counter(r["origin_split"] for r in rows)),
        "synthetic_stress_train_overlap": 0,
        "synthetic_stress_min_len": min(len(r["description"]) for r in rows if r["origin_split"] == "synthetic_stress"),
        "synthetic_stress_max_len": max(len(r["description"]) for r in rows if r["origin_split"] == "synthetic_stress"),
        "sha256_jsonl": digest(rows),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
