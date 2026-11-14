"""Собирает train_dedup_v1 из проверенной версии train_text_v1."""

import csv
import json
import re
import tempfile
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from build_train_text_v1 import ROOT, sha256
from train_text_cleaning import normalize_whitespace

SOURCE = ROOT / "data/processed/train_text_v1/train.parquet"
DESTINATION = ROOT / "data/processed/train_dedup_v1"
VISUAL_SPACES = str.maketrans(dict.fromkeys("\u2800\u3164\u115f\u1160", " "))
STOCK_SUFFIX = re.compile(
    r'Место на складе:\s*[a-z]-\d+\s+Номер продаваемого оборудования:\s*[a-z]-\d+'
    r'\s+Марка:\s*[a-z]-\d+\s+Размеры (?:оборудования|товара) на фото:'
    r'\s*[\d,.*\s]+(?:Код товара:\s*[A-Z]+-\d+)?\s*', re.I
)
REVIEWED_STOCK_SUFFIX = "Место на складе 79 ГОСТ01295 Артикул Ф- 1295 Бригада №53"
LINE_MARKUP = re.compile(r"^(?:#{1,6}|[—•-])[ \t]+(?=[^\W\d_])", re.M)
BOLD_MARKUP = re.compile(r"\*\*([^*\n]+)\*\*")
SEO_MARKER = re.compile(r"для поиска|ключевые слова|поисковые запросы|нас (?:также )?ищут", re.I)


def without_stock(text):
    marker = re.search(r"место на складе", text, re.I)
    if marker and STOCK_SUFFIX.fullmatch(text, marker.start()):
        return text[:marker.start()].rstrip()
    if text.endswith(REVIEWED_STOCK_SUFFIX):
        return text[:-len(REVIEWED_STOCK_SUFFIX)].rstrip()
    return text


def replace_visual_spaces(text):
    replaced = text.translate(VISUAL_SPACES)
    return normalize_whitespace(replaced) if replaced != text else text


def whitespace_key(text):
    return " ".join(text.casefold().split())


def duplicate_key(text):
    text = BOLD_MARKUP.sub(r"\1", LINE_MARKUP.sub("", text))
    return whitespace_key(text).rstrip(".! ")


def label_signature(row):
    return sorted((entry["mcId"], entry["label"]) for entry in row["found_microcategories"])


def duplicate_reason(row, kept):
    if row["original_description"] == kept["original_description"]:
        return "exact"
    if row["description"] == kept["description"]:
        return "exact_after_cleanup"
    if whitespace_key(row["description"]) == whitespace_key(kept["description"]):
        return "case_or_whitespace"
    return "formatting"


def main():
    if DESTINATION.exists():
        raise SystemExit(f"Версия уже существует: {DESTINATION.relative_to(ROOT)}")

    source_manifest = json.loads(SOURCE.with_name("manifest.json").read_text(encoding="utf-8"))
    source_info = source_manifest["dataset"]
    if sha256(SOURCE) != source_info["sha256"]:
        raise ValueError("SHA-256 исходного Parquet расходится с манифестом")
    table = pq.read_table(SOURCE)
    if len(table) != source_info["records"]:
        raise ValueError("Число строк исходного Parquet расходится с манифестом")

    representatives = {}
    removed = []
    counts = Counter()
    descriptions_before, descriptions_clean, descriptions_folded = set(), set(), set()

    for row in sorted(table.to_pylist(), key=lambda r: r["source_line"]):
        original_description = row["description"]
        blank_normalized = replace_visual_spaces(original_description)
        row["description"] = without_stock(blank_normalized)
        original_title = row["title"]
        row["title"] = replace_visual_spaces(original_title)
        if not row["title"] or not duplicate_key(row["description"]):
            raise ValueError(f"Пустой текст в исходной строке {row['source_line']}")

        counts["visual_spaces_in_title"] += row["title"] != original_title
        counts["visual_spaces_in_description"] += blank_normalized != original_description
        counts["stock_suffix_removed"] += row["description"] != blank_normalized
        counts["seo_rows_before"] += bool(SEO_MARKER.search(original_description))
        descriptions_before.add(original_description)
        descriptions_clean.add(row["description"])
        descriptions_folded.add(whitespace_key(row["description"]))

        row["original_description"] = original_description
        kept = representatives.setdefault(duplicate_key(row["description"]), row)
        if kept is row:
            continue
        if label_signature(row) != label_signature(kept):
            raise ValueError(f"Конфликт меток между строками {kept['source_line']} и {row['source_line']}")
        removed.append({
            "source_line": row["source_line"],
            "item_id": row["item_id"],
            "kept_source_line": kept["source_line"],
            "kept_item_id": kept["item_id"],
            "reason": duplicate_reason(row, kept),
        })

    kept_rows = list(representatives.values())
    for row in kept_rows:
        del row["original_description"]
    counts["seo_rows_after"] = sum(bool(SEO_MARKER.search(row["description"])) for row in kept_rows)
    with tempfile.TemporaryDirectory(prefix=f".{DESTINATION.name}-", dir=DESTINATION.parent) as temporary:
        staging = Path(temporary) / DESTINATION.name
        staging.mkdir()
        dataset = staging / "train.parquet"
        pq.write_table(pa.Table.from_pylist(kept_rows, schema=table.schema), dataset, compression="zstd")

        report = staging / "removed_duplicates.csv"
        with report.open("w", newline="", encoding="utf-8") as stream:
            columns = ["source_line", "item_id", "kept_source_line", "kept_item_id", "reason"]
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(removed)

        def artifact(path, records):
            return {
                "path": (DESTINATION / path.name).relative_to(ROOT).as_posix(),
                "sha256": sha256(path),
                "records": records,
            }

        manifest = {
            "version": DESTINATION.name,
            "source": source_info,
            "raw_source": source_manifest["source"],
            "dataset": artifact(dataset, len(kept_rows)),
            "removed_duplicates": artifact(report, len(removed)),
            "lineage": {
                "item_id": "исходный item_id",
                "source_line": "номер строки исходного JSONL с единицы",
                "kept_source_line": "номер исходной строки сохранённой копии в removed_duplicates.csv",
            },
            "transformations": [
                {
                    "id": "visual_spaces",
                    "description": "U+2800, U+3164, U+115F и U+1160 заменены пробелом с нормализацией разделителей",
                    "changed_title_rows": counts["visual_spaces_in_title"],
                    "changed_description_rows": counts["visual_spaces_in_description"],
                },
                {
                    "id": "reviewed_stock_suffix",
                    "description": "Удалены только подтверждённые складские окончания в description",
                    "changed_description_rows": counts["stock_suffix_removed"],
                },
                {
                    "id": "deduplicate_description",
                    "description": (
                        "Полное совпадение очищенного description без учёта регистра, пробелов, "
                        "оформления Markdown и списка и точек и восклицательных знаков в конце. "
                        "В каждой группе сохранена самая ранняя исходная строка. "
                        "Расхождение меток прерывает сборку."
                    ),
                    "unique_original_descriptions": len(descriptions_before),
                    "unique_cleaned_descriptions": len(descriptions_clean),
                    "unique_casefolded_descriptions": len(descriptions_folded),
                    "unique_group_keys": len(kept_rows),
                    "removed_rows": len(removed),
                    "removed_by_reason_relative_to_kept": dict(Counter(row["reason"] for row in removed)),
                },
            ],
            "seo": {"rows_before": counts["seo_rows_before"], "rows_after": counts["seo_rows_after"]},
            "build_files_sha256": {
                f"dataset_review/{name}": sha256(Path(__file__).with_name(name))
                for name in (Path(__file__).name, "build_train_text_v1.py", "train_text_cleaning.py")
            },
            "pyarrow_version": pa.__version__,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(DESTINATION)

    print(f"{DESTINATION.name}: {len(table)} → {len(kept_rows)} строк, удалено {len(removed)} дублей")
    print(f"SHA-256: {manifest['dataset']['sha256']}")


if __name__ == "__main__":
    main()
