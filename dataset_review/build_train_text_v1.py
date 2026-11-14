"""Собирает неизменяемую версию train_text_v1 из исходного JSONL."""

import hashlib
import json
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from train_text_cleaning import PRESERVED_NAMES_AND_CODES, STEPS, mixed_tokens

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw/public-train-annotated.jsonl"
DESTINATION = ROOT / "data/processed/train_text_v1"
VERSION = DESTINATION.name
TEXT_FIELDS = ("title", "description")
SCHEMA = pa.schema([
    pa.field("item_id", pa.int64(), nullable=False),
    pa.field("title", pa.string(), nullable=False),
    pa.field("description", pa.string(), nullable=False),
    pa.field("found_microcategories", pa.list_(pa.struct([
        pa.field("mcId", pa.int64(), nullable=False),
        pa.field("label", pa.string(), nullable=False),
    ])), nullable=False),
    pa.field("source_line", pa.int64(), nullable=False),
])


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    if DESTINATION.exists():
        raise SystemExit(f"Версия уже существует: {DESTINATION.relative_to(ROOT)}")

    rows = []
    source_hash = hashlib.sha256()
    step_counts = {name: Counter() for name, _, _ in STEPS}
    changed_fields = Counter()
    changed_rows = 0
    preserved_tokens = Counter()

    with SOURCE.open("rb") as source:
        for source_line, raw_line in enumerate(source, 1):
            source_hash.update(raw_line)
            raw = json.loads(raw_line)
            if set(raw) != set(SCHEMA.names) - {"source_line"}:
                raise ValueError(f"Строка {source_line}: неожиданные поля источника")
            clean = dict(raw, source_line=source_line)
            active_steps = set()
            for field in TEXT_FIELDS:
                text = raw[field]
                for name, transform, _ in STEPS:
                    new_text = transform(text)
                    if new_text != text:
                        step_counts[name][field] += 1
                        active_steps.add(name)
                    text = new_text
                if not text:
                    raise ValueError(f"Строка {source_line}: пустое поле {field} после очистки")
                changed_fields[field] += text != raw[field]
                preserved_tokens.update(mixed_tokens(text))
                clean[field] = text
            for name in active_steps:
                step_counts[name]["records"] += 1
            changed_rows += any(clean[field] != raw[field] for field in TEXT_FIELDS)
            rows.append(clean)

    unexpected_tokens = preserved_tokens.keys() - PRESERVED_NAMES_AND_CODES
    if unexpected_tokens:
        raise ValueError(f"Новые смешанные формы требуют проверки: {sorted(unexpected_tokens)}")

    with tempfile.TemporaryDirectory(prefix=f".{VERSION}-", dir=DESTINATION.parent) as temporary:
        staging = Path(temporary) / VERSION
        staging.mkdir()
        dataset = staging / "train.parquet"
        pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), dataset, compression="zstd")

        code_paths = (
            Path(__file__).resolve(),
            Path(__file__).resolve().with_name("train_text_cleaning.py"),
            Path(__file__).resolve().with_name("emoji_ranges.txt"),
        )
        manifest = {
            "version": VERSION,
            "source": {
                "path": SOURCE.relative_to(ROOT).as_posix(),
                "sha256": source_hash.hexdigest(),
                "records": len(rows),
            },
            "dataset": {
                "path": (DESTINATION / dataset.name).relative_to(ROOT).as_posix(),
                "sha256": sha256(dataset),
                "records": len(rows),
            },
            "lineage": {"item_id": "исходный item_id", "source_line": "номер строки JSONL с единицы"},
            "transformed_fields": list(TEXT_FIELDS),
            "changed_records": changed_rows,
            "changed_fields": dict(changed_fields),
            "transformations": [
                {
                    "id": name,
                    "description": description,
                    "changed_records": step_counts[name]["records"],
                    "changed_fields": {field: step_counts[name][field] for field in TEXT_FIELDS},
                }
                for name, _, description in STEPS
            ],
            "preserved_mixed_names_and_codes": dict(sorted(preserved_tokens.items())),
            "build_files_sha256": {path.relative_to(ROOT).as_posix(): sha256(path) for path in code_paths},
            "unicode_version": unicodedata.unidata_version,
            "pyarrow_version": pa.__version__,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(DESTINATION)

    print(f"{VERSION}: {changed_rows} из {len(rows)} записей изменились")
    print(f"SHA-256: {manifest['dataset']['sha256']}")


if __name__ == "__main__":
    main()
