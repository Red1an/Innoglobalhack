"""Снимает неподтверждённую метку 102 после дедупликации train."""

import argparse
import csv
import json
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from build_train_text_v1 import ROOT, sha256

SOURCE = ROOT / "data/processed/train_dedup_v1"
DESTINATION = ROOT / "data/processed/train_labels_102_v1"
CATEGORY = 102
PLASTER_MARKER = re.compile(r"штукатур|маяк", re.I)
LEVELING = re.compile(
    r"\b(?:выравн|выровн|ровня)\w*(?:[^\S\n]+\w+){0,2}[^\S\n]+(?:стен|потол|поверх)\w*",
    re.I,
)
FILLER = re.compile(r"\bш(?:пакл|патл)\w*", re.I)
CONFIRMED_ALTERNATIVES = {4647: "Венецианки, разные"}


def label_counts(rows):
    return Counter(entry["mcId"] for row in rows for entry in row["found_microcategories"])


def artifact(path, records, published_path=None):
    return {
        "path": str(published_path or path),
        "sha256": sha256(path),
        "records": records,
    }


def verified_path(info):
    path = ROOT / info["path"]
    if sha256(path) != info["sha256"]:
        raise ValueError(f"SHA-256 расходится с манифестом: {path}")
    return path


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build(destination):
    if destination.exists():
        raise ValueError(f"Версия уже существует: {destination}")

    parent = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    source_info = parent["dataset"]
    table = pq.read_table(verified_path(source_info))
    if len(table) != source_info["records"]:
        raise ValueError("Размер исходного Parquet расходится с манифестом")
    verified_path(parent["raw_source"])
    duplicates_info = parent["removed_duplicates"]
    with verified_path(duplicates_info).open(encoding="utf-8", newline="") as stream:
        duplicates = list(csv.DictReader(stream))
    if len(duplicates) != duplicates_info["records"]:
        raise ValueError("Число удалённых копий расходится с манифестом")

    copies = defaultdict(list)
    for duplicate in duplicates:
        copies[int(duplicate["kept_source_line"])].append(duplicate)

    rows = table.to_pylist()
    before = label_counts(rows)
    changes, changed_copies, exceptions = [], [], []
    for row in rows:
        labels = row["found_microcategories"]
        if not any(entry["mcId"] == CATEGORY for entry in labels):
            continue
        description = row["description"]
        line = row["source_line"]
        if PLASTER_MARKER.search(description):
            continue
        leveling = LEVELING.search(description)
        filler = FILLER.search(description)
        if leveling is None and filler is None:
            continue
        if line in CONFIRMED_ALTERNATIVES:
            evidence = CONFIRMED_ALTERNATIVES[line]
            if evidence not in description:
                raise ValueError(f"Цитата для исключения отсутствует в description: {line}")
            exceptions.append({
                "source_line": line, "item_id": row["item_id"], "evidence": evidence,
                "reason": "Венецианская штукатурка явно предложена в списке работ.",
            })
            continue

        reason = []
        if leveling:
            reason.append("выравнивание без указания штукатурки")
        if filler:
            reason.append("шпаклёвка")
        row["found_microcategories"] = [entry for entry in labels if entry["mcId"] != CATEGORY]
        changes.append({
            "source_line": line,
            "item_id": row["item_id"],
            "mcId": CATEGORY,
            "old": 1,
            "new": 0,
            "reason": " и ".join(reason),
            "leveling_quote": leveling.group() if leveling else "",
            "filler_quote": filler.group() if filler else "",
            "remaining_labels": ",".join(str(entry["mcId"]) for entry in row["found_microcategories"]),
            "removed_duplicates": len(copies[line]),
        })
        changed_copies.extend(copies[line])

    if {row["source_line"] for row in exceptions} != set(CONFIRMED_ALTERNATIVES):
        raise ValueError("Не все проверенные исключения найдены")
    if not changes:
        raise ValueError("Ни одна запись не подходит под правило")

    relative_destination = DESTINATION.relative_to(ROOT)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as temporary:
        staging = Path(temporary) / destination.name
        staging.mkdir()
        dataset = staging / "train.parquet"
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), dataset, compression="zstd")
        change_report = staging / "label_changes.csv"
        write_csv(change_report, changes)
        copies_report = staging / "affected_duplicates.csv"
        write_csv(copies_report, sorted(changed_copies, key=lambda row: int(row["source_line"])))

        manifest = {
            "version": DESTINATION.name,
            "source": source_info,
            "raw_source": parent["raw_source"],
            "source_removed_duplicates": duplicates_info,
            "dataset": artifact(dataset, len(rows), relative_destination / dataset.name),
            "label_changes": artifact(change_report, len(changes), relative_destination / change_report.name),
            "affected_duplicates": artifact(
                copies_report, len(changed_copies), relative_destination / copies_report.name
            ),
            "transformations": [{
                "id": "remove_unsupported_102",
                "description": (
                    "Снять 102, если в полном description указана шпаклёвка или выравнивание стен, "
                    "потолков или поверхностей, а указания на штукатурку и маяки отсутствуют. "
                    "Венецианская штукатурка в проверенной строке 4647 подтверждает 102."
                ),
                "retained_exceptions": exceptions,
                "labels_before": dict(sorted(before.items())),
                "labels_after": dict(sorted(label_counts(rows).items())),
            }],
            "build_files_sha256": {
                name: sha256(ROOT / name)
                for name in (
                    "dataset_review/build_train_labels_102_v1.py",
                    "dataset_review/build_train_text_v1.py",
                )
            },
            "pyarrow_version": pa.__version__,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(destination)

    print(f"Сняли 102 у {len(changes)} из {len(rows)} записей. У них {len(changed_copies)} удалённых копий.")
    print(f"SHA-256: {manifest['dataset']['sha256']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DESTINATION)
    build(parser.parse_args().out)
