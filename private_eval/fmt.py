"""Organizer file formats: input testcase,description and answer testcase,microcats,descriptions."""
import csv
import json
from pathlib import Path

HEADER_IN = ["testcase", "description"]
HEADER_OUT = ["testcase", "microcats", "descriptions"]
ALLOWED = set(range(101, 107))


def read_input(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(HEADER_IN) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: нет колонок {sorted(missing)}, есть {reader.fieldnames}")
        rows = []
        for r in reader:
            mc = r.get("mcId")
            rows.append({"testcase": r["testcase"], "description": r["description"] or "",
                         "mcId": int(mc) if mc not in (None, "") else None})
    ids = [r["testcase"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: повторяются testcase")
    return rows


def write_answer(path, records):
    """records: dicts with testcase, labels (sorted ints), descriptions (same length)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        w = csv.writer(stream, lineterminator="\n")
        w.writerow(HEADER_OUT)
        for r in records:
            w.writerow([r["testcase"], json.dumps(list(r["labels"]), separators=(",", ":")),
                        json.dumps(list(r["descriptions"]), ensure_ascii=False, separators=(",", ":"))])


def read_answer(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != HEADER_OUT:
            raise ValueError(f"{path}: заголовок {reader.fieldnames}, ожидается {HEADER_OUT}")
        return [{"testcase": r["testcase"], "labels": json.loads(r["microcats"]),
                 "descriptions": json.loads(r["descriptions"])} for r in reader]


def check_answer(path, input_path=None, source=101):
    """Returns a list of problems; an empty list means the file can be uploaded."""
    problems = []
    raw = Path(path).read_bytes()
    if b"\r\n" in raw:
        problems.append("переводы строк CRLF, в примере организаторов LF")
    try:
        rows = read_answer(path)
    except (ValueError, json.JSONDecodeError) as exc:
        return [f"файл не читается: {exc}"]
    for r in rows:
        tc, labels, desc = r["testcase"], r["labels"], r["descriptions"]
        if not isinstance(labels, list) or any(type(x) is not int or x not in ALLOWED for x in labels):
            problems.append(f"{tc}: microcats не список ID 101–106: {labels}")
            continue
        if labels != sorted(set(labels)):
            problems.append(f"{tc}: microcats с повторами или не по возрастанию: {labels}")
        if source in labels:
            problems.append(f"{tc}: microcats содержит исходную категорию {source}")
        if not isinstance(desc, list) or len(desc) != len(labels):
            problems.append(f"{tc}: descriptions не совпадает по длине с microcats")
        elif any(not isinstance(d, str) or not d.strip() for d in desc):
            problems.append(f"{tc}: пустой черновик в descriptions")
    if input_path is not None:
        expected = [r["testcase"] for r in read_input(input_path)]
        got = [r["testcase"] for r in rows]
        if set(expected) != set(got):
            problems.append(f"testcase не совпадают с входом: нет {len(set(expected) - set(got))}, "
                            f"лишних {len(set(got) - set(expected))}")
        elif expected != got:
            problems.append("порядок testcase отличается от входного файла")
    return problems


LABEL_COLUMNS = ("labels", "microcats")


def embedded_labels(path, source=101):
    """Rows with labels from a dataset CSV that carries them (labels, microcats or y_101..y_106); None if it has none."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        col = next((c for c in LABEL_COLUMNS if c in fields), None)
        flags = [c for c in sorted(ALLOWED) if f"y_{c}" in fields]
        if col is None and not flags:
            return None
        rows = []
        for r in reader:
            if col is not None:
                labels = json.loads(r[col] or "[]")
            else:
                labels = [c for c in flags if str(r[f"y_{c}"]).strip() in ("1", "1.0", "True", "true")]
            if any(type(x) is not int or x not in ALLOWED for x in labels):
                raise ValueError(f"{r.get('testcase')}: метки не из 101–106: {labels}")
            mc = r.get("mcId")
            src = int(mc) if mc not in (None, "") else source
            rows.append({"testcase": r["testcase"], "description": r["description"] or "",
                         "labels": sorted(set(labels) - {src})})
    return rows


def write_input(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        w = csv.writer(stream, lineterminator="\n")
        w.writerow(HEADER_IN)
        for r in rows:
            w.writerow([r["testcase"], r["description"]])
