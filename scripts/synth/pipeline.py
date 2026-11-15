import argparse
import hashlib
import json
import random
import re
from collections import Counter

import pandas as pd

from common import MC, ROOT, SYNTH, Similarity, base_train, entries_of, read_jsonl, test_texts

SEED = 42
ALL_FIVE_CAP = 800
MAX_TEST_SIMILARITY = 0.5
MAX_SYNTH_SIMILARITY = 0.9
MIN_CHARS = 80
CONTACTS = re.compile(r"https?://|www\.|@[a-z0-9_]{3,}|\+7[\s\d()-]{9,}|8[\s(-]*9\d{2}[\s)-]*\d{3}", re.I)
STEMS = {
    102: r"штукатур|оштукатур|маяк",
    103: r"плитк|плиточ|кафел|керамогранит|мозаик",
    104: r"сантех|смесител|унитаз|канализац|водопровод|водоснаб|радиатор|инсталляц|бойлер",
    105: r"электр(?!о(сталь|горск|угли|инструмент|нн))|проводк|розет|выключател|автомат(?!изир)|освещени|светильник|люстр",
    106: r"покрас|окраш|обо[ия]\b|обоев|маляр|стеклохолст",
}
OUT = ROOT / "data" / "processed" / "train_synth_v1"


def generations(name):
    texts = {}
    for path in sorted(SYNTH.glob(f"gen/{name}*.txt")):
        blocks = re.split(r"^### TASK (\S+)\s*$", path.read_text(encoding="utf-8"), flags=re.M)
        for task_id, text in zip(blocks[1::2], blocks[2::2]):
            texts[task_id] = text.strip().strip("`").strip()
    return texts


def opaque(task_id):
    return "t" + hashlib.sha256(task_id.encode()).hexdigest()[:10]


def prepare_check(name):
    texts = generations(name)
    tasks = read_jsonl(SYNTH / "tasks" / f"{name}.jsonl")
    missing = [task["task_id"] for task in tasks if task["task_id"] not in texts]
    order = [task["task_id"] for task in tasks if task["task_id"] in texts]
    random.Random(name).shuffle(order)
    body = "\n".join(f"### TEXT {opaque(task_id)}\n{texts[task_id]}\n" for task_id in order)
    (SYNTH / "check" / f"{name}_input.md").write_text(body, encoding="utf-8")
    print(name, "текстов", len(order), "нет ответа", missing)


def checker_labels(name):
    result = {}
    for path in sorted(SYNTH.glob(f"check/{name}*.tsv")):
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) < 2 or not parts[0].startswith("t"):
                continue
            labels = sorted({int(x) for x in re.findall(r"\d{3}", parts[1]) if int(x) in MC})
            result[parts[0].strip()] = {"labels": labels, "note": parts[2].strip() if len(parts) > 2 else ""}
    return result


def evaluate(names, require_check=True):
    rows = []
    for name in names:
        texts, checks = generations(name), checker_labels(name)
        for task in read_jsonl(SYNTH / "tasks" / f"{name}.jsonl"):
            text = texts.get(task["task_id"])
            check = checks.get(opaque(task["task_id"]))
            reasons = []
            if not text:
                if not require_check:
                    continue
                reasons.append("нет текста")
            elif len(text) < MIN_CHARS:
                reasons.append("короткий")
            if text and CONTACTS.search(text):
                reasons.append("контакты")
            if check is None and require_check:
                reasons.append("нет проверки")
            elif check is not None and check["labels"] != task["target_labels"]:
                reasons.append("метки не совпали")
            if check is not None and check["note"].startswith("НЕОДНОЗНАЧНО"):
                reasons.append("неоднозначная услуга")
            if text:
                for mc in task["removed_labels"]:
                    if mc != task["trap_label"] and re.search(STEMS[mc], text.lower()):
                        reasons.append(f"след {mc}")
            rows.append({**task, "batch": name, "text": text or "", "check_labels": check["labels"] if check else None,
                         "check_note": check["note"] if check else "", "checked": check is not None, "reasons": reasons})
    frame = pd.DataFrame(rows)
    ok = frame[frame.reasons.map(len) == 0].copy()
    if len(ok):
        ok["test_similarity"] = Similarity(test_texts()).max_to_reference(ok.text.tolist())
        frame.loc[ok.index[ok.test_similarity >= MAX_TEST_SIMILARITY], "reasons"] = [["похож на тест"]] * int((ok.test_similarity >= MAX_TEST_SIMILARITY).sum())
        ok = ok[ok.test_similarity < MAX_TEST_SIMILARITY]
        similarity = Similarity(ok.text.tolist())
        matrix = similarity.vectorizer.transform(ok.text.str.lower())
        pair = (matrix @ matrix.T).toarray()
        drop = set()
        for i in range(len(ok)):
            if i in drop:
                continue
            for j in range(i + 1, len(ok)):
                if pair[i, j] >= MAX_SYNTH_SIMILARITY and ok.target_labels.iloc[i] == ok.target_labels.iloc[j]:
                    drop.add(j)
        for j in drop:
            frame.at[ok.index[j], "reasons"] = ["дубль синтеза"]
    frame["accepted"] = frame.reasons.map(len) == 0
    return frame


def report(names, require_check=True):
    frame = evaluate(names, require_check)
    print("всего", len(frame), "принято", int(frame.accepted.sum()))
    print("причины", Counter(reason for reasons in frame.reasons for reason in reasons))
    print(frame.groupby("group").accepted.agg(["sum", "count"]).to_string())
    print(frame.groupby("mode").accepted.agg(["sum", "count"]).to_string())
    bad = frame[frame.reasons.map(lambda r: "метки не совпали" in r)]
    for row in bad.itertuples():
        print(f"- {row.task_id} цель {row.target_labels} проверка {row.check_labels} ловушка {row.trap_label} {row.check_note}")
    return frame


def assemble(names, require_check=True):
    frame = evaluate(names, require_check)
    accepted = frame[frame.accepted]
    train = base_train()
    rng = random.Random(SEED)
    all_five = [i for i, labels in zip(train.index, train.labels) if len(labels) == 5]
    dropped = set(rng.sample(all_five, len(all_five) - ALL_FIVE_CAP))
    real = train.drop(index=sorted(dropped)).copy()
    real["origin"] = "real"
    real["synth_task_id"] = None
    real["synth_checked"] = None
    real["found_microcategories"] = [entries_of(labels) for labels in real.labels]
    synth = pd.DataFrame({
        "item_id": [-(i + 1) for i in range(len(accepted))],
        "title": None,
        "description": accepted.text.values,
        "found_microcategories": [entries_of(labels) for labels in accepted.target_labels],
        "source_line": [-int(item) for item in accepted.source_item_id],
        "origin": "synthetic",
        "synth_task_id": accepted.task_id.values,
        "synth_checked": accepted.checked.values,
    })
    result = pd.concat([real.drop(columns=["labels"]), synth], ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUT / "train.parquet", index=False)
    frame.drop(columns=["source_text"]).to_json(OUT / "synth_review.jsonl", orient="records", lines=True, force_ascii=False)
    train.loc[sorted(dropped), ["item_id", "source_line"]].to_csv(OUT / "dropped_all_five.csv", index=False)
    manifest = {
        "version": "train_synth_v1",
        "base": "102 из train_labels_102_v1, 103–106 из train_relabel_v1",
        "records": len(result), "real": len(real), "synthetic": len(synth),
        "all_five_before": len(all_five), "all_five_after": ALL_FIVE_CAP, "seed": SEED,
        "synthesis": {"tasks": len(frame), "accepted": len(accepted),
                      "rejected": dict(Counter(reason for reasons in frame.reasons for reason in reasons))},
        "columns": {"origin": "real или synthetic", "synth_task_id": "задача синтеза",
                    "source_line": "для синтетики минус item_id исходного объявления"},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare-check", "report", "assemble"])
    parser.add_argument("names", nargs="+")
    parser.add_argument("--allow-unchecked", action="store_true", help="Принимать тексты без слепой проверки меток")
    args = parser.parse_args()
    if args.command == "prepare-check":
        for name in args.names:
            prepare_check(name)
    elif args.command == "report":
        report(args.names, not args.allow_unchecked)
    else:
        assemble(args.names, not args.allow_unchecked)


if __name__ == "__main__":
    main()
