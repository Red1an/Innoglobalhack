import random
import re
from collections import Counter, defaultdict

import pandas as pd

from common import MC, ROOT, SYNTH, base_train, labels_of, read_jsonl
from compare import PROBE_SHARE, SEED
from probe_charlr import char_lr

KINDS = [
    ("отрицание", r"прямой отказ"),
    ("чужая работа", r"партн|заказчик|другие подрядчики|другой мастер|свой сантехник"),
    ("только подготовка", r"шпаклёвка и подготовка"),
    ("выравнивание без штукатурки", r"выравнивание стен без"),
    ("только проект", r"только проект"),
    ("только демонтаж", r"только демонтаж"),
    ("пример объекта или технадзор", r"технадзор|примере объекта|выполненного объекта"),
    ("тёплый пол без «электрический»", r"тёплый пол"),
]


def kind_of(text):
    for name, pattern in KINDS:
        if text and re.search(pattern, text):
            return name
    return "другое"


def main():
    tasks = {}
    for path in SYNTH.glob("tasks/*.jsonl"):
        for task in read_jsonl(path):
            tasks[task["task_id"]] = task
    train = pd.read_parquet(ROOT / "data/processed/train_synth_v1/train.parquet")
    synth = train[train.origin == "synthetic"].copy()
    synth["labels"] = synth.found_microcategories.map(labels_of)
    sources = sorted(set(-synth.source_line))
    random.Random(SEED).shuffle(sources)
    probe_sources = set(sources[:int(len(sources) * PROBE_SHARE)])
    probe = synth[(-synth.source_line).isin(probe_sources)]
    fit = train[~train.item_id.isin(probe_sources) & ~(-train.source_line).isin(probe_sources)].copy()
    fit["labels"] = fit.found_microcategories.map(labels_of)
    base = base_train()
    base = base[~base.item_id.isin(probe_sources)]
    for name, frame in [("база", base), ("сокращение + синтез", fit)]:
        model = char_lr(frame.description.tolist(), frame.labels.tolist())
        pred = model.predict(probe.description.tolist())
        trap_total, trap_fired, other_fp, fn = Counter(), Counter(), Counter(), Counter()
        mode_err = defaultdict(lambda: [0, 0])
        for row, p in zip(probe.itertuples(), pred):
            task = tasks[row.synth_task_id]
            predicted = {mc for mc, yes in zip(MC, p) if yes}
            target = set(row.labels)
            mode_err[task["mode"]][0] += predicted != target
            mode_err[task["mode"]][1] += 1
            trap = task["trap_label"]
            if trap:
                kind = kind_of(task["trap_kind"])
                trap_total[kind] += 1
                trap_fired[kind] += trap in predicted
            for mc in predicted - target - ({trap} if trap else set()):
                other_fp["удалённая соседняя" if mc in task["removed_labels"] else "не было в исходнике"] += 1
            for mc in target - predicted:
                fn[mc] += 1
        print(f"\n== {name}")
        print("ошибка по режиму:", {k: f"{v[0]}/{v[1]}" for k, v in mode_err.items()})
        print("сработал на ловушку:")
        for kind in sorted(trap_total, key=lambda k: -trap_total[k]):
            print(f"   {kind}: {trap_fired[kind]} из {trap_total[kind]}")
        print("прочие лишние метки:", dict(other_fp))
        print("пропуски целевых:", dict(sorted(fn.items())))


if __name__ == "__main__":
    main()
