import random
from collections import Counter

from common import MC, SYNTH, Similarity, base_train, test_texts, write_jsonl

SEED = 42
MAX_SOURCE_CHARS = 3000
MAX_TEST_SIMILARITY = 0.5
MAX_SOURCE_USES = 2
PILOT_SIZE = 24
BATCH_SIZE = 60

# target labels, how many, labels the source must contain, forced trap label
GROUPS = [
    ("only_104", [104], 250, [105], None),
    ("only_105", [105], 250, [104], None),
    ("103_104", [103, 104], 120, [105], None),
    ("only_102", [102], 150, [106], None),
    ("102_103", [102, 103], 60, [106], None),
    ("only_103", [103], 50, [104], None),
    ("102_104", [102, 104], 40, [105], None),
    ("105_106", [105, 106], 40, [104], None),
    ("trap_102", None, 100, [102], 102),
    ("trap_106", None, 80, [106], 106),
]

TRAPS = {
    102: ["выравнивание стен без слов штукатурка, оштукатуривание и маяки (например гипсокартоном, шпаклёвкой или просто «выравниваем стены и потолки»)",
          "штукатурку делают другие подрядчики или заказчик",
          "прямой отказ: «штукатурные работы не выполняем»"],
    103: ["прямой отказ: «плитку не кладём»", "плитку кладут партнёры или другой мастер",
          "плитка упомянута только в описании выполненного объекта или в фотоотчёте"],
    104: ["прямой отказ: «сантехнику не делаем»", "сантехнику выполняют партнёры, у заказчика свой сантехник",
          "только демонтаж старой сантехники без установки и подключения",
          "сантехника упомянута только в перечне того, что проверяет технадзор, или в примере объекта"],
    105: ["прямой отказ: «электрикой не занимаемся»", "электромонтаж выполняют партнёры-электрики",
          "только проект или расстановка розеток в дизайн-проекте, без монтажа",
          "тёплый пол без указания, что он электрический",
          "электрика упомянута только в перечне того, что проверяет технадзор, или в примере объекта"],
    106: ["только шпаклёвка и подготовка стен под покраску или обои, а самой покраски и поклейки нет",
          "покраску и обои заказчик делает сам", "прямой отказ: «обои не клеим, малярку не делаем»"],
}

MODES = [("delete", 0.5), ("trap", 0.35), ("short", 0.15)]


def pick_mode(rng):
    value, total = rng.random(), 0.0
    for mode, share in MODES:
        total += share
        if value < total:
            return mode
    return MODES[-1][0]


def main():
    rng = random.Random(SEED)
    train = base_train()
    train = train[train.description.str.len().between(150, MAX_SOURCE_CHARS)].copy()
    similarity = Similarity(test_texts())
    train["test_similarity"] = similarity.max_to_reference(train.description.tolist())
    train = train[train.test_similarity < MAX_TEST_SIMILARITY]
    sources = list(train.itertuples())
    uses = Counter()
    tasks = []
    for group, target, count, required, forced_trap in GROUPS:
        made = 0
        while made < count:
            row = rng.choice(sources)
            labels = set(row.labels)
            if uses[row.item_id] >= MAX_SOURCE_USES or not set(required) <= labels:
                continue
            if forced_trap is None:
                if not set(target) <= labels:
                    continue
                goal = list(target)
                mode = pick_mode(rng)
            else:
                rest = sorted(labels - {forced_trap})
                if not rest:
                    continue
                goal = sorted(rng.sample(rest, rng.randint(1, min(3, len(rest)))))
                mode = "trap"
            removed = sorted(labels - set(goal))
            trap_label = forced_trap if forced_trap else (rng.choice(removed) if mode == "trap" else None)
            uses[row.item_id] += 1
            made += 1
            tasks.append({
                "task_id": f"{group}-{made:03d}", "group": group, "mode": mode,
                "source_item_id": int(row.item_id), "source_labels": sorted(labels),
                "target_labels": goal, "removed_labels": removed,
                "trap_label": trap_label, "trap_kind": rng.choice(TRAPS[trap_label]) if trap_label else None,
                "source_text": row.description,
            })
    rng.shuffle(tasks)
    pilot, rest = tasks[:PILOT_SIZE], tasks[PILOT_SIZE:]
    write_jsonl(SYNTH / "tasks" / "pilot.jsonl", pilot)
    for index in range(0, len(rest), BATCH_SIZE):
        write_jsonl(SYNTH / "tasks" / f"batch_{index // BATCH_SIZE + 1:02d}.jsonl", rest[index:index + BATCH_SIZE])
    print("задач", len(tasks), "пилот", len(pilot), "пакетов", -(-len(rest) // BATCH_SIZE))
    print("режимы", Counter(task["mode"] for task in tasks))
    print("исходников", len(uses), "из", len(sources))


if __name__ == "__main__":
    main()
