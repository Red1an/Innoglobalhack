import sys

from common import LABEL_NAMES, SYNTH, read_jsonl

MODE_TEXT = {
    "delete": "удалить лишние категории, остальное сохранить",
    "short": "удалить лишние категории и сжать до короткого объявления",
    "trap": "удалить лишние категории, а категорию-ловушку оставить в виде, который по правилам не считается",
}


def names(labels):
    return ", ".join(f"{mc} {LABEL_NAMES[mc]}" for mc in labels) or "нет"


def main():
    for name in sys.argv[1:]:
        tasks = read_jsonl(SYNTH / "tasks" / f"{name}.jsonl")
        parts = []
        for task in tasks:
            trap = f"{task['trap_label']} {LABEL_NAMES[task['trap_label']]}: {task['trap_kind']}" if task["trap_label"] else "нет"
            parts.append(
                f"### TASK {task['task_id']}\n"
                f"Режим: {task['mode']} — {MODE_TEXT[task['mode']]}\n"
                f"Оставить (целевые категории): {names(task['target_labels'])}\n"
                f"Убрать: {names([mc for mc in task['removed_labels'] if mc != task['trap_label']])}\n"
                f"Ловушка: {trap}\n"
                f"Исходное объявление:\n<<<\n{task['source_text']}\n>>>\n")
        (SYNTH / "tasks" / f"{name}.md").write_text("\n".join(parts), encoding="utf-8")
        print(name, len(tasks))


if __name__ == "__main__":
    main()
