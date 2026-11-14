# Инноглобалхак · кейс Авито

## Актуальный датасет для обучения

Объединённая версия **`train_relabel_v2_synth_v1`** содержит 5 311 объявлений: 4 214 реальных из `train_relabel_v2` и 1 097 синтетических из `train_synth_v1`, прошедших независимую проверку. Из реальной части исключены 12 пересечений с тестами и 523 строки по согласованному лимиту в 800 объявлений со всеми пятью метками.

| Назначение | Файл |
| --- | --- |
| Обучение | [`data/processed/train_relabel_v2_synth_v1/train_relabel_v2_synth_v1.parquet`](data/processed/train_relabel_v2_synth_v1/train_relabel_v2_synth_v1.parquet) |
| Сравнение моделей | [`data/test/test_gold.parquet`](data/test/test_gold.parquet) |
| Источники, состав и SHA-256 | [`manifest.json`](data/processed/train_relabel_v2_synth_v1/manifest.json) |
| Исключённые реальные строки | [`excluded_real.csv`](data/processed/train_relabel_v2_synth_v1/excluded_real.csv) |
| Проверка объединённого train | [`07_train_relabel_v2_synth_v1_audit.ipynb`](dataset_review/07_train_relabel_v2_synth_v1_audit.ipynb) |

Для повторной сборки укажите новый каталог `python dataset_review/build_train_relabel_v2_synth_v1.py --out /tmp/train_relabel_v2_synth_v1`. При разбиении в стенде `dev` состоит из реальных строк. Синтетические примеры на основе этих объявлений исключаются из обучения в этом прогоне.

## База с микроправками

В **`train_relabel_v2`** 4 749 реальных объявлений. Поверх `train_relabel_v1` исправили текст в 339 объявлениях и 86 меток в 60 объявлениях.

| Назначение | Файл |
| --- | --- |
| Реальная база | [`data/processed/train_relabel_v2/train.parquet`](data/processed/train_relabel_v2/train.parquet) |
| Источник, число строк и SHA-256 | [`manifest.json`](data/processed/train_relabel_v2/manifest.json) |
| Изменения отдельных меток | [`label_changes.csv`](data/processed/train_relabel_v2/label_changes.csv) |
| Изменения текста | [`text_changes.csv`](data/processed/train_relabel_v2/text_changes.csv) |
| Проверка и сравнение с v1 | [`06_train_relabel_v2_audit.ipynb`](dataset_review/06_train_relabel_v2_audit.ipynb) |

Сборка v2 запускается командой `python dataset_review/build_train_relabel_v2.py`. Для воспроизведения исторической сборки v1 используйте [код из коммита 8b32754](https://github.com/Red1an/Innoglobalhack/tree/8b32754/dataset_review).

В [ноутбуке для экспериментов](research/experiments/colab.ipynb) укажите:

```python
DATASET = "custom"
CUSTOM_TRAIN = ROOT / "data/processed/train_relabel_v2_synth_v1/train_relabel_v2_synth_v1.parquet"
CUSTOM_TEST = ROOT / "data/test/test_gold.parquet"
```

[Инструкция по запуску моделей](research/experiments/README.md)
