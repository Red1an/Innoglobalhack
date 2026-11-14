# Инноглобалхак · кейс Авито

## Актуальный датасет для обучения

Для обучения классификатора используем **`train_relabel_v1`**. Файл находится в [`data/processed/train_relabel_v1/train.parquet`](data/processed/train_relabel_v1/train.parquet).

В нём **4 749 объявлений** после очистки текста и дедупликации. По сравнению с `train_dedup_v1` обновили разметку категорий `102–106` в **734 объявлениях**. Версия `train_labels_102_v1` использовалась для отдельного эксперимента с категорией `102`.

| Назначение | Файл |
| --- | --- |
| Обучение | [`data/processed/train_relabel_v1/train.parquet`](data/processed/train_relabel_v1/train.parquet) |
| Сравнение моделей | [`data/test/test_gold.parquet`](data/test/test_gold.parquet) |
| Источник, число строк и SHA-256 | [`manifest.json`](data/processed/train_relabel_v1/manifest.json) |
| Изменения отдельных меток | [`label_changes.csv`](data/processed/train_relabel_v1/label_changes.csv) |

В [ноутбуке для экспериментов](research/experiments/colab.ipynb) укажите:

```python
DATASET = "custom"
CUSTOM_TRAIN = ROOT / "data/processed/train_relabel_v1/train.parquet"
CUSTOM_TEST = ROOT / "data/test/test_gold.parquet"
```

[Инструкция по запуску моделей](research/experiments/README.md)
