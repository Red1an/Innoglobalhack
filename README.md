# Инноглобалхак · кейс Авито

## Сквозное демо

Для демо нужен Python 3.10+. В ветку включены готовые ответы моделей и размеченный набор.

```bash
git clone --branch codex/avito-final https://github.com/Red1an/Innoglobalhack.git
cd Innoglobalhack
python3 avito_pipeline.py --replay --port 8767
```

Откройте [запись сквозного прогона](http://127.0.0.1:8767/demo) или [длинное объявление с готовыми черновиками](http://127.0.0.1:8767/review?case=case-113). Режим `--replay` работает без ключа и сохраняет правки в браузере.

Для живого запуска задайте свой ключ Caila в локальном файле `.env` рядом с `avito_pipeline.py`.

```dotenv
DRAFT_API_KEY=ваш_ключ_Caila
DRAFT_TEMPERATURE=0
```

```bash
python3 avito_pipeline.py --serve --port 8767
```

В [сквозном демо](http://127.0.0.1:8767/demo) кнопка «Запустить модели вживую» проверяет три объявления и CSV из двух примеров организаторов. В [редакторе](http://127.0.0.1:8767/) можно ввести свой текст, исправить черновики, скачать JSON или обработать CSV.

В [проверке черновиков](http://127.0.0.1:8767/review) доступны 262 объявления из `data/private-test-dataset.labeled.csv`. Среди подготовленных сценариев есть [длинное объявление бригады](http://127.0.0.1:8767/review?case=case-113) и [стресс-тест на 72 тыс. знаков](http://127.0.0.1:8767/review?case=case-018). Целевые категории берутся из разметки, поэтому генерация не требует вызова классификатора. Сохранённые результаты видны при открытии, правки хранятся в браузере. Нажатие на выбранный моделью фрагмент подсвечивает его в исходнике. Для сравнения с прежними текстами положите `private-test-dataset.predict.csv` в `private_eval/input/` перед запуском сервера.

Qwen3‑235B с промптом v6 и препроцессингом определяет дополнительные категории. GLM 5.3 Flash одним запросом готовит тексты для всех найденных категорий, используя исходное объявление. На выходе проверяются схема и точность цитат. Режим `--classifier glm` подключает GLM 5.3 Flash с промптом v9 для классификации.

Для интеграции вызовите `POST /drafts` на сервере `--serve` с JSON `{"mcId": 101, "description": "Штукатурим стены и меняем проводку."}`. Ответ содержит `drafts` со списком целевых `mcId`, заголовками и текстами, а также `metrics`. Если существующий классификатор уже возвращает метки, передайте его список целых `mcId` вместе с исходным текстом в `avito_pipeline.generate_targets(description, labels, generator, audit_dir)`. Используйте настроенный `draft_generator.core.HTTPModel` и новый каталог аудита для каждого запуска. Системный промпт, правила `rewrite` и сборка запроса находятся в [draft_generator/core.py](draft_generator/core.py).

```bash
python3 avito_pipeline.py --csv private_eval/examples/example.csv --out /tmp/avito-example
python3 avito_pipeline.py --export-reviewed reviewed.jsonl --input input.csv --out /tmp/avito-reviewed
```

Входной CSV содержит `testcase,description` и может задавать исходную категорию `mcId` для каждой строки. Если этой колонки нет, применяется `--default-mcid` (101 по умолчанию). Пакетный запуск сохраняет черновики и метрики в `results.jsonl` и ответы генератора в `audit/`. Если есть пустые тексты или нарушения формата, причины попадают в `validation.txt`. После правок в копии `results.jsonl` команда `--export-reviewed` проверяет результат снова без запросов к моделям. Файл `submission.csv` появляется после успешной проверки.

## Генерация черновиков на защите

- [Командная презентация с блоком генерации](presentation.html) · откройте скачанный HTML в браузере, слайды 9–13.
- [Отдельная версия блока в PowerPoint](docs/defense/avito_text_generation.pptx)
- [Рассказ на 3:20, версия на 75 секунд и действия для демо](docs/defense/avito_text_generation_talk.md)
- [Замеры генерации по облачным и локальным моделям](docs/defense/generation_benchmark.json)

## Guide v3 для хакатона

[Уточнённые правила](dataset_review/guide_v3_rules.md) учитывают услуги в SEO, явно предложенную шпаклёвку и ограничение «только комплекс». Версия собрана поверх `guide_v2`. Проверка train сочетает отбор кандидатов по всем 5 311 строкам, проверку повторяющихся шаблонов и разбор отдельных объявлений. Спорные решения видны в `ambiguous.csv`.

| Назначение | Файл |
| --- | --- |
| Обучение | [train.parquet](data/processed/train_guide_v3/train.parquet), [манифест](data/processed/train_guide_v3/manifest.json) |
| Внутренние тесты | [gold](data/test/test_guide_v3/test_gold.parquet), [hard](data/test/test_guide_v3/hard.jsonl), [стресс-срезы](data/test/test_guide_v3/stress_suite.jsonl), [манифест](data/test/test_guide_v3/manifest.json) |
| Изменения и спорные метки train | [label_changes.csv](data/processed/train_guide_v3/label_changes.csv), [ambiguous.csv](data/processed/train_guide_v3/ambiguous.csv) |
| Изменения и спорные метки тестов | [label_changes.csv](data/test/test_guide_v3/label_changes.csv), [ambiguous.csv](data/test/test_guide_v3/ambiguous.csv) |
| Сравнение guide_v2 и guide_v3 на одном новом тесте | [ноутбук](dataset_review/10_guide_v3_audit.ipynb), [метрики](output/guide_v3_comparison.json) |

```bash
python dataset_review/build_guide_v3.py train --out /tmp/train_guide_v3
python dataset_review/build_guide_v3.py test --out /tmp/test_guide_v3
python dataset_review/compare_guide_v3.py
```

Для экспериментов в `colab.ipynb` укажите эти train и gold в профиле `custom`. Скрипт сравнения использует одинаковые строки обучения и порог 0,50. Результаты относятся к внутренним gold, hard и стресс-срезам.

## Guide v2

Правила [guide_v2](dataset_review/guide_v2_rules.md) применяются к новой версии train и внутренним тестам. Формат объявления на входе остаётся прежним. Исходная категория исключается из меток, оценка учитывает 101–106.

| Назначение | Файл |
| --- | --- |
| Обучение, 5 311 строк | [train.parquet](data/processed/train_guide_v2/train.parquet) |
| Основной тест, 80 строк | [test_gold.parquet](data/test/test_guide_v2/test_gold.parquet) |
| Hard, 50 строк, и семь стресс-срезов, 440 строк | [hard.jsonl](data/test/test_guide_v2/hard.jsonl), [stress_suite.jsonl](data/test/test_guide_v2/stress_suite.jsonl) |
| Правки и спорные примеры train | [label_changes.csv](data/processed/train_guide_v2/label_changes.csv), [ambiguous.csv](data/processed/train_guide_v2/ambiguous.csv) |
| Правки и спорные примеры тестов | [label_changes.csv](data/test/test_guide_v2/label_changes.csv), [ambiguous.csv](data/test/test_guide_v2/ambiguous.csv) |
| Сравнение обеих версий train | [09_guide_v2_audit.ipynb](dataset_review/09_guide_v2_audit.ipynb), [guide_v2_comparison.json](output/guide_v2_comparison.json) |

Из корня репозитория можно пересобрать версии в новых каталогах и сравнить опубликованные версии Char-LR.

```bash
python dataset_review/build_guide_v2.py test --out /tmp/test_guide_v2
python dataset_review/build_guide_v2.py train --out /tmp/train_guide_v2
python dataset_review/compare_guide_v2.py
```

Сравнение использует одинаковые 4 351 строки fit и порог 0,50. По описаниям после нормализации пересечений fit с gold, hard, stress и исходным public test нет. Из общего результата исключены 95 повторов gold в стресс-срезах. Во всех строках train исходная категория 101, поэтому Char‑LR не обучается предсказывать 101. Ноутбук использует сохранённые результаты по умолчанию. Флаг `RERUN = True` пересчитывает их.

## Guide v1

`train_guide_v1` содержит 5 311 объявлений из предыдущего объединённого train. Исправлена 91 метка, в том числе 86 пропусков `106`. В основном тесте исправлено 13 меток. Профиль `guide` в терминале и Colab по умолчанию использует эту пару train и gold.

| Назначение | Файл |
| --- | --- |
| Обучение | [`train.parquet`](data/processed/train_guide_v1/train.parquet) |
| Основной тест (80 объявлений) | [`test_gold.parquet`](data/test/test_guide_v1/test_gold.parquet) |
| Диагностика граничных случаев | [`hard.jsonl`](data/test/test_guide_v1/hard.jsonl), [`stress_suite.jsonl`](data/test/test_guide_v1/stress_suite.jsonl) |
| Сверка train с исходной разметкой | [`manifest.json`](data/processed/train_guide_v1/manifest.json), [`label_changes.csv`](data/processed/train_guide_v1/label_changes.csv) |
| Сверка тестов | [`manifest.json`](data/test/test_guide_v1/manifest.json), [`label_changes.csv`](data/test/test_guide_v1/label_changes.csv) |
| Проверка и сравнение моделей | [`08_guide_v1_audit.ipynb`](dataset_review/08_guide_v1_audit.ipynb) |

Для повторной сборки укажите два новых каталога.

```bash
git show d926a883e0668b063aa221f4d7e079b9ae194694:data/test/stress_suite_v1/stress_suite_v1.parquet > /tmp/stress_source_for_guide_v1.parquet
python dataset_review/build_guide_v1.py \
  --stress-source /tmp/stress_source_for_guide_v1.parquet \
  --train-out /tmp/train_guide_v1 --test-out /tmp/test_guide_v1
```

Для прогона диагностики с обновлённой разметкой используйте:

```bash
python scripts/eval_stress_suite.py \
  --suite data/test/test_guide_v1/stress_suite.jsonl \
  --train data/processed/train_guide_v1/train.parquet \
  --gold data/test/test_guide_v1/test_gold.parquet \
  --report /tmp/stress_guide_v1_report.md
```

## Предыдущий объединённый train

Объединённая версия **`train_relabel_v2_synth_v1`** содержит 5 311 объявлений: 4 214 реальных из `train_relabel_v2` и 1 097 синтетических из `train_synth_v1`, прошедших независимую проверку. Из реальной части исключены 12 пересечений с тестами и 523 строки по согласованному лимиту в 800 объявлений со всеми пятью метками.

| Назначение | Файл |
| --- | --- |
| Обучение | [`data/processed/train_relabel_v2_synth_v1/train_relabel_v2_synth_v1.parquet`](data/processed/train_relabel_v2_synth_v1/train_relabel_v2_synth_v1.parquet) |
| Gold прежних прогонов | [`test_v7.parquet`](data/test/test_v7.parquet), совпадает с `data/test/test_gold.parquet` |
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

В [ноутбуке для экспериментов](research/experiments/colab.ipynb) оставьте `DATASET = "guide"`. Эквивалентный запуск Char-LR из корня репозитория:

```bash
python research/experiments/run_all.py --dataset guide --models char-lr --no-upload
```

[Инструкция по запуску моделей](research/experiments/README.md)
