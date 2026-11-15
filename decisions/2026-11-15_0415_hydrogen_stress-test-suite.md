# Лог решений: сборка диагностического стресс-бенчмарка `stress_suite_v1`

## 2026-11-15 04:15 МСК | решение | метрика | глобальное
Что: Начинаем разработку воспроизводимого диагностического стресс-бенчмарка `data/test/stress_suite_v1/` из 7 срезов (440 строк суммарно), скрипта сборки `scripts/build_stress_suite.py` и единого оценщика `scripts/eval_stress_suite.py`.
Почему: Перед финальной отправкой приватного теста нужно проверять модели не только на общем `test_gold` (80 строк), но и на целевых точках отказа: сложных пустых объявлениях, отвязанных парах `104≠105` и `102≠106`, контекстных ловушках и отрицаниях, омоглифах/грязном тексте, входных `mcId != 101` и пограничных правилах ментора (SEO-теги, бригада, наружные работы).
Где: decisions/2026-11-15_0415_hydrogen_stress-test-suite.md; scripts/build_stress_suite.py; scripts/eval_stress_suite.py; data/test/stress_suite_v1/.

## 2026-11-15 04:22 МСК | эксперимент | метрика | глобальное
Что: Собран единый воспроизводимый стресс-бенчмарк `data/test/stress_suite_v1/` (440 строк в 7 срезах: Parquet + JSONL + `manifest.json`) и оценён базовый `char-LR` (5 классов `102–106`, честный порог `p >= 0.50`, обучение на 4 351 строке `train_fit` без пересечений с `test_gold`, `dev` и новыми строками).
Почему: Нужно локализовать конкретные режимы отказа моделей до финальной сдачи приватного теста.
Результат:
- `gold_core` (80): Micro-F1 **90.48%**, Exact Match **73.75%**, FP на пустых 3/20 (15.0%).
- `hard_empty` (40): Exact Match **47.50%**, FP на пустых **21/40 (52.5%, 42 ложных метки)** — `char-LR` массово ошибается на технадзоре (5/6 FP), проектировании (5/6 FP) и демонтаже (5/6 FP).
- `decoupled_solo_and_pairs` (100): Micro-F1 **71.60%**, EM **47.00%**, склейка пар снизилась до 2.3% (104→105: 1/37, 105→104: 1/53, 102→106: 1/43), но падает Recall (58.00%) на одиночных и нетипичных парах.
- `context_traps_and_negations` (80): Micro-F1 **70.63%**, Precision **60.13%**, EM **32.50%**, склейка/ложные срабатывания **30.9%** — на явных отрицаниях EM **0.00%** (0/16), на чужой работе EM **12.50%** (2/16).
- `dirty_text_and_homoglyphs` (50): Micro-F1 **86.27%**, EM **66.00%** — слипшиеся цифры/КАПС (96.43% F1) и антифрод-хвосты (93.88% F1) держатся хорошо, но латинские омоглифы роняют F1 до **66.67%** (EM **40.00%**).
- `non_101_source_mcid` (50): Micro-F1 **98.55%**, EM **96.00%** при фильтрации `label != mcId`.
- `mentor_borderline_switches` (40): в режиме `strict` Micro-F1 **57.14%** (EM **35.00%**), в режиме `with_seo_tags` — **80.00%** (EM **60.00%**, Precision **100.00%**), т.к. `char-LR` по умолчанию учитывает SEO-теги в конце объявления.
- Итого по всему `stress_suite_v1` (440, strict): Micro-F1 **77.79%**, EM **55.91%**; по новым 360 стресс-строкам (без `gold_core`): Micro-F1 **73.20%**, EM **51.94%**.
Где: scripts/build_stress_suite.py; scripts/eval_stress_suite.py; data/test/stress_suite_v1/; output/stress_suite_v1_report.md.

## 2026-11-15 10:42 МСК | решение | метрика | глобальное
Что: Обновляем `stress_suite_v1` под актуальный официальный гайд `innoglobalhack_avito/microcategories-guide.md`.
Почему: В `microcategories-guide.md` расширено определение категории `106` (включены шпаклёвка под покраску/обои, поклейка стеклохолста, грунтование и шлифовка в составе подготовки к финишной отделке, а также снятие старых обоев при обновлении отделки), а в `105` явно указано штробление стен под проводку.
Где: innoglobalhack_avito/microcategories-guide.md; scripts/stress_suite_data.py; scripts/stress_suite_data_traps.py; data/test/stress_suite_v1/.

## 2026-11-15 10:45 МСК | эксперимент | метрика | глобальное
Что: Пересобран `stress_suite_v1` по правилам `innoglobalhack_avito/microcategories-guide.md` (440 строк, 0 пересечений с `train_relabel_v2_synth_v1`) и повторно оценён базовый `char-LR` (`p >= 0.50`).
Почему: В `TRAPS_PREP_ONLY` (16 строк) шпаклёвка под покраску/обои, стеклохолст, грунтование и шлифовка под финиш теперь размечены с `106` (кроме случая чистой штукатурки с глянцеванием без малярной подготовки — там только `102`), а из `hard_empty` убраны неоднозначные упоминания штробления под проводку (`105`) и финишной шпаклёвки плоскостей (`106`).
Результат: На обновлённом `stress_suite_v1` у `char-LR`: `context_traps_and_negations` Micro-F1 **71.58%** (Precision 64.15%, EM 33.75%; подсрез `prep_only` вырос с 80.00% до **81.82%** F1 и 43.75% EM), `hard_empty` FP **22/40 (55.0%, 43 lbl)**, итого по всему suite (strict) Micro-F1 **77.85%**, EM **55.91%** (`output/stress_suite_v1_report.md`).
Где: scripts/stress_suite_data.py; scripts/stress_suite_data_traps.py; data/test/stress_suite_v1/; output/stress_suite_v1_report.md.

## 2026-11-15 11:04 МСК | решение | данные | глобальное
Что: Проведён полный аудит всех 360 стресс-строк (вне `gold_core`) на соответствие `innoglobalhack_avito/microcategories-guide.md`, удалены скрытые конфликты из `hard_empty` и `dev`-подборки `decoupled_solo_and_pairs`, бенчмарк пересобран и повторно оценён на `char-LR`.
Почему: В `hard_empty` оставались `row_id=6, 68` из `test_gold` со шпаклёвкой (заменены на `row_id=2, 25`), а в `dev`-выборке `decoupled_solo_and_pairs` были 4 строки с неучтённой по новому гайду шпаклёвкой или смежными работами (отфильтрованы через `DEV_BAD_RE`, добор выполнен чистыми `SOLO_105_NEW` и `SOLO_102_NEW`). `gold_core` сохранён точной копией `data/test/test_gold.parquet` как фиксированный ориентир.
Результат: Итог `char-LR` (`p >= 0.50`, `train_fit=4351`) на обновлённом `stress_suite_v1`: `gold_core` Micro-F1 **90.48%**, `hard_empty` EM **42.50%** (FP 23/40), `decoupled_solo_and_pairs` Micro-F1 **70.54%** (EM **46.00%**), `context_traps_and_negations` Micro-F1 **71.58%** (EM **33.75%**), `dirty_text_and_homoglyphs` Micro-F1 **86.27%**, `non_101_source_mcid` Micro-F1 **98.55%**, `mentor_borderline` (strict/seo) **57.14% / 80.00%**, общий итог (440 строк, strict) Micro-F1 **77.60%**, EM **55.45%**.
Где: scripts/build_stress_suite.py; scripts/stress_suite_data_decoupled.py; data/test/stress_suite_v1/; output/stress_suite_v1_report.md.

## 2026-11-15 11:19 МСК | эксперимент | метрика | глобальное
Что: Переключены дефолтные источники `scripts/build_stress_suite.py` и `scripts/eval_stress_suite.py` на актуальный датасет `data/processed/train_guide_v1/train.parquet` и эталон `data/test/test_guide_v1/test_gold.parquet`, пересобран `data/test/stress_suite_v1/` (включая обновление 13 меток в `gold_core`) и выполнена оценка `char-LR` (`p >= 0.50`, `train_fit=4351`).
Почему: В проекте был собран новый обучающий и тестовый профиль `guide_v1` с 91 правкой в train и 13 правками в gold; оценка стресс-бенчмарка должна выполняться на нём.
Результат:
- `gold_core` (80): Micro-F1 **89.76%**, EM **70.00%** (полностью совпадает с контролем `guide` в `decisions/2026-11-15_0505_codex_gold-guide-verification.md`).
- `context_traps_and_negations` (80): Micro-F1 вырос с 71.58% до **74.66%** (EM **37.50%**), а подсрез `prep_only` вырос с 81.82% до **93.15%** F1 (EM **68.75%**) за счёт согласования `106` в `train_guide_v1`.
- `hard_empty` (40): EM **40.00%**, FP **24/40 (60.0%)**.
- `decoupled_solo_and_pairs` (100): Micro-F1 **70.54%**, EM **46.00%**.
- `dirty_text_and_homoglyphs` (50): Micro-F1 **85.53%**, EM **64.00%**.
- `non_101_source_mcid` (50): Micro-F1 **98.55%**, EM **96.00%**.
- `mentor_borderline` (40, strict / with_seo_tags): Micro-F1 **57.83% / 78.99%**, EM **35.00% / 57.50%**.
- Итого по всему `stress_suite_v1` (440, strict): Micro-F1 **77.88%**, EM **55.00%**.
Где: scripts/build_stress_suite.py; scripts/eval_stress_suite.py; data/test/stress_suite_v1/; output/stress_suite_v1_report.md.
