## 2026-11-15 05:09 МСК | решение | процесс | глобальное
Что: По запросу пользователя сохраняем все текущие изменения guide_v1, профилей запуска, аудита и документации, затем отправляем dev в origin.
Почему: Актуальная версия разметки по справочнику и её стандартный запуск должны быть доступны команде.
Где: data/processed/train_guide_v1/, data/test/test_guide_v1/, dataset_review/, research/experiments/, scripts/eval_stress_suite.py, README.md, decisions/.

## 2026-11-15 05:10 МСК | эксперимент | процесс | локальное
Что: Проверены синтаксис изменённых Python-файлов, JSON двух ноутбуков и staged diff с учётом CRLF. Оценщик успешно запущен с --suite, --train и --gold для guide_v1 на 440 строках; char-LR обучен на 4 351 строке.
Результат: Проверки синтаксиса, JSON, diff и запуск оценщика прошли. Unittest discover: 21 проверка, 19 успешных, две ошибки из-за отсутствующих pydantic (импорт test_api_bench) и torch (test_head_trains_and_predicts); полный набор тестов не подтверждён.
Где: /tmp/innoglobalhack-guide-tests.txt; /tmp/innoglobalhack-guide-stress-verification.md.
