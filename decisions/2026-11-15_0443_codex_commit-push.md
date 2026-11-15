## 2026-11-15 04:43 МСК | решение | процесс | глобальное
Что: По запросу пользователя сохраняем текущий stress_suite_v1, скрипты сборки и оценки, отчёт и новые логи аудита в dev и отправляем в origin.
Почему: Диагностические материалы должны быть доступны команде. Перед коммитом проверяем запуск сборщика и оценщика.
Где: scripts/*stress*, data/test/stress_suite_v1/, output/stress_suite_v1_report.md, decisions/.

## 2026-11-15 04:44 МСК | эксперимент | процесс | локальное
Что: Успешно выполнены build_stress_suite.py и eval_stress_suite.py с отчётом в /tmp; staged diff --check с учётом CRLF прошёл.
Результат: Собрано 440 строк в 7 срезах; char-LR обучен на 4 351 строке, общий Micro-F1 77,79%, Exact Match 55,91%. Метрики воспроизводят сохранённый отчёт.
Где: scripts/build_stress_suite.py; scripts/eval_stress_suite.py; /tmp/innoglobalhack-stress-verification.md.
