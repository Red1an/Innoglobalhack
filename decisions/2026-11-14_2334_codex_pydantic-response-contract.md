## 2026-11-14 23:34 МСК | решение | инфраструктура | локальное
Что: по запросу пользователя проверена Python-обработка и Pydantic: 2.11.5 уже есть в системном и клиентском окружении. Старое дерево notebook PID19512 остановлено для исправления доказанного отклонения полноценных Yandex JSON из-за finish_reason=null.
Почему: смешение ошибок формата, метаданных адаптера и ошибок классификации приводило к преждевременному отсеву Pro с F1 0.9421 и нулевому F1 у Yandex. Новый протокол: schema Pydantic, контролируемая нормализация с логом, отдельная строгая метрика, stop только по достижимости F1. {} остаётся ошибкой required labels, без догадки о семантике.
Где: research/experiments/api_bench.py; новый response_contract.py; новая группа baseline-dev-schema-v3.

## 2026-11-14 23:39 МСК | эксперимент | инфраструктура | локальное
Что: реализован response_contract.py на Pydantic2.11.5: required labels, strict integer ID101–106, extra forbid; снимается целая Markdown-обёртка, удаляется исходный ID и дубли, события явно записаны. finish_reason=null не отклоняет завершённый JSON; явный length/filter/refusal отклоняется. {} остаётся missing_labels. Отсев только по оптимистическому F1<0.80 после30; max-invalid-rate оставлен как неактивный совместимый параметр.
Результат: 28 тестов прошли; reviewer не нашёл блокирующих дефектов, замечание о названии companion-метрики учтено: legacy_strict_micro_f1 и legacy_strict_invalid_rate обозначают прежний парсер, не строгость Pydantic. Проверены py_compile, nbformat, AST ячеек, diff --check, хеши Colab-архива. Системный PROMPT dev не изменён.
Где: research/experiments/response_contract.py, api_bench.py, publish_clearml.py, api_colab.ipynb, requirements-api.txt.

## 2026-11-14 23:39 МСК | эксперимент | метрика | локальное
Что: без API-запросов перепроверены129 сохранённых ответов v2, пригодных для переноса в baseline-dev-schema-v3. Механизм --reuse-from сверяет модель/параметры, prompt, baseURL, полный test и row_hash; повторно разбирает raw, сохраняет reuse.json и запрашивает только остаток.
Результат: по первым30 GigaChat F1=0.841270, Pro=0.942149, Yandex Lite=0.845528, Yandex=0.859649; Qwen по9 F1=0.909091. Это частичные новые оценки, не результат полного139; ни один не должен быть остановлен новым правилом. Старые четыре ClearML-задачи v2 помечены superseded-response-contract, метрики не переписаны. Новый launcher PID23196 ожидает скрытый ввод ключа для продолжения.
Где: research/api-runs/member-api/baseline-dev-schema-v3; notebook; research/api-runs/member-api/baseline-dev-gold-v2.

Примечание: начальная запись этого файла с заголовком23:34 была фактически сделана23:33 МСК (время первичного Get-Date); дальнейшие заголовки получены программно.
