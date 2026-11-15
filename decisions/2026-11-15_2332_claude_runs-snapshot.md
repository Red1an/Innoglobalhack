## 2026-11-15 23:32 МСК | эксперимент | классификация | глобальное
Что: снимок всех прогонов private_eval на 15.11 23:32. Папка private_eval/runs в git не попадает, потому что в ней данные организаторов, поэтому для коллег сводка лежит здесь.
Почему: коллегам нужна история прогонов, а сами прогоны лежат только на одном ноутбуке.
Результат: 32 прогона. Лучший — GLM 5.3 Flash + classify_v9 + prep: 0,9917 на private-test-dataset.labeled, 1,0 на guide_v3_hard, 0,9888 на adversarial. Строка private-test-dataset без суффикса .labeled — старая копия того же набора. Прогоны example на 2 строки не показательны.
Где: private_eval/runs/<набор>/<папка>, свежую таблицу печатает .venv/bin/python private_eval/run.py list

| набор | время МСК | метод | бэкенд | модель | промпт | prep | строк | micro-F1 | P | R | ложные непустые | классиф., с | вызовов LLM | ошибок | токены вход/выход | папка |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| example | 15.11 17:28 | cascade | local | avibe-GGUF:Q8_0 | — | нет | 2 | 0.6667 | 1.0000 | 0.5000 | 0/1 | 13.3 | 1 | 0 | 454/6 | cascade-local-avibe-q8-seo_on |
| example | 15.11 17:33 | llm | api | qwen3-235b-a22b-2507 | classify_v3_seo_on | нет | 2 | 0.4000 | 0.3333 | 0.5000 | 1/1 | 0.9 | 2 | 2 | 0/0 | llm-api-qwen3-235b-instruct-classify_v3_seo_on |
| example | 15.11 18:45 | llm | local | avibe-GGUF:Q8_0 | classify_v3_seo_on | нет | 2 | 1.0000 | 1.0000 | 1.0000 | 0/1 | 4.4 | 2 | 0 | 2936/20 | llm-local-avibe-q8-classify_v3_seo_on |
| example | 15.11 18:54 | charlr |  | Char-LR | — | нет | 2 | 0.4000 | 0.3333 | 0.5000 | 1/1 | 0.0 | 0 | 0 | 0/0 | charlr |
| example | 15.11 18:54 | llm | local | t-lite-it-2.1:q8_0 | classify_v3_seo_on | нет | 2 | 1.0000 | 1.0000 | 1.0000 | 0/1 | 0.0 | 2 | 0 | 3304/20 | llm-local-t-lite-q8-classify_v3_seo_on |
| guide_v3_hard | 15.11 17:32 | cascade | local | avibe-GGUF:Q8_0 | — | нет | 50 | 0.6866 | 0.7188 | 0.6571 | 6/13 | 70.0 | 22 | 0 | 9708/155 | cascade-local-avibe-q8-seo_on |
| guide_v3_hard | 15.11 18:40 | charlr |  | Char-LR | — | нет | 50 | 0.6065 | 0.5529 | 0.6714 | 11/13 | 0.0 | 0 | 0 | 0/0 | charlr |
| guide_v3_hard | 15.11 18:41 | llm | local | avibe-GGUF:Q8_0 | classify_v3_seo_on | нет | 50 | 0.8199 | 0.7253 | 0.9429 | 6/13 | 0.0 | 50 | 0 | 73907/671 | llm-local-avibe-q8-classify_v3_seo_on |
| guide_v3_hard | 15.11 19:09 | llm | api | qwen3-235b-a22b-2507 | classify_v3_seo_on | нет | 50 | 0.6065 | 0.5529 | 0.6714 | 11/13 | 6.8 | 50 | 50 | 0/0 | llm-api-qwen3-235b-instruct-classify_v3_seo_on |
| guide_v3_hard | 15.11 20:02 | llm | api | qwen3-235b-a22b-2507 | classify_v5 | да | 50 | 0.9275 | 0.9412 | 0.9143 | 1/13 | 24.6 | 50 | 0 | 158259/564 | llm-api-qwen3-235b-instruct-classify_v5-prep |
| guide_v3_hard | 15.11 20:15 | llm | api | qwen3-235b-a22b-2507 | classify_v6 | да | 50 | 0.9209 | 0.9275 | 0.9143 | 1/13 | 30.0 | 50 | 0 | 162459/569 | llm-api-qwen3-235b-instruct-classify_v6-prep |
| guide_v3_hard | 15.11 20:32 | llm | api | qwen3-235b-a22b-2507 | classify_v7 | да | 50 | 0.9143 | 0.9143 | 0.9143 | 1/13 | 33.1 | 50 | 0 | 176109/2047 | llm-api-qwen3-235b-instruct-classify_v7-prep |
| guide_v3_hard | 15.11 20:39 | llm | api | qwen3-235b-a22b-2507 | classify_v8 | да | 50 | 0.8889 | 0.8193 | 0.9714 | 3/13 | 39.2 | 50 | 0 | 181759/2862 | llm-api-qwen3-235b-instruct-classify_v8-prep |
| guide_v3_hard | 15.11 23:21 | llm | api | glm-5.3-flash | classify_v9 | да | 50 | 1.0000 | 1.0000 | 1.0000 | 0/13 | 43.3 | 50 | 0 | 168514/6851 | llm-api-glm-5.3-flash-classify_v9-prep |
| private-test-dataset | 15.11 18:18 | charlr |  | Char-LR | — | нет | 262 | 0.8234 | 0.8242 | 0.8227 | 23/55 | 0.6 | 0 | 0 | 0/0 | charlr |
| private-test-dataset | 15.11 18:18 | llm | api | qwen3-235b-a22b-2507 | classify_v3_seo_on | нет | 262 | 0.8234 | 0.8242 | 0.8227 | 23/55 | 30.3 | 262 | 262 | 0/0 | llm-api-qwen3-235b-instruct-classify_v3_seo_on |
| private-test-dataset.adversarial.labeled | 15.11 23:27 | llm | api | glm-5.3-flash | classify_v9 | да | 170 | 0.9888 | 0.9779 | 1.0000 | 2/46 | 176.1 | 170 | 0 | 662211/23094 | llm-api-glm-5.3-flash-classify_v9-prep |
| private-test-dataset.labeled | 15.11 18:57 | charlr |  | Char-LR | — | нет | 262 | 0.8234 | 0.8242 | 0.8227 | 23/55 | 0.6 | 0 | 0 | 0/0 | charlr |
| private-test-dataset.labeled | 15.11 19:17 | llm | api | qwen3-235b-a22b-2507 | classify_v3_seo_on | нет | 262 | 0.9065 | 0.8920 | 0.9214 | 4/55 | 173.0 | 262 | 1 | 786392/4023 | llm-api-qwen3-235b-instruct-classify_v3_seo_on |
| private-test-dataset.labeled | 15.11 19:20 | llm | api | qwen3-235b-a22b-2507 | classify_v2_seo_off | нет | 262 | 0.9165 | 0.8740 | 0.9634 | 7/55 | 122.6 | 262 | 0 | 769796/4177 | llm-api-qwen3-235b-instruct-classify_v2_seo_off |
| private-test-dataset.labeled | 15.11 19:34 | llm | api | qwen3-235b-a22b-2507 | classify_v4 | нет | 262 | 0.9520 | 0.9257 | 0.9799 | 3/55 | 132.8 | 262 | 0 | 934070/4059 | llm-api-qwen3-235b-instruct-classify_v4 |
| private-test-dataset.labeled | 15.11 19:37 | llm | api | qwen3-235b-a22b-2507 | classify_v1 | нет | 262 | 0.8754 | 0.8021 | 0.9634 | 15/55 | 143.1 | 262 | 0 | 453038/4431 | llm-api-qwen3-235b-instruct-classify_v1 |
| private-test-dataset.labeled | 15.11 19:40 | llm | api | qwen3-235b-a22b-2507 | classify_v4 | да | 262 | 0.9504 | 0.9211 | 0.9817 | 3/55 | 131.2 | 262 | 0 | 931129/4077 | llm-api-qwen3-235b-instruct-classify_v4-prep |
| private-test-dataset.labeled | 15.11 19:42 | charlr |  | Char-LR | — | да | 262 | 0.8294 | 0.8324 | 0.8263 | 22/55 | 0.6 | 0 | 0 | 0/0 | charlr-prep |
| private-test-dataset.labeled | 15.11 20:05 | llm | api | qwen3-235b-a22b-2507 | classify_v5 | да | 262 | 0.9528 | 0.9459 | 0.9598 | 1/55 | 148.8 | 262 | 0 | 1056627/3955 | llm-api-qwen3-235b-instruct-classify_v5-prep |
| private-test-dataset.labeled | 15.11 20:14 | llm | api | qwen3-235b-a22b-2507 | classify_v6 | да | 262 | 0.9557 | 0.9446 | 0.9671 | 1/55 | 153.9 | 262 | 0 | 1078635/3976 | llm-api-qwen3-235b-instruct-classify_v6-prep |
| private-test-dataset.labeled | 15.11 20:35 | llm | api | qwen3-235b-a22b-2507 | classify_v7 | да | 262 | 0.9508 | 0.9474 | 0.9543 | 2/55 | 181.8 | 262 | 0 | 1150161/15147 | llm-api-qwen3-235b-instruct-classify_v7-prep |
| private-test-dataset.labeled | 15.11 20:53 | llm | api | qwen3-235b-a22b-2507 | classify_v8 | да | 262 | 0.9576 | 0.9265 | 0.9909 | 9/55 | 223.2 | 262 | 0 | 1179767/16901 | llm-api-qwen3-235b-instruct-classify_v8-prep |
| private-test-dataset.labeled | 15.11 20:58 | llm | api | qwen3-235b-a22b-2507 | classify_v9 | нет | 262 | 0.9615 | 0.9633 | 0.9598 | 3/55 | 274.9 | 262 | 0 | 1292486/32589 | llm-api-qwen3-235b-instruct-classify_v9 |
| private-test-dataset.labeled | 15.11 22:50 | cascade | api | qwen3-235b-a22b-2507 | — | да | 262 | 0.8606 | 0.8752 | 0.8464 | 13/55 | 228.9 | 96 | 0 | 205996/977 | cascade-api-qwen3-235b-instruct-prep |
| private-test-dataset.labeled | 15.11 23:07 | cascade | local | avibe-GGUF:Q8_0 | — | да | 262 | 0.8505 | 0.8642 | 0.8373 | 14/55 | 962.4 | 96 | 0 | 94841/1003 | cascade-local-avibe-q8-prep |
| private-test-dataset.labeled | 15.11 23:20 | llm | api | glm-5.3-flash | classify_v9 | да | 262 | 0.9917 | 0.9963 | 0.9872 | 0/55 | 227.1 | 262 | 0 | 1062306/37464 | llm-api-glm-5.3-flash-classify_v9-prep |
