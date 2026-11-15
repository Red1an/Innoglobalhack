## 2026-11-14 23:57 МСК | находка | инфраструктура | локальное
Что: пользователь спросил, почему прогон прервался. Процесс23196 завершился23:52:45 CellExecutionError после неуспеха yandexgpt/qwen/deepseek. Основной статус failed; это не early-stop. Завершены и выгружены по139: GigaChat0.812680, Pro0.936047, YandexLite0.821229. Yandex сохранил124, Qwen10, DeepSeek0.
Почему: ошибка дочернего API-процесса не попала в Jupyter outputs на Windows; точно назвать HTTP-код/причину по сохранённым логам нельзя. GET публичного каталога Caila сейчас отвечает200, это не проверка доступности inference/квоты ключа.
Где: research/api-runs/member-api/baseline-dev-schema-v3; api_colab.executed.ipynb.
