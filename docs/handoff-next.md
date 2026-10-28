# Что осталось доделать (передача работы, 28.10.2026)

Документ для следующего исполнителя (любая модель). Всё по шагам. Ничего не угадывай: сначала прочитай названные файлы.

## Жёсткие правила
1. Не делай `git reset/checkout/stash/commit`: в рабочей копии много незакоммиченных правок, и они нужны.
2. Бот — отдельная модель (`gpt-6.1-terra`, medium). Он не должен наследовать `AGENTS.md` и правила координатора Codex.
3. Реальные проверки модели — только `gpt-6-luna` / `medium`, только синтетические данные. Telegram в проверках подставной (`OfflineTelegram` из `scripts/verify_conversation.py`). Никаких сообщений в настоящий Telegram.
4. В любом скрипте с реальной моделью вызывай `personal_agent.testing.block_host_desktop()`, чтобы настоящий экран не попал в модель.
5. Тесты запускай так: `.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider`. Сейчас: **348 passed, 1 skipped**. После твоих правок должно быть не хуже.
6. `codex` нет в PATH. Используй `resolve_codex_command("codex")` из `personal_agent/backends.py`.

## Как устроен бот сейчас (кратко)
- `personal_agent/brain_gateway.py` — шлюз к модели. Передаёт нативные function tools и сам ведёт цикл: вызов инструмента → результат → продолжение. Инструмент может вернуть картинки (`ToolOutcome.images`, не больше 8 на ход, до 10 МБ каждая).
- `personal_agent/tools.py` — все инструменты: схемы и обработчики в классе `TurnTools`; список собирается внизу файла.
- `personal_agent/supervisor.py` — один вызов модели на сообщение. Заготовленных распознавателей фраз нет: модель сама выбирает инструменты.
- `personal_agent/attachments.py` — фото, стикеры, документы и голосовые из Telegram.
- Бот запущен (`Start-PersonalAgent.ps1`, PID в `.state\daemon.pid`). Новый код подхватывается только после перезапуска (шаг 5 ниже).

## Задача 1 (главная, начата): просмотр видео
Цель: ссылку на TikTok/YouTube и других сайтов или видео, присланное в Telegram, бот «смотрит». Модель получает кадры (листы-раскадровки с метками времени), субтитры или расшифровку речи и метаданные (название, автор, описание).

Уже сделано, но **не проверено и не покрыто тестами**:
- `personal_agent/video.py` (~1050 строк): `VideoAnalyzer`. Скачивание идёт через `yt-dlp` в отдельном процессе (`python -m personal_agent.video --worker`). Там же выбор формата ≤480p, субтитры (`pick_subtitle_track`, `parse_subtitles`), сетка кадров (`compose_sheets`), кэш по отправителю в `data_dir/memory/videos/<sender>/<id>/` и повторный просмотр отрезка (`start_seconds`/`end_seconds`).
- `personal_agent/tools.py`: инструмент `watch_video(url?, media_id?, start_seconds?, end_seconds?)` добавлен в список (около строки 631), обработчик `TurnTools.watch_video` — около строки 440.
- `personal_agent/backends.py`: в персону добавлена строка про `watch_video`.
- `personal_agent/testing.py`: `make_synthetic_video(...)` создаёт тестовое mp4 (цветные сегменты, крупное слово, тон или WAV).
- `yt-dlp 2026.10.26` установлен в `.venv`, но **не записан в `pyproject.toml`**.

Что сделать, по порядку:
1. Прочитай `personal_agent/video.py` целиком и проверь, что модуль импортируется и что `--worker` запускается: `.venv\Scripts\python.exe -c "import personal_agent.video"`.
2. Добавь `yt-dlp` в `dependencies` в `pyproject.toml`. `uv.lock` обнови, если `uv` есть, иначе оставь как есть.
3. Проверь без модели на временной папке (не в `.state`):
   - YouTube: `https://www.youtube.com/watch?v=jNQXAC9IVRw`;
   - любой публичный ролик TikTok.
   Должны появиться метаданные, субтитры или расшифровка и 2–4 листа кадров. Если YouTube требует JS-движок, поставь pip-пакеты внутрь `.venv` (`yt-dlp[default]`, `yt-dlp-ejs` или PyPI `deno`). Ничего не ставь в систему; cookies браузера не использовать.
4. Видео из Telegram: в `personal_agent/telegram.py` (`parse_message`) и `personal_agent/attachments.py` подключи `video`, `video_note`, `animation` и видео-документы к `VideoAnalyzer`. Кадры и расшифровка должны уходить модели в текущем ходе, как фото. Лимит Telegram на скачивание ботом — 20 МБ; учитывай `config.max_attachment_bytes`.
5. Видео должно появляться в каталоге `recent_media` (вид `video`), чтобы на «а что в конце?» модель вызвала `watch_video(media_id=..., start_seconds=...)`.
6. Очистка: добавь `memory/videos` в `cleanup_retained_data` в `personal_agent/memory.py` (7 дней, как скриншоты).
7. Офлайн-тесты (новый `tests/test_video.py`, `yt-dlp` замокан):
   - выбор формата;
   - лимиты длины и размера;
   - субтитры имеют приоритет над Whisper;
   - запасной путь через Whisper;
   - сетка кадров;
   - отмена убивает дочерний процесс;
   - отказ для приватного IP и `file://`;
   - видео одного отправителя недоступно другому;
   - видео-вложение из Telegram (через `make_synthetic_video`).
8. Реальная проверка на Luna: синтетическое видео как вложение Telegram через `TaskSupervisor` плюс продолжение про конец ролика. Образец — `scripts/verify_agent_tools_real.py`.
9. Полный прогон тестов. Коротко обнови `docs/status.md` и README: что сделано и что не проверено.

## Задача 2: проверить бота с новым кодом
1. Перезапуск, когда бот не отвечает (в `.state\memory\state.sqlite3` ключ `active_turns` = `{}`):
   - останови процессы с `personal_agent run` в командной строке;
   - запусти `Start-PersonalAgent.ps1`;
   - проверь, что `telegram_health.updated_at` обновился и что `.state\logs\*.stderr.log` пуст.
2. Пользователь должен один раз нажать «Обновить» у расширения в `edge://extensions` (версия 0.1.1), чтобы заработал снимок страницы в момент запроса. Он может ещё этого не сделать.

## Задача 3 (необязательно, позже)
- Страницы на JavaScript без текста (TikTok отдаёт пустую оболочку): рендер через невидимый Edge (`msedge --headless --dump-dom`) для публичных URL с той же проверкой адреса.
- Риск DNS rebinding в `web_open`: закрепить IP после проверки.
- Один нестабильный прогон `scripts/verify_profile_attachment_real.py` (1 из 9).

## Ручные тесты для пользователя
`C:\Users\nagas\Desktop\Тесты Personal Agent.md`. Когда появится видео, добавь туда раздел «Видео».
