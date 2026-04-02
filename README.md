# Whisper WebM Transcriber

Локальная транскрибация `.webm` записей встреч через Whisper Large с автоподбором бэкенда:
- `mlx` на macOS Apple Silicon (`mlx-whisper` + модель `mlx-community/whisper-large-v3-mlx`)
- `hf` на остальных системах (`transformers` + `openai/whisper-large-v3`)

## Quick Start: WhisperX + qwen3-30b-a3b (LM Studio)

1. Установите `ffmpeg` (например, на macOS: `brew install ffmpeg`).
2. Установите зависимости проекта и WhisperX:

```bash
uv sync --extra whisperx
```

3. Экспортируйте HF токен (нужен для diarization/pyannote):

```bash
export HF_TOKEN=hf_xxx
```

4. В LM Studio загрузите модель `qwen3-30b-a3b` и включите локальный сервер OpenAI API (`http://127.0.0.1:1234/v1`).
5. Запустите транскрибацию со спикерами и сразу саммари:

```bash
uv run python transcribe_whisperx.py meeting.webm \
  --hf-token "$HF_TOKEN" \
  --summarize \
  --summary-model qwen3-30b-a3b \
  --summary-base-url http://127.0.0.1:1234/v1
```

Результат:
- `meeting.speakers.txt` и `meeting.speakers.json` (диаризация)
- `meeting.speakers.summary.md` (режим `--summary-mode summary`, по умолчанию)
- `meeting.speakers.lecture.md` (режим `--summary-mode lecture`)
- `meeting.speakers.demo.md` (режим `--summary-mode demo`)

## 1. Установка

```bash
uv sync
```

Нужен установленный `ffmpeg` в системе.

Если на Apple Silicon хотите принудительно использовать HF-бэкенд:

```bash
uv sync --extra hf
```

### Windows + CUDA (NVIDIA)

Проект уже настроен через `pyproject.toml`: на Windows (`win32`) пакет `torch`
берется из индекса PyTorch CUDA 12.8 (`https://download.pytorch.org/whl/cu128`).

Поэтому достаточно:

```powershell
uv sync
```

Быстрая проверка:

```powershell
.venv\Scripts\python -c "import torch; print('torch=', torch.__version__, 'cuda_build=', torch.version.cuda, 'cuda_available=', torch.cuda.is_available())"
```

Для запуска без тихого fallback на CPU используйте:

```powershell
uv run python transcribe_webm.py .\meeting.webm --backend hf --device cuda
```

## 2. Запуск

```bash
uv run python transcribe_webm.py /путь/к/встрече.webm
```

По умолчанию используется `--backend auto`.

Результат по умолчанию сохранится рядом с исходным файлом:
- `встреча.txt` — полный текст
- `встреча.json` — текст + чанки и таймкоды

## 3. Явный выбор бэкенда

```bash
uv run python transcribe_webm.py meeting.webm --backend mlx
uv run python transcribe_webm.py meeting.webm --backend hf
```

## 4. Аутентификация Hugging Face (опционально)

Чтобы убрать предупреждение про unauthenticated requests и ускорить загрузку:

```bash
export HF_TOKEN=hf_xxx
uv run python transcribe_webm.py meeting.webm
```

Или передать токен напрямую:

```bash
uv run python transcribe_webm.py meeting.webm --hf-token hf_xxx
```

## 5. Полезные параметры

```bash
uv run python transcribe_webm.py meeting.webm \
  --backend auto \
  --device auto \
  --chunk-length 30 \
  --batch-size 8 \
  --hf-model-id openai/whisper-large-v3 \
  --mlx-model-id mlx-community/whisper-large-v3-mlx \
  --hf-token "$HF_TOKEN" \
  --output-txt ./out/meeting.txt \
  --output-json ./out/meeting.json
```

## 6. Разделение по спикерам (WhisperX)

Установка зависимостей WhisperX:

```bash
uv sync --extra whisperx
```

Для video speaker fusion (OCR + анализ подсветки активного говорящего):

```bash
uv sync --extra whisperx --extra video
```

Также нужен установленный бинарник `tesseract` в PATH
(например, на macOS: `brew install tesseract tesseract-lang`).

На Windows это теперь тоже поддержано (через `torch 2.8.0+cu128`).

### 6.0.1 Windows: что нужно для Video Speaker Fusion

Для `--video-speaker-fusion` на Windows дополнительно проверьте:

1. Установлены Python-зависимости:

```powershell
uv sync --extra whisperx --extra video
```

2. Установлен `ffmpeg` и доступен в `PATH`:

```powershell
ffmpeg -version
```

3. Установлен `Tesseract OCR` и доступен в `PATH`  
   (обычно путь вида `C:\Program Files\Tesseract-OCR\`).

Проверка:

```powershell
tesseract --version
tesseract --list-langs
```

В списке языков должны быть `eng` и `rus` (если используете дефолт `--video-ocr-lang rus+eng`).

Если `tesseract` не находится, после добавления в `PATH` перезапустите терминал/PowerShell.

Быстрый запуск:

```powershell
$env:HF_TOKEN = "hf_xxx"
uv run python transcribe_whisperx.py .\meeting.webm `
  --video-speaker-fusion `
  --video-profile auto
```

Запуск diarization:

```bash
export HF_TOKEN=hf_xxx
uv run python transcribe_whisperx.py meeting.webm
```

Результат по умолчанию:
- `meeting.speakers.txt` — текст в формате `[таймкод] SPEAKER_XX: ...`
- `meeting.speakers.json` — подробные сегменты/слова/спикеры

Полезные параметры:

```bash
uv run python transcribe_whisperx.py meeting.webm \
  --language ru \
  --model large-v3 \
  --device auto \
  --compute-type auto \
  --min-speakers 2 \
  --max-speakers 5 \
  --hf-token "$HF_TOKEN"
```

Примечания по WhisperX:
- для `pyannote/speaker-diarization-community-1` обычно нужен HF токен и принятие условий модели на Hugging Face;
- на Apple Silicon WhisperX обычно работает на CPU (`--device cpu`), поэтому может быть медленнее MLX-транскрибации.

### 6.1 Video Speaker Fusion

Можно уточнить `SPEAKER_XX` через видеопоток (подсветка активного говорящего + OCR имен).

Ключевые свойства:
- анализируются только окна речи из таймингов WhisperX (не весь ролик);
- после достижения порога уверенности speaker mapping lock'ается;
- после lock по умолчанию обрабатывается только каждый 6-й сегмент для верификации;
- порог уверенности конфигурируемый, по умолчанию `80%`.
- есть профиль эвристик `--video-profile`:
  - `auto` (по умолчанию) — пытается определить Телемост по видеопотоку;
  - `generic` — универсальные эвристики;
  - `yandex_telemost` — профиль, откалиброванный под Яндекс Телемост.

Важно:
- если OCR не находит надежных кандидатов имен, fusion помечается как `skipped`
  и остаются исходные `SPEAKER_XX` (это безопасный fallback без ложных имен);
- для максимальной точности рекомендуется передавать `--video-participants`
  или `--video-participants-file`.

Пример запуска:

```bash
uv run python transcribe_whisperx.py meeting.webm \
  --hf-token "$HF_TOKEN" \
  --video-speaker-fusion \
  --video-profile auto \
  --video-speaker-confidence-threshold 80 \
  --video-lock-verify-every 6
```

Для Яндекс Телемоста можно зафиксировать профиль явно:

```bash
uv run python transcribe_whisperx.py meeting.webm \
  --hf-token "$HF_TOKEN" \
  --video-speaker-fusion \
  --video-profile yandex_telemost
```

Если известен список участников, лучше передать его явно:

```bash
uv run python transcribe_whisperx.py meeting.webm \
  --hf-token "$HF_TOKEN" \
  --video-speaker-fusion \
  --video-participants "Иван Петров,Анна Смирнова,Дмитрий Соколов"
```

Или из файла (по одному имени в строке):

```bash
uv run python transcribe_whisperx.py meeting.webm \
  --hf-token "$HF_TOKEN" \
  --video-speaker-fusion \
  --video-participants-file ./participants.txt
```

## 7. Саммари/Конспект/Демо-отчет через LM Studio (OpenAI API)

После любого варианта транскрибации можно автоматически сгенерировать:
- `summary` — итоги встречи (по умолчанию);
- `lecture` — подробный конспект лекции/воркшопа (включает отдельный блок с кратким конспектом).
- `demo` — отчет по scrum-демо: показанные результаты разработки, статус целей и следующие шаги.

Генерация идет через локальный OpenAI-compatible endpoint LM Studio.

По умолчанию используется:
- `--summary-base-url http://127.0.0.1:1234/v1`
- `--summary-model local-model`
- `--summary-api-key` берется из `OPENAI_API_KEY` (если не задан, используется `lm-studio`)
- `--summary-retries 3`
- `--summary-retry-delay 2.0`
- `--summary-chunk-chars 12000`
- `--summary-chunk-overlap-chars 300`
- `--summary-mode summary` (по умолчанию), `--summary-mode lecture` или `--summary-mode demo`
- встроенный mode-specific промпт (если не передан `--summary-prompt`)

Если используете не LM Studio, а внешний OpenAI-compatible endpoint, задайте API ключ:

```bash
export OPENAI_API_KEY=sk-xxx
```

Если стенограмма не влезает в один запрос, текст автоматически делится на чанки,
каждый чанк обрабатывается отдельно, затем промежуточные результаты объединяются в итоговый.

Пример с `transcribe_webm.py`:

```bash
uv run python transcribe_webm.py meeting.webm \
  --backend auto \
  --summarize \
  --summary-mode summary \
  --summary-model your-lmstudio-model \
  --summary-api-key "$OPENAI_API_KEY"
```

Пример с `transcribe_whisperx.py`:

```bash
uv run python transcribe_whisperx.py meeting.webm \
  --device auto \
  --summarize \
  --summary-mode summary \
  --summary-model your-lmstudio-model \
  --summary-api-key "$OPENAI_API_KEY"
```

Пример подробного конспекта лекции/воркшопа:

```bash
uv run python transcribe_whisperx.py lecture.webm \
  --device auto \
  --summarize \
  --summary-mode lecture \
  --summary-model your-lmstudio-model \
  --summary-api-key "$OPENAI_API_KEY"
```

Пример отчета по scrum-демо:

```bash
uv run python transcribe_whisperx.py sprint_demo.webm \
  --device auto \
  --summarize \
  --summary-mode demo \
  --summary-model your-lmstudio-model \
  --summary-api-key "$OPENAI_API_KEY"
```

Если нужно, можно передать кастомный промпт:

```bash
uv run python transcribe_webm.py meeting.webm \
  --summarize \
  --summary-model your-lmstudio-model \
  --summary-timeout 1200 \
  --summary-chunk-chars 10000 \
  --summary-prompt "Ваш уточненный промпт"
```

Отдельный запуск саммари по уже готовому JSON транскрибации:

```bash
uv run python summarize_transcript_json.py meeting.json \
  --summary-mode summary \
  --summary-model qwen3-30b-a3b
```

Для JSON со спикерами (`meeting.speakers.json`) можно принудительно использовать
формат со спикер-метками:

```bash
uv run python summarize_transcript_json.py meeting.speakers.json \
  --speaker-format always \
  --summary-mode summary \
  --summary-model qwen3-30b-a3b
```

Отдельный запуск конспекта лекции по JSON:

```bash
uv run python summarize_transcript_json.py lecture.speakers.json \
  --speaker-format always \
  --summary-mode lecture \
  --summary-model qwen3-30b-a3b
```

Отдельный запуск отчета по demo-встрече из JSON:

```bash
uv run python summarize_transcript_json.py sprint_demo.speakers.json \
  --speaker-format always \
  --summary-mode demo \
  --summary-model qwen3-30b-a3b
```

Параметр `--speaker-format`:
- `auto` (по умолчанию): если в JSON есть speaker-сегменты, генерация строится по строкам вида `SPEAKER_XX: ...`, иначе по обычному тексту.
- `always`: принудительно использовать speaker-формат; если speaker-сегментов нет, скрипт завершится с ошибкой.
- `never`: всегда использовать plain text без speaker-меток.

## Примечание

Скрипт принудительно запускает транскрибацию на русском (`language=russian`, `task=transcribe`), чтобы не было авто-перевода на английский.
