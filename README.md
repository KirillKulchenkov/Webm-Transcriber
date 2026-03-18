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
- `meeting.speakers.summary.md` (саммари)

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

На Windows это теперь тоже поддержано (через `torch 2.8.0+cu128`).

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

## 7. Саммари через LM Studio (OpenAI API)

После любого варианта транскрибации можно автоматически сделать summary через
локальный OpenAI-compatible endpoint LM Studio.

По умолчанию используется:
- `--summary-base-url http://127.0.0.1:1234/v1`
- `--summary-model local-model`
- `--summary-retries 3`
- `--summary-retry-delay 2.0`
- встроенный промпт для структурированного саммари на русском

Пример с `transcribe_webm.py`:

```bash
uv run python transcribe_webm.py meeting.webm \
  --backend auto \
  --summarize \
  --summary-model your-lmstudio-model
```

Пример с `transcribe_whisperx.py`:

```bash
uv run python transcribe_whisperx.py meeting.webm \
  --device auto \
  --summarize \
  --summary-model your-lmstudio-model
```

Если нужно, можно передать кастомный промпт:

```bash
uv run python transcribe_webm.py meeting.webm \
  --summarize \
  --summary-model your-lmstudio-model \
  --summary-prompt "Ваш уточненный промпт"
```

## Примечание

Скрипт принудительно запускает транскрибацию на русском (`language=russian`, `task=transcribe`), чтобы не было авто-перевода на английский.
