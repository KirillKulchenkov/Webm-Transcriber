from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_SUMMARY_PROMPT = (
    "Ты мой эффективный AI-ассистент по анализу стенограмм совещаний и лекций. "
    "Вот тебе текст разговора нескольких людей. Сделай из него структурированное "
    "саммари на русском языке. В саммари обязательно выдели следующие пункты "
    "(можно использовать маркированные списки):\n"
    "1. Основные обсуждавшиеся темы или вопросы.\n"
    "2. Ключевые аргументы, предложения или идеи, высказанные участниками (если были).\n"
    "3. Принятые решения (если таковые были).\n"
    "4. Поставленные задачи с указанием ответственных лиц "
    "(если это можно однозначно понять из текста).\n"
    "5. Главные выводы или итоги обсуждения."
)


def add_summary_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="После транскрибации дополнительно сделать саммари через OpenAI-compatible API (LM Studio).",
    )
    parser.add_argument(
        "--summary-base-url",
        default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1"),
        help="Базовый URL OpenAI-compatible API (по умолчанию: http://127.0.0.1:1234/v1)",
    )
    parser.add_argument(
        "--summary-model",
        default=os.getenv("SUMMARY_MODEL", "local-model"),
        help="Имя LLM модели в LM Studio (по умолчанию: local-model)",
    )
    parser.add_argument(
        "--summary-api-key",
        default=os.getenv("OPENAI_API_KEY", "lm-studio"),
        help="API key для OpenAI-compatible API (по умолчанию: OPENAI_API_KEY или lm-studio)",
    )
    parser.add_argument(
        "--summary-temperature",
        type=float,
        default=0.1,
        help="Temperature для генерации саммари (по умолчанию: 0.1)",
    )
    parser.add_argument(
        "--summary-timeout",
        type=int,
        default=300,
        help="Таймаут HTTP-запроса на саммари в секундах (по умолчанию: 300)",
    )
    parser.add_argument(
        "--summary-retries",
        type=int,
        default=3,
        help="Количество попыток запроса саммари (по умолчанию: 3)",
    )
    parser.add_argument(
        "--summary-retry-delay",
        type=float,
        default=2.0,
        help="Пауза между попытками саммари в секундах (по умолчанию: 2.0)",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Куда сохранить саммари (по умолчанию: рядом с транскриптом, *.summary.md)",
    )
    parser.add_argument(
        "--summary-prompt",
        default=DEFAULT_SUMMARY_PROMPT,
        help="Промпт для саммари. Можно передать свой.",
    )


def default_summary_output_path(transcript_txt_path: Path) -> Path:
    return transcript_txt_path.with_suffix(".summary.md")


def save_summary(summary_text: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(summary_text.strip() + "\n", encoding="utf-8")
    return output_path


def _extract_content(chat_response: dict[str, Any]) -> str:
    choices = chat_response.get("choices")
    if not choices or not isinstance(choices, list):
        raise RuntimeError("В ответе LLM отсутствует поле choices.")

    message = choices[0].get("message", {})
    content = message.get("content", "")

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
        return "\n".join(parts).strip()

    return str(content).strip()


def request_summary(
    transcript_text: str,
    *,
    base_url: str,
    model: str,
    api_key: str | None,
    prompt: str,
    temperature: float,
    timeout: int,
    retries: int = 3,
    retry_delay: float = 2.0,
) -> str:
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    "Ниже стенограмма разговора. "
                    "Подготовь структурированное саммари по инструкции.\n\n"
                    f"{transcript_text}"
                ),
            },
        ],
    }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = Request(
        endpoint,
        method="POST",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
    )

    attempts = max(1, retries)
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8")

            parsed = json.loads(body)
            summary = _extract_content(parsed)
            if not summary:
                raise RuntimeError("LLM вернул пустое саммари.")
            return summary
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="ignore")
            # Retry only for transient HTTP failures.
            retryable = exc.code >= 500 or exc.code in {408, 409, 429}
            last_error = RuntimeError(
                f"Ошибка LLM API ({exc.code}): {error_body or exc.reason}"
            )
            if not retryable or attempt >= attempts:
                raise last_error from exc
        except URLError as exc:
            last_error = RuntimeError(
                "Не удалось подключиться к LLM API. "
                "Проверьте, что LM Studio запущен и API доступен."
            )
            if attempt >= attempts:
                raise last_error from exc
        except json.JSONDecodeError as exc:
            last_error = RuntimeError(f"Некорректный JSON от LLM API: {body[:300]}")
            if attempt >= attempts:
                raise last_error from exc
        except RuntimeError as exc:
            last_error = exc
            if attempt >= attempts:
                raise

        sleep_seconds = max(0.0, retry_delay) * attempt
        print(
            f"[warn] Саммари: попытка {attempt}/{attempts} не удалась. "
            f"Повтор через {sleep_seconds:.1f}с..."
        )
        time.sleep(sleep_seconds)

    if last_error is not None:
        raise last_error
    raise RuntimeError("Не удалось получить саммари по неизвестной причине.")
