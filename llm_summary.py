from __future__ import annotations

import argparse
import json
import os
import socket
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
DEFAULT_SUMMARY_CHUNK_CHARS = int(os.getenv("SUMMARY_CHUNK_CHARS", "32000"))
DEFAULT_SUMMARY_CHUNK_OVERLAP_CHARS = int(
    os.getenv("SUMMARY_CHUNK_OVERLAP_CHARS", "3000")
)


def add_summary_args(
    parser: argparse.ArgumentParser,
    *,
    include_toggle: bool = True,
) -> None:
    if include_toggle:
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
        "--summary-chunk-chars",
        type=int,
        default=DEFAULT_SUMMARY_CHUNK_CHARS,
        help=(
            "Максимальный размер одного чанка стенограммы в символах "
            f"(по умолчанию: {DEFAULT_SUMMARY_CHUNK_CHARS})"
        ),
    )
    parser.add_argument(
        "--summary-chunk-overlap-chars",
        type=int,
        default=DEFAULT_SUMMARY_CHUNK_OVERLAP_CHARS,
        help=(
            "Перекрытие чанков стенограммы в символах "
            f"(по умолчанию: {DEFAULT_SUMMARY_CHUNK_OVERLAP_CHARS})"
        ),
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


def _split_text_chunks(
    text: str,
    *,
    max_chars: int,
    overlap_chars: int,
) -> list[str]:
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return []

    if max_chars <= 0 or len(normalized) <= max_chars:
        return [normalized]

    safe_overlap = max(0, min(overlap_chars, max_chars // 4))
    chunks: list[str] = []
    start = 0
    total = len(normalized)

    while start < total:
        hard_end = min(total, start + max_chars)
        end = hard_end

        if hard_end < total:
            search_from = min(total, start + max_chars // 2)
            newline_pos = normalized.rfind("\n", search_from, hard_end)
            space_pos = normalized.rfind(" ", search_from, hard_end)
            split_pos = max(newline_pos, space_pos)
            if split_pos > start:
                end = split_pos

        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= total:
            break

        next_start = end - safe_overlap if safe_overlap else end
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


def _format_blocks(blocks: list[str], label: str) -> str:
    parts: list[str] = []
    for idx, block in enumerate(blocks, start=1):
        cleaned = block.strip()
        if not cleaned:
            continue
        parts.append(f"{label} {idx}:\n{cleaned}")
    return "\n\n".join(parts).strip()


def _group_blocks_for_reduce(blocks: list[str], *, max_chars: int) -> list[str]:
    groups: list[str] = []
    current: list[str] = []
    current_len = 0

    for idx, block in enumerate(blocks, start=1):
        entry = f"Промежуточное саммари {idx}:\n{block.strip()}\n"
        entry_len = len(entry)

        if entry_len > max_chars:
            if current:
                groups.append("".join(current).strip())
                current = []
                current_len = 0
            groups.extend(
                _split_text_chunks(
                    entry,
                    max_chars=max_chars,
                    overlap_chars=0,
                )
            )
            continue

        if current and current_len + entry_len > max_chars:
            groups.append("".join(current).strip())
            current = [entry]
            current_len = entry_len
            continue

        current.append(entry)
        current_len += entry_len

    if current:
        groups.append("".join(current).strip())

    return [group for group in groups if group]


def _request_summary_single(
    user_content: str,
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
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    attempts = max(1, retries)
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        body = ""
        try:
            payload = {
                "model": model,
                "temperature": temperature,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_content},
                ],
            }
            request = Request(
                endpoint,
                method="POST",
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers=headers,
            )
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
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                last_error = RuntimeError(
                    "Таймаут ожидания ответа от LLM API. "
                    "Увеличьте --summary-timeout или уменьшите размер стенограммы."
                )
            else:
                last_error = RuntimeError(
                    "Не удалось подключиться к LLM API. "
                    "Проверьте, что LM Studio запущен и API доступен."
                )
            if attempt >= attempts:
                raise last_error from exc
        except (TimeoutError, socket.timeout) as exc:
            last_error = RuntimeError(
                "Таймаут ожидания ответа от LLM API. "
                "Увеличьте --summary-timeout или уменьшите размер стенограммы."
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
    chunk_chars: int = DEFAULT_SUMMARY_CHUNK_CHARS,
    chunk_overlap_chars: int = DEFAULT_SUMMARY_CHUNK_OVERLAP_CHARS,
) -> str:
    normalized = transcript_text.strip()
    if not normalized:
        raise RuntimeError("Пустая стенограмма для саммари.")

    safe_chunk_chars = max(1000, chunk_chars)
    safe_overlap = max(0, chunk_overlap_chars)
    chunks = _split_text_chunks(
        normalized,
        max_chars=safe_chunk_chars,
        overlap_chars=safe_overlap,
    )

    if len(chunks) <= 1:
        return _request_summary_single(
            user_content=(
                "Ниже стенограмма разговора. "
                "Подготовь структурированное саммари по инструкции.\n\n"
                f"{normalized}"
            ),
            base_url=base_url,
            model=model,
            api_key=api_key,
            prompt=prompt,
            temperature=temperature,
            timeout=timeout,
            retries=retries,
            retry_delay=retry_delay,
        )

    print(
        f"[info] Саммари: длинная стенограмма, делю на чанки: {len(chunks)} "
        f"(chunk_chars={safe_chunk_chars}, overlap={safe_overlap})"
    )
    partial_summaries: list[str] = []

    for idx, chunk in enumerate(chunks, start=1):
        print(f"[info] Саммари: обрабатываю чанк {idx}/{len(chunks)}...")
        partial = _request_summary_single(
            user_content=(
                "Ниже часть стенограммы разговора. "
                f"Это часть {idx} из {len(chunks)}. "
                "Сделай структурированное саммари ТОЛЬКО по этой части "
                "и не придумывай факты вне текста.\n\n"
                f"{chunk}"
            ),
            base_url=base_url,
            model=model,
            api_key=api_key,
            prompt=prompt,
            temperature=temperature,
            timeout=timeout,
            retries=retries,
            retry_delay=retry_delay,
        )
        partial_summaries.append(partial)

    reduce_prompt = (
        "Ты объединяешь промежуточные саммари одной и той же встречи. "
        "Сохраняй факты, убирай дубли, не добавляй новых деталей."
    )
    reduce_round = 1
    current = [item.strip() for item in partial_summaries if item.strip()]

    while len(current) > 1:
        grouped = _group_blocks_for_reduce(current, max_chars=safe_chunk_chars)
        if len(grouped) == 1:
            break

        print(
            f"[info] Саммари: reduce-раунд {reduce_round}, групп для объединения: {len(grouped)}"
        )
        next_level: list[str] = []
        for group_idx, group_text in enumerate(grouped, start=1):
            print(
                f"[info] Саммари: агрегирую группу {group_idx}/{len(grouped)}..."
            )
            reduced = _request_summary_single(
                user_content=(
                    "Ниже несколько промежуточных саммари частей одной встречи. "
                    "Объедини их в единое краткое саммари без повторов и без новых фактов.\n\n"
                    f"{group_text}"
                ),
                base_url=base_url,
                model=model,
                api_key=api_key,
                prompt=reduce_prompt,
                temperature=temperature,
                timeout=timeout,
                retries=retries,
                retry_delay=retry_delay,
            )
            next_level.append(reduced)
        current = next_level
        reduce_round += 1

    final_input = _format_blocks(current, "Промежуточное саммари")
    return _request_summary_single(
        user_content=(
            "Ниже промежуточные саммари частей одной и той же встречи. "
            "Собери единое итоговое структурированное саммари строго по инструкции. "
            "Не добавляй факты, которых нет в промежуточных саммари.\n\n"
            f"{final_input}"
        ),
        base_url=base_url,
        model=model,
        api_key=api_key,
        prompt=prompt,
        temperature=temperature,
        timeout=timeout,
        retries=retries,
        retry_delay=retry_delay,
    )
