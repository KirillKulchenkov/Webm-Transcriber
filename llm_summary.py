from __future__ import annotations

import argparse
import json
import os
import socket
import time
from dataclasses import dataclass
from http.client import IncompleteRead, RemoteDisconnected
from pathlib import Path
from typing import Any, Literal
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

DEFAULT_LECTURE_PROMPT = (
    "Ты мой AI-методист по анализу образовательных стенограмм. "
    "Ниже будет текст лекции или воркшопа. Подготовь подробный и практичный "
    "конспект на русском языке. Структура ответа:\n"
    "1. Краткий конспект лекции: 8-15 тезисов в порядке изложения.\n"
    "2. Тема лекции и контекст (о чем лекция и для кого).\n"
    "3. Подробный план и основные разделы (в порядке изложения).\n"
    "4. Ключевые понятия, определения, формулы/правила (если есть).\n"
    "5. Примеры, кейсы, демонстрации и разборы, которые приводил спикер.\n"
    "6. Практические шаги, рекомендации и чек-листы для применения.\n"
    "7. Вопросы аудитории и ответы спикера (если были).\n"
    "8. Итоговые выводы и что стоит изучить дальше.\n"
    "Не придумывай факты, которых нет в тексте стенограммы."
)

DEFAULT_DEMO_PROMPT = (
    "Ты мой AI-ассистент для подготовки отчета по scrum-демо. "
    "Ниже будет стенограмма встречи, где команда показывает итоги разработки. "
    "Сделай структурированный отчет на русском языке. Структура ответа:\n"
    "1. Краткий конспект демо: 6-12 тезисов с главными показанными результатами.\n"
    "2. Что было продемонстрировано: фичи, улучшения, исправления (по пунктам).\n"
    "3. Статус по целям спринта: что завершено, что частично готово, что перенесено.\n"
    "4. Решения и договоренности команды по итогам демо.\n"
    "5. Обратная связь, вопросы и замечания от участников (если были).\n"
    "6. Риски, блокеры и технические долги, озвученные на демо.\n"
    "7. Следующие шаги: приоритеты, владельцы и ближайшие действия (если можно определить).\n"
    "Не придумывай факты, которых нет в тексте стенограммы."
)

DEFAULT_SUMMARY_CHUNK_CHARS = int(os.getenv("SUMMARY_CHUNK_CHARS", "32000"))
DEFAULT_SUMMARY_CHUNK_OVERLAP_CHARS = int(
    os.getenv("SUMMARY_CHUNK_OVERLAP_CHARS", "3000")
)

SummaryMode = Literal["summary", "lecture", "demo"]


@dataclass(frozen=True)
class GenerationProfile:
    mode: SummaryMode
    display_name: str
    default_prompt: str
    output_suffix: str
    single_user_prefix: str
    chunk_user_template: str
    reduce_system_prompt: str
    reduce_user_prefix: str
    final_user_prefix: str
    intermediate_label: str
    empty_input_error: str


SUMMARY_PROFILE = GenerationProfile(
    mode="summary",
    display_name="Саммари",
    default_prompt=DEFAULT_SUMMARY_PROMPT,
    output_suffix=".summary.md",
    single_user_prefix=(
        "Ниже стенограмма разговора. "
        "Подготовь структурированное саммари по инструкции."
    ),
    chunk_user_template=(
        "Ниже часть стенограммы разговора. "
        "Это часть {idx} из {total}. "
        "Сделай структурированное саммари ТОЛЬКО по этой части "
        "и не придумывай факты вне текста.\n\n{chunk}"
    ),
    reduce_system_prompt=(
        "Ты объединяешь промежуточные саммари одной и той же встречи. "
        "Сохраняй факты, убирай дубли, не добавляй новых деталей."
    ),
    reduce_user_prefix=(
        "Ниже несколько промежуточных саммари частей одной встречи. "
        "Объедини их в единое краткое саммари без повторов и без новых фактов."
    ),
    final_user_prefix=(
        "Ниже промежуточные саммари частей одной и той же встречи. "
        "Собери единое итоговое структурированное саммари строго по инструкции. "
        "Не добавляй факты, которых нет в промежуточных саммари."
    ),
    intermediate_label="Промежуточное саммари",
    empty_input_error="Пустая стенограмма для саммари.",
)

LECTURE_PROFILE = GenerationProfile(
    mode="lecture",
    display_name="Конспект",
    default_prompt=DEFAULT_LECTURE_PROMPT,
    output_suffix=".lecture.md",
    single_user_prefix=(
        "Ниже стенограмма лекции или воркшопа. "
        "Подготовь подробный конспект по инструкции."
    ),
    chunk_user_template=(
        "Ниже часть стенограммы лекции/воркшопа. "
        "Это часть {idx} из {total}. "
        "Сделай подробный конспект ТОЛЬКО по этой части "
        "и не придумывай факты вне текста.\n\n{chunk}"
    ),
    reduce_system_prompt=(
        "Ты объединяешь промежуточные конспекты одной и той же лекции. "
        "Сохраняй факты, убирай дубли и не добавляй новых деталей."
    ),
    reduce_user_prefix=(
        "Ниже несколько промежуточных конспектов частей одной лекции/воркшопа. "
        "Объедини их в единый цельный конспект без повторов и без новых фактов."
    ),
    final_user_prefix=(
        "Ниже промежуточные конспекты частей одной и той же лекции/воркшопа. "
        "Собери единый подробный итоговый конспект строго по инструкции. "
        "Не добавляй факты, которых нет в промежуточных конспектах."
    ),
    intermediate_label="Промежуточный конспект",
    empty_input_error="Пустая стенограмма для конспекта лекции.",
)

DEMO_PROFILE = GenerationProfile(
    mode="demo",
    display_name="Демо-отчет",
    default_prompt=DEFAULT_DEMO_PROMPT,
    output_suffix=".demo.md",
    single_user_prefix=(
        "Ниже стенограмма scrum-демо с показом итогов разработки. "
        "Подготовь структурированный отчет по инструкции."
    ),
    chunk_user_template=(
        "Ниже часть стенограммы scrum-демо. "
        "Это часть {idx} из {total}. "
        "Сделай структурированный отчет ТОЛЬКО по этой части "
        "и не придумывай факты вне текста.\n\n{chunk}"
    ),
    reduce_system_prompt=(
        "Ты объединяешь промежуточные отчеты по одному и тому же scrum-демо. "
        "Сохраняй факты, убирай дубли и не добавляй новых деталей."
    ),
    reduce_user_prefix=(
        "Ниже несколько промежуточных отчетов частей одного scrum-демо. "
        "Объедини их в единый цельный отчет без повторов и без новых фактов."
    ),
    final_user_prefix=(
        "Ниже промежуточные отчеты частей одного и того же scrum-демо. "
        "Собери единый итоговый отчет строго по инструкции. "
        "Не добавляй факты, которых нет в промежуточных отчетах."
    ),
    intermediate_label="Промежуточный отчет",
    empty_input_error="Пустая стенограмма для отчета по demo-встрече.",
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
            help="После транскрибации дополнительно сгенерировать текст через OpenAI-compatible API (LM Studio).",
        )
    parser.add_argument(
        "--summary-mode",
        choices=("summary", "lecture", "demo"),
        default=normalize_summary_mode(os.getenv("SUMMARY_MODE", "summary")),
        help=(
            "Режим генерации: summary (итоги встречи, по умолчанию) "
            "или lecture (подробный конспект лекции/воркшопа), "
            "или demo (отчет по scrum-демо итогов разработки)."
        ),
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
        help="Temperature для генерации текста (по умолчанию: 0.1)",
    )
    parser.add_argument(
        "--summary-timeout",
        type=int,
        default=300,
        help="Таймаут HTTP-запроса в секундах (по умолчанию: 300)",
    )
    parser.add_argument(
        "--summary-retries",
        type=int,
        default=3,
        help="Количество попыток запроса (по умолчанию: 3)",
    )
    parser.add_argument(
        "--summary-retry-delay",
        type=float,
        default=2.0,
        help="Пауза между попытками в секундах (по умолчанию: 2.0)",
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
        help=(
            "Куда сохранить результат. По умолчанию: *.summary.md для режима summary "
            "и *.lecture.md для режима lecture, и *.demo.md для режима demo."
        ),
    )
    parser.add_argument(
        "--summary-prompt",
        default=None,
        help=(
            "Кастомный system prompt. По умолчанию берется встроенный промпт "
            "в зависимости от --summary-mode."
        ),
    )


def normalize_summary_mode(mode: str) -> SummaryMode:
    normalized = mode.strip().lower()
    if normalized == "lecture":
        return "lecture"
    if normalized == "demo":
        return "demo"
    return "summary"


def _profile_for_mode(summary_mode: SummaryMode) -> GenerationProfile:
    if summary_mode == "lecture":
        return LECTURE_PROFILE
    if summary_mode == "demo":
        return DEMO_PROFILE
    return SUMMARY_PROFILE


def resolve_summary_prompt(
    summary_mode: SummaryMode,
    prompt_override: str | None,
) -> str:
    custom = (prompt_override or "").strip()
    if custom:
        return custom
    return _profile_for_mode(summary_mode).default_prompt


def default_mode_output_path(
    transcript_txt_path: Path,
    summary_mode: SummaryMode,
) -> Path:
    profile = _profile_for_mode(summary_mode)
    return transcript_txt_path.with_suffix(profile.output_suffix)


def default_summary_output_path(transcript_txt_path: Path) -> Path:
    return default_mode_output_path(transcript_txt_path, "summary")


def default_lecture_output_path(transcript_txt_path: Path) -> Path:
    return default_mode_output_path(transcript_txt_path, "lecture")


def default_demo_output_path(transcript_txt_path: Path) -> Path:
    return default_mode_output_path(transcript_txt_path, "demo")


def mode_output_label(summary_mode: SummaryMode) -> str:
    if summary_mode == "lecture":
        return "LECTURE"
    if summary_mode == "demo":
        return "DEMO"
    return "SUMMARY"


def save_generated_text(generated_text: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(generated_text.strip() + "\n", encoding="utf-8")
    return output_path


def save_summary(summary_text: str, output_path: Path) -> Path:
    return save_generated_text(summary_text, output_path)


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


def _group_blocks_for_reduce(
    blocks: list[str],
    *,
    max_chars: int,
    entry_label: str,
) -> list[str]:
    groups: list[str] = []
    current: list[str] = []
    current_len = 0

    for idx, block in enumerate(blocks, start=1):
        entry = f"{entry_label} {idx}:\n{block.strip()}\n"
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


def _request_single_generation(
    user_content: str,
    *,
    base_url: str,
    model: str,
    api_key: str | None,
    prompt: str,
    temperature: float,
    timeout: int,
    retries: int,
    retry_delay: float,
    mode_label: str,
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
            generated = _extract_content(parsed)
            if not generated:
                raise RuntimeError("LLM вернул пустой результат.")
            return generated
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="ignore")
            retryable = exc.code >= 500 or exc.code in {408, 409, 429}
            last_error = RuntimeError(
                f"Ошибка LLM API ({exc.code}): {error_body or exc.reason}"
            )
            if not retryable or attempt >= attempts:
                raise last_error from exc
        except URLError as exc:
            raw_reason = exc.reason
            if isinstance(raw_reason, BaseException):
                reason_text = f"{type(raw_reason).__name__}: {raw_reason}"
            else:
                reason_text = str(raw_reason)
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                last_error = RuntimeError(
                    "Таймаут ожидания ответа от LLM API. "
                    "Увеличьте --summary-timeout или уменьшите размер стенограммы. "
                    f"Причина: {reason_text}"
                )
            else:
                last_error = RuntimeError(
                    "Не удалось подключиться к LLM API. "
                    "Проверьте URL, API key и сетевой доступ. "
                    f"Причина: {reason_text}"
                )
            if attempt >= attempts:
                raise last_error from exc
        except (IncompleteRead, RemoteDisconnected) as exc:
            last_error = RuntimeError(
                "Соединение с LLM API было прервано до получения полного ответа. "
                f"Причина: {type(exc).__name__}: {exc}"
            )
            if attempt >= attempts:
                raise last_error from exc
        except (TimeoutError, socket.timeout) as exc:
            last_error = RuntimeError(
                "Таймаут ожидания ответа от LLM API. "
                "Увеличьте --summary-timeout или уменьшите размер стенограммы. "
                f"Причина: {type(exc).__name__}: {exc}"
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
        error_hint = str(last_error) if last_error else "неизвестная ошибка"
        print(
            f"[warn] {mode_label}: попытка {attempt}/{attempts} не удалась: {error_hint}. "
            f"Повтор через {sleep_seconds:.1f}с..."
        )
        time.sleep(sleep_seconds)

    if last_error is not None:
        raise last_error
    raise RuntimeError("Не удалось получить результат по неизвестной причине.")


def request_summary_mode(
    transcript_text: str,
    *,
    summary_mode: SummaryMode,
    base_url: str,
    model: str,
    api_key: str | None,
    prompt: str | None,
    temperature: float,
    timeout: int,
    retries: int = 3,
    retry_delay: float = 2.0,
    chunk_chars: int = DEFAULT_SUMMARY_CHUNK_CHARS,
    chunk_overlap_chars: int = DEFAULT_SUMMARY_CHUNK_OVERLAP_CHARS,
) -> str:
    profile = _profile_for_mode(summary_mode)
    effective_prompt = resolve_summary_prompt(summary_mode, prompt)

    normalized = transcript_text.strip()
    if not normalized:
        raise RuntimeError(profile.empty_input_error)

    safe_chunk_chars = max(1000, chunk_chars)
    safe_overlap = max(0, chunk_overlap_chars)
    chunks = _split_text_chunks(
        normalized,
        max_chars=safe_chunk_chars,
        overlap_chars=safe_overlap,
    )

    if len(chunks) <= 1:
        return _request_single_generation(
            user_content=(
                f"{profile.single_user_prefix}\n\n"
                f"{normalized}"
            ),
            base_url=base_url,
            model=model,
            api_key=api_key,
            prompt=effective_prompt,
            temperature=temperature,
            timeout=timeout,
            retries=retries,
            retry_delay=retry_delay,
            mode_label=profile.display_name,
        )

    print(
        f"[info] {profile.display_name}: длинная стенограмма, делю на чанки: "
        f"{len(chunks)} (chunk_chars={safe_chunk_chars}, overlap={safe_overlap})"
    )
    partial_outputs: list[str] = []

    for idx, chunk in enumerate(chunks, start=1):
        print(f"[info] {profile.display_name}: обрабатываю чанк {idx}/{len(chunks)}...")
        partial = _request_single_generation(
            user_content=profile.chunk_user_template.format(
                idx=idx,
                total=len(chunks),
                chunk=chunk,
            ),
            base_url=base_url,
            model=model,
            api_key=api_key,
            prompt=effective_prompt,
            temperature=temperature,
            timeout=timeout,
            retries=retries,
            retry_delay=retry_delay,
            mode_label=profile.display_name,
        )
        partial_outputs.append(partial)

    reduce_round = 1
    current = [item.strip() for item in partial_outputs if item.strip()]

    while len(current) > 1:
        grouped = _group_blocks_for_reduce(
            current,
            max_chars=safe_chunk_chars,
            entry_label=profile.intermediate_label,
        )
        if len(grouped) == 1:
            break

        print(
            f"[info] {profile.display_name}: reduce-раунд {reduce_round}, "
            f"групп для объединения: {len(grouped)}"
        )
        next_level: list[str] = []
        for group_idx, group_text in enumerate(grouped, start=1):
            print(
                f"[info] {profile.display_name}: агрегирую группу "
                f"{group_idx}/{len(grouped)}..."
            )
            reduced = _request_single_generation(
                user_content=(
                    f"{profile.reduce_user_prefix}\n\n"
                    f"{group_text}"
                ),
                base_url=base_url,
                model=model,
                api_key=api_key,
                prompt=profile.reduce_system_prompt,
                temperature=temperature,
                timeout=timeout,
                retries=retries,
                retry_delay=retry_delay,
                mode_label=profile.display_name,
            )
            next_level.append(reduced)
        current = next_level
        reduce_round += 1

    final_input = _format_blocks(current, profile.intermediate_label)
    return _request_single_generation(
        user_content=(
            f"{profile.final_user_prefix}\n\n"
            f"{final_input}"
        ),
        base_url=base_url,
        model=model,
        api_key=api_key,
        prompt=effective_prompt,
        temperature=temperature,
        timeout=timeout,
        retries=retries,
        retry_delay=retry_delay,
        mode_label=profile.display_name,
    )


def request_summary(
    transcript_text: str,
    *,
    base_url: str,
    model: str,
    api_key: str | None,
    prompt: str | None,
    temperature: float,
    timeout: int,
    retries: int = 3,
    retry_delay: float = 2.0,
    chunk_chars: int = DEFAULT_SUMMARY_CHUNK_CHARS,
    chunk_overlap_chars: int = DEFAULT_SUMMARY_CHUNK_OVERLAP_CHARS,
) -> str:
    return request_summary_mode(
        transcript_text,
        summary_mode="summary",
        base_url=base_url,
        model=model,
        api_key=api_key,
        prompt=prompt,
        temperature=temperature,
        timeout=timeout,
        retries=retries,
        retry_delay=retry_delay,
        chunk_chars=chunk_chars,
        chunk_overlap_chars=chunk_overlap_chars,
    )


def request_lecture_description(
    transcript_text: str,
    *,
    base_url: str,
    model: str,
    api_key: str | None,
    prompt: str | None,
    temperature: float,
    timeout: int,
    retries: int = 3,
    retry_delay: float = 2.0,
    chunk_chars: int = DEFAULT_SUMMARY_CHUNK_CHARS,
    chunk_overlap_chars: int = DEFAULT_SUMMARY_CHUNK_OVERLAP_CHARS,
) -> str:
    return request_summary_mode(
        transcript_text,
        summary_mode="lecture",
        base_url=base_url,
        model=model,
        api_key=api_key,
        prompt=prompt,
        temperature=temperature,
        timeout=timeout,
        retries=retries,
        retry_delay=retry_delay,
        chunk_chars=chunk_chars,
        chunk_overlap_chars=chunk_overlap_chars,
    )


def request_demo_report(
    transcript_text: str,
    *,
    base_url: str,
    model: str,
    api_key: str | None,
    prompt: str | None,
    temperature: float,
    timeout: int,
    retries: int = 3,
    retry_delay: float = 2.0,
    chunk_chars: int = DEFAULT_SUMMARY_CHUNK_CHARS,
    chunk_overlap_chars: int = DEFAULT_SUMMARY_CHUNK_OVERLAP_CHARS,
) -> str:
    return request_summary_mode(
        transcript_text,
        summary_mode="demo",
        base_url=base_url,
        model=model,
        api_key=api_key,
        prompt=prompt,
        temperature=temperature,
        timeout=timeout,
        retries=retries,
        retry_delay=retry_delay,
        chunk_chars=chunk_chars,
        chunk_overlap_chars=chunk_overlap_chars,
    )
