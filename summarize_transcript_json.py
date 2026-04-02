#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Literal

from llm_summary import (
    add_summary_args,
    default_mode_output_path,
    mode_output_label,
    request_summary_mode,
    save_summary,
)

SpeakerFormat = Literal["auto", "always", "never"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Сделать LLM-генерацию (саммари/конспект/демо-отчет) "
            "по уже готовому JSON транскрибации "
            "(transcribe_webm/transcribe_whisperx)."
        )
    )
    parser.add_argument(
        "input_json",
        type=Path,
        help="Путь к JSON файлу транскрибации (.json / .speakers.json)",
    )
    parser.add_argument(
        "--speaker-format",
        choices=("auto", "always", "never"),
        default="auto",
        help=(
            "Формат входа для генерации: auto (по умолчанию), always "
            "(принудительно по спикерам), never (только plain text)."
        ),
    )
    add_summary_args(parser, include_toggle=False)
    return parser


def format_timestamp(seconds: float | int | None) -> str:
    if seconds is None:
        return "--:--:--.---"
    total_ms = int(float(seconds) * 1000)
    h = total_ms // 3_600_000
    total_ms %= 3_600_000
    m = total_ms // 60_000
    total_ms %= 60_000
    s = total_ms // 1000
    ms = total_ms % 1000
    return f"{h:02}:{m:02}:{s:02}.{ms:03}"


def to_speaker_text(segments: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        speaker_id = str(seg.get("speaker", "SPEAKER_UNKNOWN"))
        speaker_name = str(seg.get("speaker_name", "")).strip()
        speaker = f"{speaker_name} [{speaker_id}]" if speaker_name else speaker_id
        start = format_timestamp(seg.get("start"))
        end = format_timestamp(seg.get("end"))
        lines.append(f"[{start} - {end}] {speaker}: {text}")
    return "\n".join(lines).strip()


def _extract_chunks_text(chunks: list[Any]) -> str:
    lines: list[str] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        text = str(chunk.get("text", "")).strip()
        if text:
            lines.append(text)
    return "\n".join(lines).strip()


def _extract_segments_text(segments: list[Any]) -> str:
    lines: list[str] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        text = str(segment.get("text", "")).strip()
        if text:
            lines.append(text)
    return "\n".join(lines).strip()


def extract_transcript_text(
    payload: dict[str, Any],
    *,
    speaker_format: SpeakerFormat,
) -> tuple[str, str]:
    raw_text = str(payload.get("text", "")).strip()
    raw_segments = payload.get("segments", [])
    raw_chunks = payload.get("chunks", [])

    segments = raw_segments if isinstance(raw_segments, list) else []
    chunks = raw_chunks if isinstance(raw_chunks, list) else []

    has_speaker_labels = any(
        isinstance(seg, dict) and seg.get("speaker") is not None for seg in segments
    )

    use_speaker_format = (
        speaker_format == "always" or (speaker_format == "auto" and has_speaker_labels)
    )

    if use_speaker_format:
        speaker_text = to_speaker_text([seg for seg in segments if isinstance(seg, dict)])
        if speaker_text:
            return speaker_text, "speaker_segments"
        if speaker_format == "always":
            raise RuntimeError(
                "Запрошен --speaker-format always, но в JSON нет валидных speaker-сегментов."
            )

    if raw_text:
        return raw_text, "text_field"

    chunks_text = _extract_chunks_text(chunks)
    if chunks_text:
        return chunks_text, "chunks_text"

    segments_text = _extract_segments_text(segments)
    if segments_text:
        return segments_text, "segments_text"

    raise RuntimeError(
        "Не удалось извлечь текст из JSON. Ожидается поле `text`, `chunks[].text` или `segments[].text`."
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.input_json.exists():
        print(f"[error] Файл не найден: {args.input_json}", file=sys.stderr)
        return 1

    if args.input_json.suffix.lower() != ".json":
        print("[error] Ожидается JSON файл (.json).", file=sys.stderr)
        return 1

    try:
        data = json.loads(args.input_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"[error] Некорректный JSON: {exc}", file=sys.stderr)
        return 1

    if not isinstance(data, dict):
        print("[error] Корневой объект JSON должен быть объектом.", file=sys.stderr)
        return 1

    try:
        transcript_text, source = extract_transcript_text(
            data,
            speaker_format=args.speaker_format,
        )
        print(
            f"[info] Источник текста для генерации: {source}; "
            f"mode={args.summary_mode}"
        )
        summary_text = request_summary_mode(
            transcript_text=transcript_text,
            summary_mode=args.summary_mode,
            base_url=args.summary_base_url,
            model=args.summary_model,
            api_key=args.summary_api_key,
            prompt=args.summary_prompt,
            temperature=args.summary_temperature,
            timeout=args.summary_timeout,
            retries=args.summary_retries,
            retry_delay=args.summary_retry_delay,
            chunk_chars=args.summary_chunk_chars,
            chunk_overlap_chars=args.summary_chunk_overlap_chars,
        )
        output_path = save_summary(
            summary_text,
            args.summary_output
            or default_mode_output_path(args.input_json, args.summary_mode),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    print(f"[ok] {mode_output_label(args.summary_mode)}: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
