#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import sys
import warnings
from pathlib import Path
from typing import Any, Literal

from llm_summary import (
    add_summary_args,
    default_summary_output_path,
    request_summary,
    save_summary,
)

TRANSCRIBE_TASK = "transcribe"
DEFAULT_WHISPERX_MODEL = "large-v3"
DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"

DevicePreference = Literal["auto", "cuda", "cpu"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Транскрибация и diarization через WhisperX "
            "(вывод с SPEAKER_00/01/...)."
        )
    )
    parser.add_argument("input", type=Path, help="Путь к видео/аудио файлу (например .webm)")
    parser.add_argument(
        "--model",
        default=DEFAULT_WHISPERX_MODEL,
        help=f"WhisperX ASR модель (по умолчанию: {DEFAULT_WHISPERX_MODEL})",
    )
    parser.add_argument(
        "--language",
        default="ru",
        help="Код языка (по умолчанию: ru)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size для ASR (по умолчанию: 8)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=30,
        help="Размер чанка VAD/ASR в секундах (по умолчанию: 30)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Устройство: auto (по умолчанию), cuda или cpu",
    )
    parser.add_argument(
        "--compute-type",
        choices=("auto", "float16", "float32", "int8"),
        default="auto",
        help="Тип вычислений faster-whisper (по умолчанию: auto)",
    )
    parser.add_argument(
        "--diarization-model",
        default=DEFAULT_DIARIZATION_MODEL,
        help=(
            "HF модель diarization "
            f"(по умолчанию: {DEFAULT_DIARIZATION_MODEL})"
        ),
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Точное число спикеров (если известно)",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=None,
        help="Минимальное число спикеров",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Максимальное число спикеров",
    )
    parser.add_argument(
        "--skip-align",
        action="store_true",
        help="Пропустить forced alignment (быстрее, но хуже привязка слов)",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help=(
            "Токен Hugging Face Hub. Если не задан, берется из "
            "HF_TOKEN или HUGGINGFACE_HUB_TOKEN."
        ),
    )
    parser.add_argument(
        "--output-txt",
        type=Path,
        default=None,
        help="Куда сохранить текст со спикерами (по умолчанию: рядом с входным файлом)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Куда сохранить JSON с сегментами/спикерами (по умолчанию: рядом с входным файлом)",
    )
    add_summary_args(parser)
    return parser


def ensure_prerequisites(input_path: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Файл не найден: {input_path}")
    if shutil.which("ffmpeg") is None:
        raise EnvironmentError(
            "Не найден ffmpeg. Установите ffmpeg и повторите запуск."
        )
    if not has_module("whisperx"):
        raise RuntimeError(
            "Не найден модуль whisperx.\n"
            "Установите зависимости: `uv sync --extra whisperx`."
        )


def has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def resolve_hf_token(cli_hf_token: str | None) -> str | None:
    if cli_hf_token:
        return cli_hf_token
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")


def configure_hf_auth(hf_token: str | None) -> None:
    if not hf_token:
        return
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token


def resolve_device(requested: DevicePreference) -> str:
    import torch

    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Запрошен --device cuda, но CUDA недоступна. "
                "На Apple Silicon обычно используем --device cpu."
            )
        return "cuda"

    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_compute_type(requested: str, device: str) -> str:
    if requested != "auto":
        return requested
    if device == "cuda":
        return "float16"
    return "int8"


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
        speaker = str(seg.get("speaker", "SPEAKER_UNKNOWN"))
        start = format_timestamp(seg.get("start"))
        end = format_timestamp(seg.get("end"))
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        lines.append(f"[{start} - {end}] {speaker}: {text}")
    return "\n".join(lines).strip() + "\n"


def sanitize_for_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(v) for v in value]
    # numpy / torch scalar types
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            pass
    return str(value)


def run_whisperx_pipeline(
    input_path: Path,
    model_name: str,
    language: str,
    batch_size: int,
    chunk_size: int,
    device: str,
    compute_type: str,
    diarization_model: str,
    hf_token: str | None,
    num_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
    skip_align: bool,
) -> dict[str, Any]:
    # Known pyannote warning is very verbose and non-fatal for our pipeline.
    warnings.filterwarnings(
        "ignore",
        message=".*torchcodec is not installed correctly.*",
        category=UserWarning,
    )

    import whisperx
    from whisperx.diarize import DiarizationPipeline, assign_word_speakers

    print(
        f"[info] whisperx model={model_name}, device={device}, "
        f"compute_type={compute_type}, language={language}"
    )

    audio = whisperx.load_audio(str(input_path))

    asr_model = whisperx.load_model(
        whisper_arch=model_name,
        device=device,
        compute_type=compute_type,
        language=language,
        task=TRANSCRIBE_TASK,
        use_auth_token=hf_token,
    )

    transcription = asr_model.transcribe(
        audio,
        batch_size=batch_size,
        language=language,
        task=TRANSCRIBE_TASK,
        chunk_size=chunk_size,
    )

    result_for_diarization: dict[str, Any] = transcription

    if skip_align:
        print("[warn] Forced alignment пропущен (--skip-align).")
    else:
        try:
            align_model, align_metadata = whisperx.load_align_model(
                language_code=str(transcription.get("language") or language),
                device=device,
            )
            result_for_diarization = whisperx.align(
                transcription["segments"],
                align_model,
                align_metadata,
                audio,
                device,
                return_char_alignments=False,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                "[warn] Не удалось выполнить alignment, продолжаем без него: "
                f"{exc}"
            )
            result_for_diarization = transcription

    print(f"[info] diarization model={diarization_model}")
    diarize_pipeline = DiarizationPipeline(
        model_name=diarization_model,
        token=hf_token,
        device=device,
    )
    diarize_df = diarize_pipeline(
        audio,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    speaker_result = assign_word_speakers(diarize_df, result_for_diarization)

    segments: list[dict[str, Any]] = speaker_result.get("segments", [])
    speakers_detected = sorted(
        {str(seg.get("speaker")) for seg in segments if seg.get("speaker") is not None}
    )

    diarize_rows = diarize_df.to_dict("records")
    for row in diarize_rows:
        row.pop("segment", None)

    return {
        "text": str(speaker_result.get("text", "")).strip(),
        "segments": segments,
        "word_segments": speaker_result.get("word_segments", []),
        "language": speaker_result.get("language", transcription.get("language")),
        "metadata": {
            "backend": "whisperx",
            "task": TRANSCRIBE_TASK,
            "asr_model": model_name,
            "diarization_model": diarization_model,
            "device": device,
            "compute_type": compute_type,
            "batch_size": batch_size,
            "chunk_size": chunk_size,
            "skip_align": skip_align,
            "num_speakers": num_speakers,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
            "speakers_detected": speakers_detected,
            "platform": {
                "system": platform.system(),
                "machine": platform.machine(),
            },
        },
        "diarization_segments": diarize_rows,
    }


def save_outputs(
    result: dict[str, Any],
    input_path: Path,
    output_txt: Path | None,
    output_json: Path | None,
) -> tuple[Path, Path]:
    txt_path = output_txt or input_path.with_suffix(".speakers.txt")
    json_path = output_json or input_path.with_suffix(".speakers.json")

    txt_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    speaker_text = to_speaker_text(result.get("segments", []))
    txt_path.write_text(speaker_text, encoding="utf-8")

    json_path.write_text(
        json.dumps(sanitize_for_json(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return txt_path, json_path


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    summary_path: Path | None = None

    try:
        ensure_prerequisites(args.input)
        hf_token = resolve_hf_token(args.hf_token)
        configure_hf_auth(hf_token)
        if not hf_token:
            print(
                "[warn] HF токен не задан. Для diarization на pyannote обычно нужен токен "
                "и принятие условий модели на Hugging Face."
            )

        device = resolve_device(args.device)
        compute_type = resolve_compute_type(args.compute_type, device)
        print(f"[info] hf_auth={'on' if hf_token else 'off'}")

        result = run_whisperx_pipeline(
            input_path=args.input,
            model_name=args.model,
            language=args.language,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
            device=device,
            compute_type=compute_type,
            diarization_model=args.diarization_model,
            hf_token=hf_token,
            num_speakers=args.num_speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            skip_align=args.skip_align,
        )
        txt_path, json_path = save_outputs(
            result=result,
            input_path=args.input,
            output_txt=args.output_txt,
            output_json=args.output_json,
        )

        if args.summarize:
            print("[info] Запускаю LLM-саммари...")
            speaker_text = to_speaker_text(result.get("segments", []))
            transcript_for_summary = speaker_text.strip() or result.get("text", "").strip()
            summary_text = request_summary(
                transcript_text=transcript_for_summary,
                base_url=args.summary_base_url,
                model=args.summary_model,
                api_key=args.summary_api_key,
                prompt=args.summary_prompt,
                temperature=args.summary_temperature,
                timeout=args.summary_timeout,
                retries=args.summary_retries,
                retry_delay=args.summary_retry_delay,
            )
            summary_path = save_summary(
                summary_text,
                args.summary_output or default_summary_output_path(txt_path),
            )
    except Exception as exc:  # noqa: BLE001
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    speakers = result.get("metadata", {}).get("speakers_detected", [])
    print("[ok] WhisperX diarization завершена.")
    print(f"[ok] Найденные спикеры: {', '.join(speakers) if speakers else 'нет данных'}")
    print(f"[ok] TXT:  {txt_path}")
    print(f"[ok] JSON: {json_path}")
    if summary_path:
        print(f"[ok] SUMMARY: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
