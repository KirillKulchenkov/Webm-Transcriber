#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
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

DEFAULT_HF_MODEL_ID = "openai/whisper-large-v3"
DEFAULT_MLX_MODEL_ID = "mlx-community/whisper-large-v3-mlx"
TRANSCRIBE_LANGUAGE = "russian"

Backend = Literal["auto", "hf", "mlx"]
DevicePreference = Literal["auto", "cuda", "cpu"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Локальная транскрибация .webm через Whisper Large "
            "(auto: MLX на Apple Silicon, HF на остальных системах)."
        )
    )
    parser.add_argument("input", type=Path, help="Путь к .webm файлу встречи")
    parser.add_argument(
        "--backend",
        choices=("auto", "hf", "mlx"),
        default="auto",
        help=(
            "Бэкенд транскрибации: auto (по умолчанию), hf (transformers), "
            "mlx (Apple Silicon)"
        ),
    )
    parser.add_argument(
        "--hf-model-id",
        "--model-id",
        dest="hf_model_id",
        default=DEFAULT_HF_MODEL_ID,
        help=f"HF модель Whisper (по умолчанию: {DEFAULT_HF_MODEL_ID})",
    )
    parser.add_argument(
        "--mlx-model-id",
        default=DEFAULT_MLX_MODEL_ID,
        help=f"MLX модель Whisper (по умолчанию: {DEFAULT_MLX_MODEL_ID})",
    )
    parser.add_argument(
        "--chunk-length",
        type=int,
        default=30,
        help="Длина чанка в секундах для длинных записей (по умолчанию: 30)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size для инференса (по умолчанию: 8)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help=(
            "Устройство для HF-бэкенда: auto (cuda при доступности, иначе cpu), "
            "cuda (ошибка, если CUDA недоступна), cpu"
        ),
    )
    parser.add_argument(
        "--output-txt",
        type=Path,
        default=None,
        help="Куда сохранить текст транскрибации (по умолчанию: рядом с входным файлом)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Куда сохранить JSON c чанками и таймкодами (по умолчанию: рядом с входным файлом)",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help=(
            "Токен Hugging Face Hub. Если не задан, берется из "
            "HF_TOKEN или HUGGINGFACE_HUB_TOKEN."
        ),
    )
    add_summary_args(parser)
    return parser


def ensure_dependencies(input_path: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Файл не найден: {input_path}")
    if input_path.suffix.lower() != ".webm":
        raise ValueError("Ожидается файл с расширением .webm")
    if shutil.which("ffmpeg") is None:
        raise EnvironmentError(
            "Не найден ffmpeg. Установите ffmpeg и повторите запуск."
        )


def is_apple_silicon_mac() -> bool:
    return platform.system() == "Darwin" and platform.machine().lower() in {
        "arm64",
        "aarch64",
    }


def has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def resolve_backend(requested_backend: Backend) -> Literal["hf", "mlx"]:
    if requested_backend == "hf":
        return "hf"
    if requested_backend == "mlx":
        return "mlx"

    if is_apple_silicon_mac():
        if has_module("mlx_whisper"):
            return "mlx"
        if has_module("torch") and has_module("transformers"):
            return "hf"
    else:
        if has_module("torch") and has_module("transformers"):
            return "hf"
        if has_module("mlx_whisper"):
            return "mlx"

    raise RuntimeError(
        "Не найден подходящий бэкенд ASR.\n"
        "Запустите `uv sync` для платформенных зависимостей.\n"
        "Если на Apple Silicon нужен HF-бэкенд, используйте `uv sync --extra hf`."
    )


def resolve_hf_token(cli_hf_token: str | None) -> str | None:
    if cli_hf_token:
        return cli_hf_token
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")


def configure_hf_auth(hf_token: str | None) -> None:
    if not hf_token:
        return
    # Both env names are supported by HF ecosystem in different tools.
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token


def collect_torch_diagnostics() -> dict[str, Any]:
    import torch

    cuda_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count() if cuda_available else 0
    first_cuda_device_name = (
        torch.cuda.get_device_name(0) if cuda_available and cuda_device_count > 0 else None
    )

    cudnn_available: bool | None
    try:
        cudnn_available = bool(torch.backends.cudnn.is_available())
    except Exception:  # noqa: BLE001
        cudnn_available = None

    return {
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "first_cuda_device_name": first_cuda_device_name,
        "cudnn_available": cudnn_available,
    }


def format_cuda_unavailable_hint(diagnostics: dict[str, Any]) -> str:
    details = [
        f"torch={diagnostics['torch_version']}",
        f"torch.version.cuda={diagnostics['torch_cuda_build']}",
        f"cuda_available={diagnostics['cuda_available']}",
        f"cuda_device_count={diagnostics['cuda_device_count']}",
    ]

    if diagnostics["platform_system"] == "Windows":
        if diagnostics["torch_cuda_build"] is None:
            details.append(
                "Похоже установлен CPU-only PyTorch. Для Windows нужен CUDA wheel."
            )
            details.append(
                "PowerShell: uv pip install --index-url https://download.pytorch.org/whl/cu128 torch"
            )
        else:
            details.append(
                "CUDA build PyTorch найден, но CUDA недоступна в рантайме. "
                "Проверьте NVIDIA driver и совместимость версии CUDA."
            )

    return "; ".join(details)


def choose_device(
    requested_device: DevicePreference,
) -> tuple[int, Any, str, dict[str, Any]]:
    import torch

    diagnostics = collect_torch_diagnostics()

    if requested_device == "cpu":
        return -1, torch.float32, "cpu", diagnostics

    if diagnostics["cuda_available"]:
        return 0, torch.float16, "cuda", diagnostics

    if requested_device == "cuda":
        raise RuntimeError(
            "Запрошен --device cuda, но CUDA недоступна.\n"
            f"{format_cuda_unavailable_hint(diagnostics)}"
        )

    return -1, torch.float32, "cpu", diagnostics


def run_transcription(
    input_path: Path,
    model_id: str,
    chunk_length: int,
    batch_size: int,
    hf_token: str | None,
    requested_device: DevicePreference,
) -> dict[str, Any]:
    from transformers import pipeline

    device, torch_dtype, device_name, diagnostics = choose_device(requested_device)
    print(
        f"[info] device={device_name}, requested_device={requested_device}, "
        f"dtype={torch_dtype}, model={model_id}"
    )
    print(
        "[info] "
        f"torch={diagnostics['torch_version']}, "
        f"torch.version.cuda={diagnostics['torch_cuda_build']}, "
        f"cuda_available={diagnostics['cuda_available']}, "
        f"cuda_devices={diagnostics['cuda_device_count']}, "
        f"cudnn_available={diagnostics['cudnn_available']}"
    )
    if diagnostics["first_cuda_device_name"]:
        print(f"[info] cuda_device_0={diagnostics['first_cuda_device_name']}")
    if requested_device == "auto" and device_name == "cpu":
        print(
            "[warn] CUDA недоступна, используется CPU. "
            "Для строгой проверки запускайте с `--device cuda`."
        )
        print(f"[warn] {format_cuda_unavailable_hint(diagnostics)}")

    asr = pipeline(
        task="automatic-speech-recognition",
        model=model_id,
        chunk_length_s=chunk_length,
        batch_size=batch_size,
        torch_dtype=torch_dtype,
        device=device,
        token=hf_token,
        model_kwargs={"low_cpu_mem_usage": True, "use_safetensors": True},
    )

    # Встречи на русском: просим модель делать именно транскрибацию русского,
    # а не авто-перевод в английский.
    result = asr(
        str(input_path),
        return_timestamps=True,
        generate_kwargs={"language": TRANSCRIBE_LANGUAGE, "task": "transcribe"},
    )

    if not isinstance(result, dict) or "text" not in result:
        raise RuntimeError("Неожиданный формат ответа от модели.")
    result["backend"] = "hf"
    result["model_id"] = model_id
    result["requested_language"] = TRANSCRIBE_LANGUAGE
    return result


def run_mlx_transcription(input_path: Path, model_id: str) -> dict[str, Any]:
    import mlx_whisper

    print(f"[info] backend=mlx, model={model_id}")

    raw = mlx_whisper.transcribe(
        str(input_path),
        path_or_hf_repo=model_id,
        task="transcribe",
        language=TRANSCRIBE_LANGUAGE,
        word_timestamps=True,
    )

    if not isinstance(raw, dict) or "text" not in raw:
        raise RuntimeError("Неожиданный формат ответа от MLX Whisper.")

    chunks: list[dict[str, Any]] = []
    for segment in raw.get("segments", []):
        text = str(segment.get("text", "")).strip()
        chunk: dict[str, Any] = {"text": text}
        start = segment.get("start")
        end = segment.get("end")
        if start is not None or end is not None:
            chunk["timestamp"] = [
                float(start) if start is not None else None,
                float(end) if end is not None else None,
            ]
        words = segment.get("words")
        if words is not None:
            chunk["words"] = words
        chunks.append(chunk)

    normalized: dict[str, Any] = {
        "text": str(raw["text"]),
        "chunks": chunks,
        "segments": raw.get("segments", []),
        "language": raw.get("language"),
        "backend": "mlx",
        "model_id": model_id,
        "requested_language": TRANSCRIBE_LANGUAGE,
    }
    return normalized


def run_with_backend(
    backend: Literal["hf", "mlx"],
    input_path: Path,
    hf_model_id: str,
    mlx_model_id: str,
    chunk_length: int,
    batch_size: int,
    hf_token: str | None,
    device_preference: DevicePreference,
) -> dict[str, Any]:
    if backend == "hf":
        if not (has_module("torch") and has_module("transformers")):
            raise RuntimeError(
                "Для backend=hf нужны зависимости `torch` и `transformers`.\n"
                "Установите их командой `uv sync --extra hf`."
            )
        return run_transcription(
            input_path=input_path,
            model_id=hf_model_id,
            chunk_length=chunk_length,
            batch_size=batch_size,
            hf_token=hf_token,
            requested_device=device_preference,
        )

    if not has_module("mlx_whisper"):
        raise RuntimeError(
            "Для backend=mlx нужна зависимость `mlx-whisper`.\n"
            "Установите ее командой `uv sync --extra mlx`."
        )
    return run_mlx_transcription(
        input_path=input_path,
        model_id=mlx_model_id,
    )


def save_outputs(
    result: dict[str, Any],
    input_path: Path,
    output_txt: Path | None,
    output_json: Path | None,
) -> tuple[Path, Path]:
    txt_path = output_txt or input_path.with_suffix(".txt")
    json_path = output_json or input_path.with_suffix(".json")

    txt_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    txt_path.write_text(result["text"].strip() + "\n", encoding="utf-8")
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return txt_path, json_path


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    summary_path: Path | None = None

    try:
        ensure_dependencies(args.input)
        hf_token = resolve_hf_token(args.hf_token)
        configure_hf_auth(hf_token)
        backend = resolve_backend(args.backend)
        print(f"[info] backend={backend}")
        print(f"[info] hf_auth={'on' if hf_token else 'off'}")
        result = run_with_backend(
            backend=backend,
            input_path=args.input,
            hf_model_id=args.hf_model_id,
            mlx_model_id=args.mlx_model_id,
            chunk_length=args.chunk_length,
            batch_size=args.batch_size,
            hf_token=hf_token,
            device_preference=args.device,
        )
        txt_path, json_path = save_outputs(
            result=result,
            input_path=args.input,
            output_txt=args.output_txt,
            output_json=args.output_json,
        )

        if args.summarize:
            print(f"[info] Запускаю LLM-генерацию (mode={args.summary_mode})...")
            summary_text = request_summary_mode(
                transcript_text=result["text"].strip(),
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
            summary_path = save_summary(
                summary_text,
                args.summary_output
                or default_mode_output_path(txt_path, args.summary_mode),
            )
    except Exception as exc:  # noqa: BLE001
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    print("[ok] Транскрибация завершена.")
    print(f"[ok] TXT:  {txt_path}")
    print(f"[ok] JSON: {json_path}")
    if summary_path:
        print(f"[ok] {mode_output_label(args.summary_mode)}: {summary_path}")
    print("\n=== Текст транскрибации ===\n")
    print(result["text"].strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
