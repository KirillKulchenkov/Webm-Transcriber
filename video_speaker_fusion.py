from __future__ import annotations

import math
import re
from concurrent.futures import Executor, ThreadPoolExecutor
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


TEXT_FILTER_BLACKLIST = {
    "zoom",
    "teams",
    "google",
    "meet",
    "mute",
    "unmute",
    "recording",
    "запись",
    "микрофон",
    "камера",
    "поделиться",
    "демонстрация",
    "share",
    "screen",
    "leave",
    "meeting",
    "chat",
    "participants",
}
TEXT_FILTER_SUBSTRINGS = {
    "телемост",
    "meeting",
    "яндекс",
    "zoom",
    "teams",
    "google",
    "recording",
    "демонстра",
    "поделиться",
    "participants",
    "screen",
    "share",
}

VideoProfile = Literal["auto", "generic", "yandex_telemost"]


@dataclass(slots=True)
class VideoSpeakerFusionConfig:
    profile: VideoProfile = "auto"
    confidence_threshold: float = 0.8
    frame_sample_fps: float = 1.5
    segment_padding_sec: float = 0.35
    min_segment_duration_sec: float = 0.8
    lock_verify_every: int = 6
    lock_min_processed_segments: int = 3
    lock_min_total_votes: float = 8.0
    max_frames_per_segment: int = 8
    ocr_lang: str = "rus+eng"
    ocr_workers: int = 1
    participants: tuple[str, ...] = ()


@dataclass(slots=True)
class SegmentWindow:
    index: int
    start: float
    end: float


@dataclass(slots=True)
class SpeakerLockState:
    label: str
    votes: Counter[str]
    locked_name: str | None = None
    locked_confidence: float = 0.0
    processed_segments: int = 0
    skipped_segments_by_lock: int = 0
    mismatch_streak: int = 0


def run_video_speaker_fusion(
    input_path: Path,
    segments: list[dict[str, Any]],
    config: VideoSpeakerFusionConfig,
) -> dict[str, Any]:
    try:
        import cv2
        import pytesseract
    except ImportError as exc:  # pragma: no cover - guarded in prerequisites
        raise RuntimeError(
            "Для video speaker fusion нужны модули cv2 и pytesseract. "
            "Установите `uv sync --extra whisperx --extra video`."
        ) from exc

    participant_source = "provided" if config.participants else "ocr_discovery"
    normalized_participants = _normalize_participants(config.participants)
    participants_for_matching = [name for _, name in normalized_participants]
    ocr_workers = max(1, int(config.ocr_workers))

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        return {
            "enabled": True,
            "status": "skipped",
            "reason": "cannot_open_video_stream",
        }
    ocr_executor: Executor | None = None
    if ocr_workers > 1:
        ocr_executor = ThreadPoolExecutor(
            max_workers=ocr_workers,
            thread_name_prefix="fusion-ocr",
        )

    def _cleanup() -> None:
        cap.release()
        if ocr_executor is not None:
            ocr_executor.shutdown(wait=True)

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    duration_sec = _safe_duration(cap, fps)
    resolved_profile = _resolve_video_profile(
        cap=cap,
        duration_sec=duration_sec,
        requested_profile=config.profile,
        ocr_lang=config.ocr_lang,
        pytesseract_module=pytesseract,
        cv2_module=cv2,
    )

    speaker_windows = _group_segment_windows(
        segments,
        padding_sec=config.segment_padding_sec,
        min_segment_duration_sec=config.min_segment_duration_sec,
        duration_sec=duration_sec,
    )

    if not speaker_windows:
        _cleanup()
        return {
            "enabled": True,
            "status": "skipped",
            "reason": "no_valid_segments",
            "requested_profile": config.profile,
            "resolved_profile": resolved_profile,
        }

    if not participants_for_matching:
        discovered = _discover_participants(
            cap=cap,
            duration_sec=duration_sec,
            ocr_lang=config.ocr_lang,
            profile=resolved_profile,
            pytesseract_module=pytesseract,
            cv2_module=cv2,
            ocr_executor=ocr_executor,
        )
        normalized_participants = _normalize_participants(tuple(discovered))
        participants_for_matching = [name for _, name in normalized_participants]

    # If OCR discovery cannot find reliable candidates, prefer safe fallback:
    # leave SPEAKER_XX unchanged instead of injecting noisy names.
    if participant_source == "ocr_discovery" and not participants_for_matching:
        _cleanup()
        return {
            "enabled": True,
            "status": "skipped",
            "reason": "insufficient_participants_after_ocr_filter",
            "participants_source": participant_source,
            "participants": [],
            "requested_profile": config.profile,
            "resolved_profile": resolved_profile,
        }

    speaker_states: dict[str, SpeakerLockState] = {
        speaker_label: SpeakerLockState(label=speaker_label, votes=Counter())
        for speaker_label in speaker_windows
    }

    frame_reads = 0
    analyzed_segments = 0
    total_skipped_by_lock = 0

    for speaker_label, windows in speaker_windows.items():
        state = speaker_states[speaker_label]

        for order, window in enumerate(windows, start=1):
            if (
                state.locked_name
                and config.lock_verify_every > 1
                and (order % config.lock_verify_every) != 0
            ):
                _annotate_segment(
                    segments[window.index],
                    state.label,
                    state.locked_name,
                    state.locked_confidence,
                    locked=True,
                    evidence={"strategy": "lock_skip"},
                )
                state.skipped_segments_by_lock += 1
                total_skipped_by_lock += 1
                continue

            votes, frames_used, evidence = _analyze_segment_window(
                cap=cap,
                start_sec=window.start,
                end_sec=window.end,
                sample_fps=config.frame_sample_fps,
                max_frames_per_segment=config.max_frames_per_segment,
                participants=participants_for_matching,
                normalized_participants=normalized_participants,
                ocr_lang=config.ocr_lang,
                profile=resolved_profile,
                pytesseract_module=pytesseract,
                cv2_module=cv2,
                ocr_executor=ocr_executor,
            )
            frame_reads += frames_used
            state.processed_segments += 1
            analyzed_segments += 1

            if not votes:
                if state.locked_name:
                    _annotate_segment(
                        segments[window.index],
                        state.label,
                        state.locked_name,
                        state.locked_confidence,
                        locked=True,
                        evidence={
                            "strategy": "lock_fallback",
                            "frames": frames_used,
                            **evidence,
                        },
                    )
                else:
                    _annotate_segment(
                        segments[window.index],
                        state.label,
                        None,
                        0.0,
                        locked=False,
                        evidence={
                            "strategy": "no_video_signal",
                            "frames": frames_used,
                            **evidence,
                        },
                    )
                continue

            state.votes.update(votes)
            name, confidence = _top_confidence(state.votes)

            if participant_source == "ocr_discovery":
                lock_confidence_threshold = max(config.confidence_threshold, 0.9)
                lock_min_processed_segments = max(config.lock_min_processed_segments + 2, 5)
                lock_min_total_votes = max(config.lock_min_total_votes + 12.0, 20.0)
            else:
                lock_confidence_threshold = config.confidence_threshold
                lock_min_processed_segments = max(1, config.lock_min_processed_segments)
                lock_min_total_votes = max(0.0, config.lock_min_total_votes)

            if state.locked_name:
                state = _update_existing_lock(state, votes, config.confidence_threshold)
                name = state.locked_name or name
                confidence = max(confidence, state.locked_confidence)
            elif (
                confidence >= lock_confidence_threshold
                and name
                and state.processed_segments >= lock_min_processed_segments
                and sum(state.votes.values()) >= lock_min_total_votes
            ):
                state.locked_name = name
                state.locked_confidence = confidence

            _annotate_segment(
                segments[window.index],
                state.label,
                name,
                confidence,
                locked=state.locked_name is not None,
                evidence={
                    "strategy": "segment_analysis",
                    "frames": frames_used,
                    "window_start": window.start,
                    "window_end": window.end,
                    **evidence,
                },
            )

        speaker_states[speaker_label] = state

    _cleanup()

    speaker_mapping: dict[str, dict[str, Any]] = {}
    for speaker_label, state in speaker_states.items():
        candidate_name, candidate_confidence = _top_confidence(state.votes)
        if participant_source == "ocr_discovery" and not state.locked_name:
            final_name = None
            final_confidence = 0.0
        else:
            final_name = state.locked_name or candidate_name
            final_confidence = (
                state.locked_confidence if state.locked_name else candidate_confidence
            )
        speaker_mapping[speaker_label] = {
            "speaker_name": final_name,
            "confidence": round(final_confidence, 4),
            "locked": bool(state.locked_name),
            "processed_segments": state.processed_segments,
            "skipped_segments_by_lock": state.skipped_segments_by_lock,
            "vote_totals": dict(state.votes),
        }

    for segment in segments:
        speaker_label = str(segment.get("speaker", "SPEAKER_UNKNOWN"))
        if segment.get("speaker_name"):
            continue
        mapping = speaker_mapping.get(speaker_label)
        if not mapping:
            continue
        speaker_name = mapping.get("speaker_name")
        confidence = float(mapping.get("confidence") or 0.0)
        if speaker_name:
            _annotate_segment(
                segment,
                speaker_label,
                str(speaker_name),
                confidence,
                locked=bool(mapping.get("locked")),
                evidence={"strategy": "speaker_level_fallback"},
            )

    return {
        "enabled": True,
        "status": "ok",
        "requested_profile": config.profile,
        "resolved_profile": resolved_profile,
        "participants_source": participant_source,
        "confidence_threshold": config.confidence_threshold,
        "frame_sample_fps": config.frame_sample_fps,
        "segment_padding_sec": config.segment_padding_sec,
        "min_segment_duration_sec": config.min_segment_duration_sec,
        "lock_verify_every": config.lock_verify_every,
        "lock_min_processed_segments": config.lock_min_processed_segments,
        "lock_min_total_votes": config.lock_min_total_votes,
        "max_frames_per_segment": config.max_frames_per_segment,
        "ocr_lang": config.ocr_lang,
        "ocr_workers": ocr_workers,
        "participants": participants_for_matching,
        "speakers": speaker_mapping,
        "stats": {
            "total_segments": len(segments),
            "analyzed_segments": analyzed_segments,
            "skipped_segments_by_lock": total_skipped_by_lock,
            "frames_processed": frame_reads,
            "fps": fps,
            "duration_sec": duration_sec,
        },
    }


def _safe_duration(cap: Any, fps: float) -> float | None:
    import cv2

    frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    if fps <= 0.0 or frame_count <= 0.0:
        return None
    return frame_count / fps


def _group_segment_windows(
    segments: list[dict[str, Any]],
    *,
    padding_sec: float,
    min_segment_duration_sec: float,
    duration_sec: float | None,
) -> dict[str, list[SegmentWindow]]:
    grouped: dict[str, list[SegmentWindow]] = defaultdict(list)

    for idx, segment in enumerate(segments):
        start_raw = segment.get("start")
        end_raw = segment.get("end")
        if start_raw is None or end_raw is None:
            continue

        start = max(0.0, float(start_raw) - padding_sec)
        end = max(start, float(end_raw) + padding_sec)
        if duration_sec is not None:
            end = min(end, duration_sec)

        if end - start < min_segment_duration_sec:
            continue

        speaker_label = str(segment.get("speaker", "SPEAKER_UNKNOWN"))
        grouped[speaker_label].append(
            SegmentWindow(index=idx, start=start, end=end)
        )

    return grouped


def _resolve_video_profile(
    *,
    cap: Any,
    duration_sec: float | None,
    requested_profile: VideoProfile,
    ocr_lang: str,
    pytesseract_module: Any,
    cv2_module: Any,
) -> VideoProfile:
    profile = _normalize_profile(requested_profile)
    if profile != "auto":
        return profile

    probe_times = _profile_probe_times(duration_sec)
    telemost_tile_hits = 0
    for second in probe_times:
        frame = _read_frame_at(cap, second)
        if frame is None:
            continue

        frame_h = frame.shape[0]
        top_start = min(frame_h - 1, max(0, int(frame_h * 0.01)))
        top_end = min(frame_h, max(top_start + 1, int(frame_h * 0.12)))
        top_strip = frame[top_start:top_end, :]
        if top_strip.size == 0:
            continue

        lines = _ocr_lines(top_strip, ocr_lang, pytesseract_module, cv2_module)
        if any(_contains_telemost_marker(line) for line in lines):
            return "yandex_telemost"

        rect, score = _detect_active_tile_telemost(frame, cv2_module)
        if rect is None:
            continue
        _, y, _, h = rect
        top_ratio = (y + h) / max(1, frame_h)
        if top_ratio <= 0.3 and score >= 0.16:
            telemost_tile_hits += 1
            if telemost_tile_hits >= 2:
                return "yandex_telemost"

    return "generic"


def _normalize_profile(requested_profile: str) -> VideoProfile:
    normalized = requested_profile.strip().lower()
    if normalized in {"generic", "yandex_telemost"}:
        return normalized
    return "auto"


def _profile_probe_times(duration_sec: float | None) -> list[float]:
    if duration_sec is None or duration_sec <= 0:
        return [0.0, 4.0, 12.0, 25.0]

    checkpoints = {0.0}
    for ratio in (0.03, 0.08, 0.2, 0.5):
        checkpoints.add(max(0.0, min(duration_sec - 0.05, duration_sec * ratio)))
    return sorted(round(point, 3) for point in checkpoints)


def _contains_telemost_marker(text: str) -> bool:
    cleaned = re.sub(r"[^A-Za-zА-Яа-яЁё ]+", " ", text.lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return False
    return "телемост" in cleaned or "telemost" in cleaned


def _discover_participants(
    *,
    cap: Any,
    duration_sec: float | None,
    ocr_lang: str,
    profile: VideoProfile,
    pytesseract_module: Any,
    cv2_module: Any,
    ocr_executor: Executor | None = None,
) -> list[str]:
    if duration_sec is None or duration_sec <= 0:
        sample_times = [0.0, 10.0, 20.0]
    else:
        upper_bound = min(duration_sec, 1200.0)
        step = max(5.0, upper_bound / 20)
        sample_times = [
            round(step * i, 3)
            for i in range(int(upper_bound // step) + 1)
        ]

    candidates: Counter[str] = Counter()

    for second in sample_times:
        frame = _read_frame_at(cap, second)
        if frame is None:
            continue
        strips = _build_name_regions(frame, active_tile=None, profile=profile)
        for cleaned in _ocr_region_candidates_batch(
            strips,
            ocr_lang=ocr_lang,
            pytesseract_module=pytesseract_module,
            cv2_module=cv2_module,
            ocr_executor=ocr_executor,
        ):
            candidates[cleaned] += 1

    min_hits = 2 if profile == "yandex_telemost" else 3
    discovered = [
        name
        for name, count in candidates.most_common(25)
        if count >= min_hits
        and _is_plausible_name_candidate(name)
        and (profile != "yandex_telemost" or len(name.split()) >= 2)
    ][:15]
    if profile == "yandex_telemost" and not discovered:
        # Fallback for narrow top bars where OCR sees names rarely.
        discovered = [
            name
            for name, count in candidates.most_common(25)
            if count >= 1
            and _is_plausible_name_candidate(name)
            and len(name.split()) >= 2
        ][:10]
    return discovered


def _analyze_segment_window(
    *,
    cap: Any,
    start_sec: float,
    end_sec: float,
    sample_fps: float,
    max_frames_per_segment: int,
    participants: list[str],
    normalized_participants: list[tuple[str, str]],
    ocr_lang: str,
    profile: VideoProfile,
    pytesseract_module: Any,
    cv2_module: Any,
    ocr_executor: Executor | None = None,
) -> tuple[Counter[str], int, dict[str, Any]]:
    votes: Counter[str] = Counter()
    frames = 0
    active_tile_detections = 0
    frame_tile_scores: list[float] = []
    frame_best_names: list[str | None] = []
    frame_best_scores: list[float] = []
    task_regions: list[Any] = []
    task_frame_indexes: list[int] = []

    timestamps = _segment_timestamps(
        start_sec,
        end_sec,
        sample_fps,
        max_frames=max_frames_per_segment,
    )
    for timestamp in timestamps:
        frame = _read_frame_at(cap, timestamp)
        if frame is None:
            continue
        frames += 1

        active_tile, tile_score = _detect_active_tile(
            frame,
            cv2_module,
            profile=profile,
        )
        if active_tile is not None:
            active_tile_detections += 1

        regions = _build_name_regions(frame, active_tile, profile=profile)
        frame_index = len(frame_tile_scores)
        frame_tile_scores.append(tile_score)
        frame_best_names.append(None)
        frame_best_scores.append(0.0)

        for region in regions:
            task_regions.append(region)
            task_frame_indexes.append(frame_index)

    if task_regions:
        grouped_candidates = _ocr_region_candidates_groups(
            task_regions,
            ocr_lang=ocr_lang,
            pytesseract_module=pytesseract_module,
            cv2_module=cv2_module,
            ocr_executor=ocr_executor,
        )
        for task_idx, candidates in enumerate(grouped_candidates):
            frame_index = task_frame_indexes[task_idx]
            best_name = frame_best_names[frame_index]
            best_score = frame_best_scores[frame_index]
            for cleaned in candidates:
                matched_name, name_score = _match_name(
                    cleaned,
                    participants,
                    normalized_participants,
                )
                if matched_name and name_score > best_score:
                    best_name = matched_name
                    best_score = name_score
            frame_best_names[frame_index] = best_name
            frame_best_scores[frame_index] = best_score

    for frame_index, best_name in enumerate(frame_best_names):
        if not best_name:
            continue
        tile_score = frame_tile_scores[frame_index]
        best_score = frame_best_scores[frame_index]
        weighted_score = best_score * (1.0 + tile_score)
        votes[best_name] += weighted_score

    evidence = {
        "active_tile_ratio": round(
            (active_tile_detections / frames) if frames else 0.0,
            4,
        ),
    }
    return votes, frames, evidence


def _update_existing_lock(
    state: SpeakerLockState,
    current_votes: Counter[str],
    confidence_threshold: float,
) -> SpeakerLockState:
    if not state.locked_name or not current_votes:
        return state

    current_name, current_confidence = _top_confidence(current_votes)
    if not current_name:
        return state

    if (
        current_name != state.locked_name
        and current_confidence >= confidence_threshold
    ):
        state.mismatch_streak += 1
        if state.mismatch_streak >= 2:
            state.locked_name = current_name
            state.locked_confidence = current_confidence
            state.mismatch_streak = 0
    else:
        state.mismatch_streak = 0
        if current_name == state.locked_name:
            state.locked_confidence = max(state.locked_confidence, current_confidence)

    return state


def _annotate_segment(
    segment: dict[str, Any],
    speaker_label: str,
    speaker_name: str | None,
    confidence: float,
    *,
    locked: bool,
    evidence: dict[str, Any],
) -> None:
    segment["speaker_original"] = speaker_label
    if speaker_name:
        segment["speaker_name"] = speaker_name
        segment["speaker_confidence"] = round(confidence, 4)
    segment["speaker_locked"] = locked
    segment["speaker_evidence"] = evidence


def _read_frame_at(cap: Any, seconds: float) -> Any | None:
    import cv2

    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, seconds) * 1000.0)
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def _segment_timestamps(
    start_sec: float,
    end_sec: float,
    sample_fps: float,
    *,
    max_frames: int,
) -> list[float]:
    if sample_fps <= 0:
        return [start_sec]

    step = max(0.1, 1.0 / sample_fps)
    timestamps: list[float] = []
    current = start_sec
    while current <= end_sec:
        timestamps.append(round(current, 3))
        current += step

    if not timestamps:
        return [round(start_sec, 3)]

    if max_frames > 0 and len(timestamps) > max_frames:
        if max_frames == 1:
            return [timestamps[len(timestamps) // 2]]
        selected: list[float] = []
        last_index = len(timestamps) - 1
        for i in range(max_frames):
            idx = round((i / (max_frames - 1)) * last_index)
            selected.append(timestamps[idx])
        return sorted(set(selected))
    return timestamps


def _detect_active_tile(
    frame: Any,
    cv2_module: Any,
    *,
    profile: VideoProfile,
) -> tuple[tuple[int, int, int, int] | None, float]:
    if profile == "yandex_telemost":
        return _detect_active_tile_telemost(frame, cv2_module)
    return _detect_active_tile_generic(frame, cv2_module)


def _detect_active_tile_telemost(
    frame: Any,
    cv2_module: Any,
) -> tuple[tuple[int, int, int, int] | None, float]:
    frame_h, frame_w = frame.shape[:2]
    top_limit = max(1, int(frame_h * 0.24))
    top_strip = frame[:top_limit, :]

    hsv_top = cv2_module.cvtColor(top_strip, cv2_module.COLOR_BGR2HSV)
    green_mask = cv2_module.inRange(hsv_top, (32, 75, 95), (95, 255, 255))
    kernel = cv2_module.getStructuringElement(cv2_module.MORPH_RECT, (3, 3))
    green_mask = cv2_module.morphologyEx(
        green_mask,
        cv2_module.MORPH_CLOSE,
        kernel,
        iterations=1,
    )
    contours, _ = cv2_module.findContours(
        green_mask,
        cv2_module.RETR_EXTERNAL,
        cv2_module.CHAIN_APPROX_SIMPLE,
    )

    frame_area = float(frame_h * frame_w)
    best_rect: tuple[int, int, int, int] | None = None
    best_score = 0.0

    for contour in contours:
        x, y, w, h = cv2_module.boundingRect(contour)
        area = float(w * h)
        if area < frame_area * 0.001 or area > frame_area * 0.12:
            continue

        aspect = w / max(h, 1)
        if aspect < 0.6 or aspect > 4.8:
            continue

        roi = green_mask[y : y + h, x : x + w]
        if roi.size == 0:
            continue

        border_width = max(2, int(min(w, h) * 0.08))
        border = _build_border_mask(roi.shape[0], roi.shape[1], border_width)
        border_pixels = roi[border]
        if border_pixels.size == 0:
            continue

        border_ratio = float((border_pixels > 0).mean())
        interior_ratio = 0.0
        if h > border_width * 2 and w > border_width * 2:
            interior = roi[
                border_width : h - border_width,
                border_width : w - border_width,
            ]
            if interior.size > 0:
                interior_ratio = float((interior > 0).mean())

        area_bonus = min(0.25, math.log1p(area / frame_area * 100.0) / 10.0)
        score = border_ratio - (interior_ratio * 0.9) + area_bonus
        if score > best_score:
            best_score = score
            best_rect = (x, y, w, h)

    if best_rect is not None and best_score >= 0.12:
        return best_rect, round(best_score, 4)

    return _detect_active_tile_generic(frame, cv2_module)


def _detect_active_tile_generic(
    frame: Any,
    cv2_module: Any,
) -> tuple[tuple[int, int, int, int] | None, float]:
    hsv = cv2_module.cvtColor(frame, cv2_module.COLOR_BGR2HSV)

    green_mask = cv2_module.inRange(hsv, (35, 60, 80), (95, 255, 255))
    blue_mask = cv2_module.inRange(hsv, (90, 60, 80), (140, 255, 255))
    white_glow = cv2_module.inRange(hsv, (0, 0, 220), (180, 40, 255))
    mask = cv2_module.bitwise_or(green_mask, blue_mask)
    mask = cv2_module.bitwise_or(mask, white_glow)

    kernel = cv2_module.getStructuringElement(cv2_module.MORPH_RECT, (5, 5))
    mask = cv2_module.morphologyEx(mask, cv2_module.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2_module.findContours(
        mask,
        cv2_module.RETR_EXTERNAL,
        cv2_module.CHAIN_APPROX_SIMPLE,
    )

    frame_h, frame_w = frame.shape[:2]
    frame_area = float(frame_h * frame_w)
    best_rect: tuple[int, int, int, int] | None = None
    best_score = 0.0

    for contour in contours:
        x, y, w, h = cv2_module.boundingRect(contour)
        area = float(w * h)
        if area < frame_area * 0.005 or area > frame_area * 0.9:
            continue

        aspect = w / max(h, 1)
        if aspect < 0.35 or aspect > 3.6:
            continue

        roi = mask[y : y + h, x : x + w]
        if roi.size == 0:
            continue

        border_width = max(2, int(min(w, h) * 0.06))
        border = _build_border_mask(roi.shape[0], roi.shape[1], border_width)
        border_pixels = roi[border]
        if border_pixels.size == 0:
            continue

        border_ratio = float((border_pixels > 0).mean())
        interior_ratio = 0.0
        if h > border_width * 2 and w > border_width * 2:
            interior = roi[
                border_width : h - border_width,
                border_width : w - border_width,
            ]
            if interior.size > 0:
                interior_ratio = float((interior > 0).mean())

        area_bonus = min(0.2, math.log1p(area / frame_area * 100.0) / 12.0)
        score = border_ratio - (interior_ratio * 0.6) + area_bonus

        if score > best_score:
            best_score = score
            best_rect = (x, y, w, h)

    if best_rect is None or best_score < 0.08:
        return None, 0.0
    return best_rect, round(best_score, 4)


def _build_border_mask(height: int, width: int, border_width: int) -> Any:
    import numpy as np

    mask = np.zeros((height, width), dtype=bool)
    mask[:border_width, :] = True
    mask[-border_width:, :] = True
    mask[:, :border_width] = True
    mask[:, -border_width:] = True
    return mask


def _build_name_regions(
    frame: Any,
    active_tile: tuple[int, int, int, int] | None,
    *,
    profile: VideoProfile,
) -> list[Any]:
    if profile == "yandex_telemost":
        return _build_name_regions_telemost(frame, active_tile)
    return _build_name_regions_generic(frame, active_tile)


def _build_name_regions_telemost(
    frame: Any,
    active_tile: tuple[int, int, int, int] | None,
) -> list[Any]:
    frame_h, frame_w = frame.shape[:2]
    regions: list[Any] = []

    if active_tile is not None:
        x, y, w, h = active_tile
        tile_bottom_start = y + int(h * 0.62)
        tile_bottom_end = min(frame_h, y + h)
        if tile_bottom_end > tile_bottom_start:
            regions.append(frame[tile_bottom_start:tile_bottom_end, x : x + w])

        x0 = max(0, x - int(w * 0.06))
        x1 = min(frame_w, x + w + int(w * 0.06))
        y0 = max(0, y)
        y1 = min(frame_h, y + h + int(h * 0.1))
        if y1 > y0 and x1 > x0:
            regions.append(frame[y0:y1, x0:x1])

    top_start = max(0, int(frame_h * 0.03))
    top_end = min(frame_h, max(top_start + 1, int(frame_h * 0.24)))
    if top_end > top_start:
        regions.append(frame[top_start:top_end, :])

    badge_top = max(0, int(frame_h * 0.86))
    badge_bottom = min(frame_h, int(frame_h * 0.95))
    badge_left = max(0, int(frame_w * 0.30))
    badge_right = min(frame_w, int(frame_w * 0.70))
    if badge_bottom > badge_top and badge_right > badge_left:
        regions.append(frame[badge_top:badge_bottom, badge_left:badge_right])

    if active_tile is None:
        bottom_strip_start = max(0, int(frame_h * 0.78))
        regions.append(frame[bottom_strip_start:, :])

    return [region for region in regions if region.size > 0]


def _build_name_regions_generic(
    frame: Any,
    active_tile: tuple[int, int, int, int] | None,
) -> list[Any]:
    frame_h, frame_w = frame.shape[:2]
    regions: list[Any] = []

    if active_tile is not None:
        x, y, w, h = active_tile
        tile_bottom_start = y + int(h * 0.65)
        tile_bottom_end = min(frame_h, y + h)
        if tile_bottom_end > tile_bottom_start:
            regions.append(frame[tile_bottom_start:tile_bottom_end, x : x + w])

    top_strip_end = max(1, int(frame_h * 0.18))
    bottom_strip_start = max(0, int(frame_h * 0.78))

    regions.append(frame[:top_strip_end, :])
    if active_tile is None:
        regions.append(frame[bottom_strip_start:, :])

    return [region for region in regions if region.size > 0]


def _ocr_lines(
    image: Any,
    ocr_lang: str,
    pytesseract_module: Any,
    cv2_module: Any,
) -> list[str]:
    gray = cv2_module.cvtColor(image, cv2_module.COLOR_BGR2GRAY)
    gray = cv2_module.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2_module.INTER_CUBIC)
    gray = cv2_module.GaussianBlur(gray, (3, 3), 0)

    adaptive = cv2_module.adaptiveThreshold(
        gray,
        255,
        cv2_module.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2_module.THRESH_BINARY,
        35,
        11,
    )

    lines: list[str] = []
    for psm in (7, 6):
        config = f"--oem 3 --psm {psm}"
        text = pytesseract_module.image_to_string(adaptive, lang=ocr_lang, config=config)
        if not text.strip():
            continue
        for raw in text.splitlines():
            candidate = raw.strip()
            if candidate:
                lines.append(candidate)

    # Preserve order while de-duplicating.
    seen: set[str] = set()
    unique_lines: list[str] = []
    for line in lines:
        if line in seen:
            continue
        seen.add(line)
        unique_lines.append(line)
    return unique_lines


def _ocr_region_candidates(
    region: Any,
    ocr_lang: str,
    pytesseract_module: Any,
    cv2_module: Any,
) -> list[str]:
    candidates: list[str] = []
    for text in _ocr_lines(region, ocr_lang, pytesseract_module, cv2_module):
        for fragment in _extract_name_fragments(text):
            cleaned = _clean_text_candidate(fragment)
            if cleaned:
                candidates.append(cleaned)

    # Preserve order while de-duplicating.
    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _ocr_region_candidates_batch(
    regions: list[Any],
    *,
    ocr_lang: str,
    pytesseract_module: Any,
    cv2_module: Any,
    ocr_executor: Executor | None,
) -> list[str]:
    grouped = _ocr_region_candidates_groups(
        regions,
        ocr_lang=ocr_lang,
        pytesseract_module=pytesseract_module,
        cv2_module=cv2_module,
        ocr_executor=ocr_executor,
    )

    combined: list[str] = []
    seen: set[str] = set()
    for group in grouped:
        for candidate in group:
            key = candidate.lower()
            if key in seen:
                continue
            seen.add(key)
            combined.append(candidate)
    return combined


def _ocr_region_candidates_groups(
    regions: list[Any],
    *,
    ocr_lang: str,
    pytesseract_module: Any,
    cv2_module: Any,
    ocr_executor: Executor | None,
) -> list[list[str]]:
    if not regions:
        return []

    if ocr_executor is not None and len(regions) > 1:
        mapped = ocr_executor.map(
            _ocr_region_candidates,
            regions,
            [ocr_lang] * len(regions),
            [pytesseract_module] * len(regions),
            [cv2_module] * len(regions),
        )
        return list(mapped)

    return [
        _ocr_region_candidates(
            region,
            ocr_lang=ocr_lang,
            pytesseract_module=pytesseract_module,
            cv2_module=cv2_module,
        )
        for region in regions
    ]


def _extract_name_fragments(text: str) -> list[str]:
    base = re.sub(r"\s+", " ", text.strip())
    if not base:
        return []

    fragments: list[str] = [base]
    chunks: list[str] = []
    # OCR of top ribbon often returns several names in one line separated by symbols.
    for chunk in re.split(r"[|/&;,@№•·]+", base):
        normalized_chunk = re.sub(r"\s+", " ", chunk).strip()
        if normalized_chunk:
            chunks.append(normalized_chunk)
            fragments.append(normalized_chunk)

    title_pair = re.compile(r"\b[A-ZА-ЯЁ][a-zа-яё]{2,}\s+[A-ZА-ЯЁ][a-zа-яё]{2,}\b")
    for scope in [base, *chunks]:
        fragments.extend(title_pair.findall(scope))

    seen: set[str] = set()
    unique: list[str] = []
    for fragment in fragments:
        cleaned = fragment.strip()
        if not cleaned:
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)
        unique.append(cleaned)
    return unique


def _clean_text_candidate(text: str) -> str | None:
    cleaned = text.strip()
    if not cleaned:
        return None

    cleaned = re.sub(r"[^0-9A-Za-zА-Яа-яЁё\-_. ]+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) < 2 or len(cleaned) > 40:
        return None

    lowered = cleaned.lower()
    if lowered in TEXT_FILTER_BLACKLIST:
        return None

    letters_only = re.sub(r"[^A-Za-zА-Яа-яЁё]", "", cleaned)
    if len(letters_only) < 2:
        return None

    lowered = cleaned.lower()
    for token in TEXT_FILTER_SUBSTRINGS:
        if token in lowered:
            return None

    return cleaned


def _is_plausible_name_candidate(value: str) -> bool:
    cleaned = value.strip()
    if not cleaned:
        return False

    lowered = cleaned.lower()
    for token in TEXT_FILTER_SUBSTRINGS:
        if token in lowered:
            return False

    if re.search(r"\d", cleaned):
        return False

    words = [w for w in re.split(r"\s+", cleaned) if w]
    if not words or len(words) > 3:
        return False

    letters_only = re.sub(r"[^A-Za-zА-Яа-яЁё]", "", cleaned)
    if len(letters_only) < 4:
        return False

    has_cyr = bool(re.search(r"[А-Яа-яЁё]", cleaned))
    has_lat = bool(re.search(r"[A-Za-z]", cleaned))
    if has_cyr and has_lat:
        return False

    unique_ratio = len(set(letters_only.lower())) / max(1, len(letters_only))
    if unique_ratio < 0.35:
        return False

    if re.search(r"(.)\1\1", letters_only.lower()):
        return False

    for word in words:
        alpha = re.sub(r"[^A-Za-zА-Яа-яЁё]", "", word)
        min_word_len = 4 if len(words) == 1 else 3
        if len(alpha) < min_word_len:
            return False

        lowered_word = alpha.lower()
        if has_cyr:
            vowels = "аеёиоуыэюя"
        else:
            vowels = "aeiouy"

        if not any(ch in vowels for ch in lowered_word):
            return False

    return True


def _normalize_name(value: str) -> str:
    lowered = value.lower().strip()
    lowered = re.sub(r"\s+", " ", lowered)
    lowered = re.sub(r"[^0-9a-zа-яё\-_. ]+", "", lowered)
    return lowered


def _normalize_participants(participants: tuple[str, ...]) -> list[tuple[str, str]]:
    normalized: list[tuple[str, str]] = []
    seen: set[str] = set()

    for participant in participants:
        original = participant.strip()
        if not original:
            continue
        norm = _normalize_name(original)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        normalized.append((norm, original))

    return normalized


def _match_name(
    ocr_candidate: str,
    participants: list[str],
    normalized_participants: list[tuple[str, str]],
) -> tuple[str | None, float]:
    from difflib import SequenceMatcher

    if not participants:
        return None, 0.0

    normalized_candidate = _normalize_name(ocr_candidate)
    if not normalized_candidate:
        return None, 0.0

    best_name = None
    best_score = 0.0

    for normalized_name, original_name in normalized_participants:
        score = SequenceMatcher(None, normalized_candidate, normalized_name).ratio()
        if normalized_candidate in normalized_name or normalized_name in normalized_candidate:
            score = max(score, 0.88)
        if score > best_score:
            best_score = score
            best_name = original_name

    if best_score < 0.55:
        return None, 0.0
    return best_name, best_score


def _top_confidence(votes: Counter[str]) -> tuple[str | None, float]:
    if not votes:
        return None, 0.0

    total = float(sum(votes.values()))
    if total <= 0:
        return None, 0.0

    name, score = votes.most_common(1)[0]
    return name, float(score) / total
