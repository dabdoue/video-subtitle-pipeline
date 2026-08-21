from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
from urllib.parse import urlparse


VERSION = "0.2.0"
DEFAULT_ASR_MODEL = "nvidia/nemotron-3.5-asr-streaming-0.6b"

LANGUAGE_CODES = {
    "korean": "ko",
    "japanese": "ja",
    "chinese": "zh",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "portuguese": "pt",
    "english": "en",
}

LANGUAGE_FONTS = {
    "ko": "Noto Sans CJK KR",
    "ja": "Noto Sans CJK JP",
    "zh": "Noto Sans CJK SC",
}


class PipelineError(RuntimeError):
    """A user-facing pipeline failure."""


@dataclass
class Segment:
    id: str
    start: float
    end: float
    text: str = ""
    translation_parts: list[str] = field(default_factory=list)
    raw_asr_text: str = ""
    asr_audio_start: float | None = None
    asr_audio_end: float | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Cue:
    start: float
    end: float
    text: str
    segment_id: str


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_parts(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = normalize_text(value)
        return [text] if text else []
    if not isinstance(value, list):
        raise PipelineError("translation_parts must be a string or list of strings")
    return [text for item in value if (text := normalize_text(item))]


def parse_time(value: str | float | int) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    value = value.strip()
    if ":" not in value:
        try:
            return float(value)
        except ValueError as exc:
            raise PipelineError(f"Invalid timestamp: {value!r}") from exc
    cleaned = value.replace(".", ",")
    match = re.fullmatch(r"(?:(\d+):)?(\d{1,2}):(\d{1,2})(?:,(\d{1,3}))?", cleaned)
    if not match:
        raise PipelineError(f"Invalid timestamp: {value!r}")
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2))
    seconds = int(match.group(3))
    millis = int((match.group(4) or "0").ljust(3, "0"))
    return hours * 3600 + minutes * 60 + seconds + millis / 1000


def format_srt_time(value: float) -> str:
    total_ms = max(0, round(value * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def validate_segments(segments: Sequence[Segment]) -> None:
    if not segments:
        raise PipelineError("No segments were found")
    seen: set[str] = set()
    previous_end = -math.inf
    for segment in segments:
        if segment.id in seen:
            raise PipelineError(f"Duplicate segment id: {segment.id}")
        seen.add(segment.id)
        if segment.start < 0 or segment.end <= segment.start:
            raise PipelineError(
                f"Invalid range for segment {segment.id}: {segment.start} -> {segment.end}"
            )
        if segment.start < previous_end - 0.001:
            raise PipelineError(f"Overlapping nominal anchors at segment {segment.id}")
        previous_end = segment.end


def parse_srt(path: Path) -> list[Segment]:
    content = path.read_text(encoding="utf-8-sig").strip()
    if not content:
        raise PipelineError(f"SRT file is empty: {path}")
    segments: list[Segment] = []
    for ordinal, block in enumerate(re.split(r"\r?\n\s*\r?\n", content), 1):
        lines = block.splitlines()
        time_index = next((index for index, line in enumerate(lines) if " --> " in line), None)
        if time_index is None:
            raise PipelineError(f"SRT block {ordinal} has no timestamp line")
        start_text, end_text = lines[time_index].split(" --> ", 1)
        identifier = lines[0].strip() if time_index > 0 else str(ordinal)
        segments.append(
            Segment(
                id=identifier or str(ordinal),
                start=parse_time(start_text),
                end=parse_time(end_text),
                text=normalize_text(" ".join(lines[time_index + 1 :])),
            )
        )
    validate_segments(segments)
    return segments


def parse_json_segments(path: Path) -> list[Segment]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("segments") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise PipelineError("JSON anchors must be a list or object with a segments list")
    segments: list[Segment] = []
    for ordinal, item in enumerate(items, 1):
        if not isinstance(item, dict):
            raise PipelineError(f"JSON segment {ordinal} must be an object")
        parts = item.get("translation_parts", item.get("translated_parts"))
        if parts is None and item.get("translation") is not None:
            parts = [item["translation"]]
        segments.append(
            Segment(
                id=str(item.get("id", ordinal)),
                start=parse_time(item["start"]),
                end=parse_time(item["end"]),
                text=normalize_text(item.get("text", "")),
                translation_parts=normalize_parts(parts),
                raw_asr_text=normalize_text(item.get("raw_asr_text", "")),
                asr_audio_start=(
                    float(item["asr_audio_start"])
                    if item.get("asr_audio_start") is not None
                    else None
                ),
                asr_audio_end=(
                    float(item["asr_audio_end"])
                    if item.get("asr_audio_end") is not None
                    else None
                ),
            )
        )
    validate_segments(segments)
    return segments


def load_segments(path: Path) -> list[Segment]:
    if path.suffix.lower() == ".srt":
        return parse_srt(path)
    if path.suffix.lower() == ".json":
        return parse_json_segments(path)
    raise PipelineError("Anchors must be .srt or .json")


def fixed_segments(
    duration: float, seconds: float, minimum_tail: float = 1.0
) -> list[Segment]:
    if seconds <= 0:
        raise PipelineError("--anchor-seconds must be greater than zero")
    count = math.ceil(duration / seconds)
    if count > 1 and duration - (count - 1) * seconds < minimum_tail:
        count -= 1
    return [
        Segment(
            id=f"{index + 1:04d}",
            start=index * seconds,
            end=(duration if index == count - 1 else min(duration, (index + 1) * seconds)),
        )
        for index in range(count)
    ]


def run(
    command: Sequence[str],
    *,
    capture: bool = False,
    input_text: str | None = None,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(command),
            check=True,
            text=True,
            input=input_text,
            cwd=cwd,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
        )
    except FileNotFoundError as exc:
        raise PipelineError(f"Required command is not installed: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        details = "\n".join(part for part in [exc.stdout, exc.stderr] if part)
        raise PipelineError(
            f"Command failed ({' '.join(command[:3])}):\n{details[-5000:]}"
        ) from exc


def require_commands(names: Iterable[str]) -> None:
    missing = [name for name in names if shutil.which(name) is None]
    if missing:
        raise PipelineError(f"Missing required commands: {', '.join(missing)}")


def probe_video(path: Path) -> dict[str, Any]:
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration,size:stream=index,codec_type,codec_name,width,height,r_frame_rate,sample_rate,channels",
            "-of",
            "json",
            str(path),
        ],
        capture=True,
    )
    payload = json.loads(result.stdout)
    duration = float(payload.get("format", {}).get("duration", 0))
    audio_streams = [
        stream for stream in payload.get("streams", []) if stream.get("codec_type") == "audio"
    ]
    if duration <= 0:
        raise PipelineError(f"Could not determine video duration: {path}")
    if not audio_streams:
        raise PipelineError(f"Video has no audio stream: {path}")
    return {"duration": duration, "probe": payload, "audio_streams": audio_streams}


def load_env_file(path: Path | None) -> dict[str, str]:
    if path is None or not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, value = line.partition("=")
        if separator:
            values[name.strip()] = value.strip().strip('"').strip("'")
    return values


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.is_file():
        raise PipelineError(f"Config file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PipelineError(f"Config file is not valid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PipelineError("Config must be a JSON object")
    return payload


def env_value(name: str, env_file: dict[str, str], default: str = "") -> str:
    return os.environ.get(name) or env_file.get(name) or default


def is_local_url(url: str) -> bool:
    hostname = (urlparse(url).hostname or "").lower()
    return hostname in {"localhost", "127.0.0.1", "::1"}


def extract_audio_window(
    video: Path,
    *,
    start: float,
    end: float,
    destination: Path,
    audio_stream: int,
) -> None:
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{end - start:.3f}",
            "-i",
            str(video),
            "-map",
            f"0:a:{audio_stream}",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            "-y",
            str(destination),
        ]
    )


def transcribe_openai_compatible(
    audio: Path,
    *,
    url: str,
    model: str,
    api_key: str,
) -> str:
    command = [
        "curl",
        "--fail-with-body",
        "--silent",
        "--show-error",
        "--retry",
        "2",
        "--retry-all-errors",
        "--max-time",
        "180",
        url,
    ]
    if api_key:
        command.extend(["-H", f"Authorization: Bearer {api_key}"])
    command.extend(["-F", f"model={model}", "-F", f"file=@{audio};type=audio/wav"])
    result = run(command, capture=True)
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise PipelineError(f"ASR returned invalid JSON: {result.stdout[:500]}") from exc
    if payload.get("error"):
        raise PipelineError(f"ASR error: {payload['error']}")
    return normalize_text(payload.get("text"))


def format_command(template: str, *, audio: Path, model: str) -> list[str]:
    try:
        return [part.format(audio=str(audio), model=model) for part in shlex.split(template)]
    except (KeyError, ValueError) as exc:
        raise PipelineError(
            "Invalid command template; supported placeholders are {audio} and {model}"
        ) from exc


def transcribe_command(audio: Path, *, template: str, model: str) -> str:
    result = run(format_command(template, audio=audio, model=model), capture=True)
    raw = result.stdout.strip()
    if not raw:
        return ""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return normalize_text(raw)
    if isinstance(payload, dict):
        return normalize_text(payload.get("text", payload.get("transcript", "")))
    return normalize_text(raw)


def transcribe_missing_segments(
    video: Path,
    segments: Sequence[Segment],
    *,
    duration: float,
    workdir: Path,
    audio_stream: int,
    overlap: float,
    workers: int,
    transcriber: Callable[[Path], str],
) -> None:
    missing = [segment for segment in segments if not segment.text]

    def transcribe_one(index: int, segment: Segment) -> tuple[Segment, str, float, float]:
        start = max(0.0, segment.start - overlap)
        end = min(duration, segment.end + overlap)
        audio = workdir / f"segment-{index:04d}.wav"
        extract_audio_window(
            video,
            start=start,
            end=end,
            destination=audio,
            audio_stream=audio_stream,
        )
        return segment, transcriber(audio), start, end

    def apply(result: tuple[Segment, str, float, float]) -> None:
        segment, text, start, end = result
        segment.raw_asr_text = text
        segment.asr_audio_start = start
        segment.asr_audio_end = end
        if overlap == 0:
            segment.text = text

    if workers == 1:
        for index, segment in enumerate(missing, 1):
            print(f"ASR {index}/{len(missing)}: segment {segment.id}", flush=True)
            apply(transcribe_one(index, segment))
        return

    print(
        f"ASR: {len(missing)} segment(s), {workers} worker(s), "
        f"{overlap:.2f}s context on each available side",
        flush=True,
    )
    failures: list[tuple[int, Segment, Exception]] = []
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(transcribe_one, index, segment): (index, segment)
            for index, segment in enumerate(missing, 1)
        }
        for future in concurrent.futures.as_completed(futures):
            index, segment = futures[future]
            try:
                apply(future.result())
                completed += 1
                print(f"ASR completed {completed}/{len(missing)}: segment {segment.id}", flush=True)
            except Exception as exc:
                failures.append((index, segment, exc))
                print(
                    f"ASR failed for segment {segment.id}; queued for sequential retry",
                    file=sys.stderr,
                    flush=True,
                )
    for retry_number, (index, segment, original_error) in enumerate(failures, 1):
        try:
            apply(transcribe_one(index, segment))
            print(
                f"ASR sequential retry {retry_number}/{len(failures)} passed: {segment.id}",
                flush=True,
            )
        except Exception as retry_error:
            raise PipelineError(
                f"ASR failed twice for segment {segment.id}. First: {original_error}. "
                f"Retry: {retry_error}"
            ) from retry_error


def normalized_word(token: str) -> str:
    return re.sub(r"[^\w']+", "", token, flags=re.UNICODE).casefold()


def longest_token_overlap(left: str, right: str, maximum: int = 40) -> int:
    left_tokens = left.split()
    right_tokens = right.split()
    limit = min(len(left_tokens), len(right_tokens), maximum)
    left_norm = [normalized_word(token) for token in left_tokens]
    right_norm = [normalized_word(token) for token in right_tokens]
    for count in range(limit, 0, -1):
        if left_norm[-count:] == right_norm[:count] and any(left_norm[-count:]):
            return count
    return 0


def stitch_deterministic(segments: Sequence[Segment]) -> None:
    previous = ""
    for segment in segments:
        if segment.asr_audio_start is None:
            previous = segment.text
            continue
        raw = segment.raw_asr_text
        if not raw:
            segment.text = ""
            continue
        tokens = raw.split()
        shared = longest_token_overlap(previous, raw) if previous else 0
        segment.text = normalize_text(" ".join(tokens[shared:]))
        previous = raw


def json_from_text(raw: str) -> Any:
    stripped = raw.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.I)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(stripped[start : end + 1])
            except json.JSONDecodeError:
                pass
    raise PipelineError(f"LLM response was not valid JSON: {stripped[:800]}")


def object_array_schema(
    ids: Sequence[str],
    *,
    value_name: str,
    value_schema: dict[str, Any],
) -> dict[str, Any]:
    count = len(ids)
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["segments"],
        "properties": {
            "segments": {
                "type": "array",
                "minItems": count,
                "maxItems": count,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", value_name],
                    "properties": {
                        "id": {"type": "string", "enum": list(ids)},
                        value_name: value_schema,
                    },
                },
            }
        },
    }


def parse_exact_items(
    payload: Any,
    ids: Sequence[str],
    value_name: str,
) -> dict[str, Any]:
    items = payload.get("segments") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        raise PipelineError("LLM response has no segments array")
    result: dict[str, Any] = {}
    for item in items:
        if not isinstance(item, dict):
            raise PipelineError("LLM returned a non-object segment")
        identifier = str(item.get("id", ""))
        if identifier in result:
            raise PipelineError(f"LLM returned duplicate segment id: {identifier}")
        result[identifier] = item.get(value_name)
    if set(result) != set(ids):
        missing = sorted(set(ids) - set(result))
        extra = sorted(set(result) - set(ids))
        raise PipelineError(f"LLM ids mismatch; missing={missing}, extra={extra}")
    return result


def run_llm_json(
    prompt: str,
    *,
    provider: str,
    model: str,
    effort: str,
    command_template: str | None,
    schema: dict[str, Any],
    workdir: Path,
    label: str,
    validator: Callable[[Any], dict[str, Any]],
) -> dict[str, Any]:
    last_error: PipelineError | None = None
    for attempt in range(1, 4):
        attempt_prompt = prompt
        if last_error:
            attempt_prompt += (
                "\n\nThe previous response failed validation: "
                f"{last_error}. Correct that error and return only valid JSON."
            )
        print(f"{label}, attempt {attempt}/3 with {provider}:{model}", flush=True)
        try:
            if provider == "codex":
                require_commands(["codex"])
                schema_path = workdir / f"{label.replace(' ', '-')}-{attempt}.schema.json"
                output_path = workdir / f"{label.replace(' ', '-')}-{attempt}.json"
                schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
                run(
                    [
                        "codex",
                        "exec",
                        "--model",
                        model,
                        "--sandbox",
                        "read-only",
                        "--ephemeral",
                        "--skip-git-repo-check",
                        "--color",
                        "never",
                        "--config",
                        f'model_reasoning_effort="{effort}"',
                        "--output-schema",
                        str(schema_path),
                        "--output-last-message",
                        str(output_path),
                        "-",
                    ],
                    input_text=attempt_prompt,
                    cwd=workdir,
                )
                payload = json_from_text(output_path.read_text(encoding="utf-8"))
            elif provider == "command":
                if not command_template:
                    raise PipelineError("--llm-command is required for command LLM provider")
                command = [part.format(model=model) for part in shlex.split(command_template)]
                payload = json_from_text(run(command, capture=True, input_text=attempt_prompt).stdout)
            else:
                raise PipelineError(f"Unsupported LLM provider: {provider}")
            return validator(payload)
        except PipelineError as exc:
            last_error = exc
            if attempt < 3:
                print(f"{label} failed validation; retrying: {exc}", flush=True)
    assert last_error is not None
    raise last_error


def stitch_with_llm(
    segments: Sequence[Segment],
    *,
    provider: str,
    model: str,
    effort: str,
    command_template: str | None,
    batch_size: int,
    workdir: Path,
) -> None:
    target_indices = [
        index for index, segment in enumerate(segments) if segment.asr_audio_start is not None
    ]
    for batch_number, offset in enumerate(range(0, len(target_indices), batch_size), 1):
        indexes = target_indices[offset : offset + batch_size]
        targets = [segments[index] for index in indexes]
        context_start = max(0, indexes[0] - 1)
        context_end = min(len(segments), indexes[-1] + 2)
        context = segments[context_start:context_end]
        ids = [segment.id for segment in targets]
        compact = [
            {
                "id": segment.id,
                "nominal_start": segment.start,
                "nominal_end": segment.end,
                "audio_start": segment.asr_audio_start,
                "audio_end": segment.asr_audio_end,
                "raw_text": segment.raw_asr_text or segment.text,
                "output_required": segment.id in ids,
            }
            for segment in context
        ]
        prompt = textwrap.dedent(
            f"""
            Reconcile overlapping ASR windows into one source transcript per nominal subtitle anchor.

            Rules:
            - Return only JSON matching the supplied schema.
            - Return every output_required id exactly once and in its input order.
            - The audio windows overlap, so boundary speech may appear in two raw transcripts.
            - Assign every spoken phrase exactly once; do not duplicate overlap text.
            - Repair a word cut at a nominal boundary only when neighboring raw transcripts support it.
            - Preserve technical terms, names, numbers, units, uncertainty, and spoken meaning.
            - Do not translate, summarize, improve grammar, or invent inaudible content.
            - Use neighboring non-output records only as context.
            - An output may be empty when its nominal interval is silence.

            Windows:
            {json.dumps(compact, ensure_ascii=False, separators=(',', ':'))}
            """
        ).strip()
        schema = object_array_schema(
            ids,
            value_name="text",
            value_schema={"type": "string"},
        )
        result = run_llm_json(
            prompt,
            provider=provider,
            model=model,
            effort=effort,
            command_template=command_template,
            schema=schema,
            workdir=workdir,
            label=f"stitch-batch-{batch_number}",
            validator=lambda payload, expected=ids: parse_exact_items(payload, expected, "text"),
        )
        for segment in targets:
            segment.text = normalize_text(result[segment.id])


def stitch_transcripts(
    segments: Sequence[Segment],
    *,
    provider: str,
    model: str,
    effort: str,
    command_template: str | None,
    batch_size: int,
    workdir: Path,
) -> None:
    if provider == "none":
        for segment in segments:
            if segment.asr_audio_start is not None:
                segment.text = segment.raw_asr_text
    elif provider == "deterministic":
        stitch_deterministic(segments)
    else:
        stitch_with_llm(
            segments,
            provider=provider,
            model=model,
            effort=effort,
            command_template=command_template,
            batch_size=batch_size,
            workdir=workdir,
        )


def drop_short_trailing_fragment(
    segments: Sequence[Segment], minimum_characters: int
) -> bool:
    if not segments or minimum_characters <= 0:
        return False
    tail = segments[-1]
    character_count = len(re.findall(r"\w", tail.text, flags=re.UNICODE))
    if tail.duration < 1.0 and 0 < character_count < minimum_characters:
        tail.text = ""
        tail.translation_parts = []
        return True
    return False


def merge_translation_file(segments: Sequence[Segment], path: Path) -> None:
    translated = load_segments(path)
    if len(translated) != len(segments):
        raise PipelineError(
            f"Translation file has {len(translated)} segments; expected {len(segments)}"
        )
    for source, target in zip(segments, translated, strict=True):
        if abs(source.start - target.start) > 0.05 or abs(source.end - target.end) > 0.05:
            raise PipelineError(f"Translation timing mismatch at segment {source.id}")
        source.translation_parts = target.translation_parts or ([target.text] if target.text else [])


def apply_translation_overrides(segments: Sequence[Segment], values: Sequence[str]) -> None:
    by_id = {segment.id: segment for segment in segments}
    for value in values:
        key, separator, text = value.partition("=")
        if not separator:
            raise PipelineError(f"Translated override must be ID=TEXT: {value!r}")
        segment = by_id.get(key)
        if segment is None and key.isdigit() and 1 <= int(key) <= len(segments):
            segment = segments[int(key) - 1]
        if segment is None:
            raise PipelineError(f"Unknown translated segment: {key}")
        segment.translation_parts = normalize_parts(text)


def translate_segments(
    segments: Sequence[Segment],
    *,
    source_language: str,
    target_language: str,
    provider: str,
    model: str,
    effort: str,
    command_template: str | None,
    batch_size: int,
    max_parts: int,
    workdir: Path,
) -> None:
    missing = [segment for segment in segments if segment.text and not segment.translation_parts]
    for batch_number, offset in enumerate(range(0, len(missing), batch_size), 1):
        batch = missing[offset : offset + batch_size]
        ids = [segment.id for segment in batch]
        compact = [
            {"id": segment.id, "start": segment.start, "end": segment.end, "text": segment.text}
            for segment in batch
        ]
        prompt = textwrap.dedent(
            f"""
            Translate subtitle segments from {source_language} to {target_language}.

            Rules:
            - Return only JSON matching the supplied schema.
            - Preserve every id exactly once and in input order.
            - Preserve technical terms, product names, acronyms, formulas, numbers, and units.
            - Translate accurately without adding explanations or facts.
            - Use 1 to {max_parts} concise chronological phrase parts per segment.
            - Do not call tools or modify files.

            Segments:
            {json.dumps(compact, ensure_ascii=False, separators=(',', ':'))}
            """
        ).strip()
        schema = object_array_schema(
            ids,
            value_name="parts",
            value_schema={
                "type": "array",
                "minItems": 1,
                "maxItems": max_parts,
                "items": {"type": "string", "minLength": 1},
            },
        )
        result = run_llm_json(
            prompt,
            provider=provider,
            model=model,
            effort=effort,
            command_template=command_template,
            schema=schema,
            workdir=workdir,
            label=f"translate-batch-{batch_number}",
            validator=lambda payload, expected=ids: parse_exact_items(payload, expected, "parts"),
        )
        for segment in batch:
            segment.translation_parts = normalize_parts(result[segment.id])


def split_text_into_count(text: str, count: int) -> list[str]:
    if count <= 1:
        return [normalize_text(text)]
    words = normalize_text(text).split()
    if not words:
        return [""] * count
    return [
        " ".join(words[round(index * len(words) / count) : round((index + 1) * len(words) / count)])
        for index in range(count)
    ]


def choose_part_boundaries(start: float, end: float, count: int) -> list[float]:
    return [start + (end - start) * index / count for index in range(count + 1)]


def build_cues(
    segments: Sequence[Segment],
    *,
    use_source_text: bool,
    include_source: bool,
    max_parts: int,
) -> list[Cue]:
    cues: list[Cue] = []
    for segment in segments:
        if segment.translation_parts:
            parts = segment.translation_parts[:max_parts]
        elif use_source_text:
            parts = [segment.text] if segment.text else []
        else:
            parts = []
        if not parts:
            continue
        source_parts = split_text_into_count(segment.text, len(parts))
        boundaries = choose_part_boundaries(segment.start, segment.end, len(parts))
        for index, part in enumerate(parts):
            text = part
            if include_source:
                text = f"{source_parts[index]}\n{part}" if source_parts[index] else part
            cues.append(Cue(boundaries[index], boundaries[index + 1], text, segment.id))
    return cues


def wrap_line(text: str, maximum: int) -> str:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > maximum:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines)


def wrap_subtitle(text: str, maximum: int) -> str:
    return "\n".join(wrap_line(line, maximum) for line in text.splitlines())


def write_srt(cues: Sequence[Cue], path: Path, maximum: int) -> None:
    blocks = [
        f"{index}\n{format_srt_time(cue.start)} --> {format_srt_time(cue.end)}\n"
        f"{wrap_subtitle(cue.text, maximum)}"
        for index, cue in enumerate(cues, 1)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def language_code(target: str, requested: str | None) -> str:
    return requested or LANGUAGE_CODES.get(target.casefold(), target[:2].casefold())


def subtitle_font(target: str, code: str, requested: str | None) -> str:
    if requested:
        return requested
    base = code.casefold().split("-", 1)[0]
    return LANGUAGE_FONTS.get(base, "Noto Sans")


def ffmpeg_escape(path: Path) -> str:
    return str(path.resolve()).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def burn_subtitles(
    video: Path,
    srt: Path,
    output: Path,
    *,
    audio_stream: int,
    font: str,
    font_size: int,
    crf: int,
    preset: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    style = (
        f"FontName={font},FontSize={font_size},PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=0,"
        "Alignment=2,MarginV=34"
    )
    subtitle_filter = f"subtitles='{ffmpeg_escape(srt)}':force_style='{style}'"
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-stats_period",
            "10",
            "-progress",
            "pipe:1",
            "-nostats",
            "-y",
            "-i",
            str(video),
            "-map",
            "0:v:0",
            "-map",
            f"0:a:{audio_stream}",
            "-vf",
            subtitle_filter,
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-profile:v",
            "high",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )


def write_manifest(
    path: Path,
    *,
    args: argparse.Namespace,
    video_info: dict[str, Any],
    segments: Sequence[Segment],
    cues: Sequence[Cue],
    output_srt: Path,
    output_video: Path | None,
) -> None:
    payload = {
        "pipeline_version": VERSION,
        "source_video": str(args.video.resolve()),
        "duration": video_info["duration"],
        "anchors": str(args.anchors.resolve()) if args.anchors else None,
        "anchor_seconds": args.anchor_seconds if not args.anchors else None,
        "audio_overlap_seconds": args.audio_overlap,
        "asr": {
            "provider": args.asr_provider,
            "model": args.asr_model,
            "endpoint_configured": (
                bool(args.asr_url) if args.asr_provider == "openai-compatible" else False
            ),
            "workers": args.asr_workers,
        },
        "stitching": {
            "provider": args.stitch_provider,
            "model": args.stitch_model if args.stitch_provider in {"codex", "command"} else None,
            "note": "Raw overlapping ASR text and extraction bounds are retained per segment.",
        },
        "translation": {
            "provider": args.translate_provider,
            "model": args.translation_model if args.translate_provider != "none" else None,
            "source_language": args.source_language,
            "target_language": args.target_language,
        },
        "include_source": args.include_source,
        "outputs": {
            "srt": str(output_srt.resolve()),
            "video": str(output_video.resolve()) if output_video else None,
        },
        "segments": [asdict(segment) for segment in segments],
        "cues": [asdict(cue) for cue in cues],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    implicit_config = os.environ.get("VIDEO_SUBTITLE_CONFIG")
    if not implicit_config and Path("config.local.json").is_file():
        implicit_config = "config.local.json"
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=Path(implicit_config) if implicit_config else None)
    preliminary, _ = pre_parser.parse_known_args(argv)
    config = load_config(preliminary.config.expanduser() if preliminary.config else None)

    parser = argparse.ArgumentParser(
        description="Overlap-aware video transcription, translation, and subtitles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=preliminary.config,
        help="JSON defaults; CLI arguments override config values",
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("--anchors", type=Path, help="Existing source SRT or JSON anchors")
    parser.add_argument("--anchor-seconds", type=float, default=5.0)
    parser.add_argument(
        "--minimum-anchor-seconds",
        type=float,
        default=1.0,
        help="Merge a shorter generated tail into the preceding anchor",
    )
    parser.add_argument("--audio-overlap", type=float, default=1.0, help="ASR context added before and after each nominal anchor")
    parser.add_argument("--audio-stream", type=int, default=0)
    parser.add_argument("--source-language", default="English")
    parser.add_argument("--target-language", default="Korean")
    parser.add_argument("--language-code")

    parser.add_argument("--asr-provider", choices=["openai-compatible", "command"], default="openai-compatible")
    parser.add_argument("--asr-url", default=os.environ.get("ASR_URL", ""))
    parser.add_argument("--asr-model", default=os.environ.get("ASR_MODEL", DEFAULT_ASR_MODEL))
    parser.add_argument("--asr-api-key-env", default="ASR_API_KEY")
    parser.add_argument("--asr-command", help="Local command template; supports {audio} and {model}")
    parser.add_argument("--asr-workers", type=int, default=4)
    parser.add_argument("--allow-audio-upload", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--allow-empty-text", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))

    parser.add_argument("--stitch-provider", choices=["codex", "command", "deterministic", "none"], default="codex")
    parser.add_argument("--stitch-model", default=os.environ.get("STITCH_MODEL", "gpt-5.6-luna"))
    parser.add_argument("--stitch-batch-size", type=int, default=40)

    parser.add_argument("--translation-file", type=Path)
    parser.add_argument("--translated-segment", action="append", default=[], metavar="ID=TEXT")
    parser.add_argument("--translate-provider", choices=["none", "codex", "command"], default="none")
    parser.add_argument("--translation-model", default=os.environ.get("TRANSLATION_MODEL", "gpt-5.6-luna"))
    parser.add_argument("--translation-batch-size", type=int, default=40)
    parser.add_argument("--llm-command", help="LLM command receiving a prompt on stdin; supports {model}")
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high", "xhigh", "max"], default="low")
    parser.add_argument("--max-parts", type=int, default=1)
    parser.add_argument(
        "--minimum-tail-text-chars",
        type=int,
        default=3,
        help="Drop a sub-second final ASR fragment with fewer alphanumeric characters; 0 disables",
    )
    parser.add_argument("--use-source-text", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-source", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--output-srt", type=Path)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--burn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-video", type=Path)
    parser.add_argument("--font")
    parser.add_argument("--font-size", type=int, default=20)
    parser.add_argument("--max-line-chars", type=int, default=44)
    parser.add_argument("--crf", type=int, default=19)
    parser.add_argument("--preset", default="fast")
    parser.add_argument("--keep-workdir", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)

    valid_keys = {action.dest for action in parser._actions if action.dest not in {"help", "video", "config"}}
    unknown = sorted(set(config) - valid_keys)
    if unknown:
        raise PipelineError(f"Unknown config key(s): {', '.join(unknown)}")
    parser.set_defaults(**config)
    return parser.parse_args(argv)


def execute(args: argparse.Namespace) -> int:
    args.video = args.video.expanduser()
    args.config = Path(args.config).expanduser() if args.config else None
    args.anchors = Path(args.anchors).expanduser() if args.anchors else None
    args.translation_file = Path(args.translation_file).expanduser() if args.translation_file else None
    args.env_file = Path(args.env_file).expanduser() if args.env_file else None
    args.output_srt = Path(args.output_srt).expanduser() if args.output_srt else None
    args.output_manifest = Path(args.output_manifest).expanduser() if args.output_manifest else None
    args.output_video = Path(args.output_video).expanduser() if args.output_video else None
    if not args.video.is_file():
        raise PipelineError(f"Video does not exist: {args.video}")
    if args.anchors and not args.anchors.is_file():
        raise PipelineError(f"Anchors do not exist: {args.anchors}")
    if args.translation_file and not args.translation_file.is_file():
        raise PipelineError(f"Translation file does not exist: {args.translation_file}")
    positive = [
        args.anchor_seconds,
        args.minimum_anchor_seconds,
        args.asr_workers,
        args.stitch_batch_size,
        args.translation_batch_size,
        args.max_parts,
    ]
    if (
        any(value <= 0 for value in positive)
        or args.audio_overlap < 0
        or args.minimum_tail_text_chars < 0
    ):
        raise PipelineError("Durations, worker counts, batch sizes, and part counts must be positive; overlap may be zero")

    require_commands(["ffmpeg", "ffprobe"])
    video_info = probe_video(args.video)
    if args.audio_stream >= len(video_info["audio_streams"]):
        raise PipelineError(
            f"Audio stream {args.audio_stream} is unavailable; found {len(video_info['audio_streams'])}"
        )
    segments = load_segments(args.anchors) if args.anchors else fixed_segments(
        video_info["duration"], args.anchor_seconds, args.minimum_anchor_seconds
    )
    if segments[-1].end > video_info["duration"] + 0.25:
        raise PipelineError("The final anchor extends beyond the video")
    if args.translation_file:
        merge_translation_file(segments, args.translation_file)
    apply_translation_overrides(segments, args.translated_segment)

    missing = [segment for segment in segments if not segment.text]
    if args.anchors and args.allow_empty_text:
        missing = []
    if args.dry_run:
        print(
            json.dumps(
                {
                    "video": str(args.video),
                    "duration": video_info["duration"],
                    "nominal_segments": len(segments),
                    "segments_needing_asr": len(missing),
                    "audio_overlap_seconds": args.audio_overlap,
                    "asr_provider": args.asr_provider,
                    "stitch_provider": args.stitch_provider,
                    "translation_provider": args.translate_provider,
                    "burn": args.burn,
                },
                indent=2,
            )
        )
        return 0

    env_file = load_env_file(args.env_file)
    temporary: tempfile.TemporaryDirectory[str] | None
    if args.keep_workdir:
        temporary = None
        workdir = Path(tempfile.mkdtemp(prefix="video-subtitle-pipeline-"))
    else:
        temporary = tempfile.TemporaryDirectory(prefix="video-subtitle-pipeline-")
        workdir = Path(temporary.name)
    try:
        if missing:
            if args.asr_provider == "openai-compatible":
                require_commands(["curl"])
                args.asr_url = args.asr_url or env_value("ASR_URL", env_file)
                if not args.asr_url:
                    raise PipelineError("--asr-url or ASR_URL is required for the HTTP ASR provider")
                if not is_local_url(args.asr_url) and not args.allow_audio_upload:
                    raise PipelineError(
                        "Remote ASR requires --allow-audio-upload. Only extracted WAV windows are sent."
                    )
                api_key = env_value(args.asr_api_key_env, env_file)
                transcriber = lambda audio: transcribe_openai_compatible(
                    audio,
                    url=args.asr_url,
                    model=args.asr_model,
                    api_key=api_key,
                )
            else:
                if not args.asr_command:
                    raise PipelineError("--asr-command is required for command ASR")
                command_template = args.asr_command
                transcriber = lambda audio: transcribe_command(
                    audio, template=command_template, model=args.asr_model
                )
            transcribe_missing_segments(
                args.video,
                segments,
                duration=video_info["duration"],
                workdir=workdir,
                audio_stream=args.audio_stream,
                overlap=args.audio_overlap,
                workers=args.asr_workers,
                transcriber=transcriber,
            )
            if args.audio_overlap > 0:
                stitch_transcripts(
                    segments,
                    provider=args.stitch_provider,
                    model=args.stitch_model,
                    effort=args.reasoning_effort,
                    command_template=args.llm_command,
                    batch_size=args.stitch_batch_size,
                    workdir=workdir,
                )

            tail_text = segments[-1].text
            if drop_short_trailing_fragment(segments, args.minimum_tail_text_chars):
                print(
                    f"Dropping short trailing ASR fragment from segment "
                    f"{segments[-1].id}: {tail_text!r}",
                    flush=True,
                )

        if args.translate_provider != "none":
            translate_segments(
                segments,
                source_language=args.source_language,
                target_language=args.target_language,
                provider=args.translate_provider,
                model=args.translation_model,
                effort=args.reasoning_effort,
                command_template=args.llm_command,
                batch_size=args.translation_batch_size,
                max_parts=args.max_parts,
                workdir=workdir,
            )

        code = language_code(args.target_language, args.language_code)
        cues = build_cues(
            segments,
            use_source_text=args.use_source_text,
            include_source=args.include_source,
            max_parts=args.max_parts,
        )
        if not cues:
            raise PipelineError(
                "No subtitle cues were produced; enable translation, provide translations, or use --use-source-text"
            )
        output_srt = args.output_srt or args.video.with_name(f"{args.video.stem}.{code}.srt")
        output_manifest = args.output_manifest or args.video.with_name(
            f"{args.video.stem}.{code}.segments.json"
        )
        output_video = (
            args.output_video or args.video.with_name(f"{args.video.stem}.{code}.hardsub.mp4")
            if args.burn
            else None
        )
        write_srt(cues, output_srt, args.max_line_chars)
        write_manifest(
            output_manifest,
            args=args,
            video_info=video_info,
            segments=segments,
            cues=cues,
            output_srt=output_srt,
            output_video=output_video,
        )
        print(f"Wrote subtitles: {output_srt}")
        print(f"Wrote manifest: {output_manifest}")
        if output_video:
            burn_subtitles(
                args.video,
                output_srt,
                output_video,
                audio_stream=args.audio_stream,
                font=subtitle_font(args.target_language, code, args.font),
                font_size=args.font_size,
                crf=args.crf,
                preset=args.preset,
            )
            print(f"Wrote video: {output_video}")
        if args.keep_workdir:
            print(f"Kept work directory: {workdir}")
        return 0
    finally:
        if temporary is not None:
            temporary.cleanup()
