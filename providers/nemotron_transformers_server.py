#!/usr/bin/env python3
"""OpenAI-compatible Nemotron 3.5 ASR server with native word timestamps.

This optional provider requires current Transformers, PyTorch, FastAPI,
Uvicorn, NumPy, SoundFile, and FFmpeg. Configuration is supplied through
environment variables so private paths and deployment details stay local.
"""

from __future__ import annotations

import io
import gc
import os
import re
import subprocess
import tempfile
import threading
import time
import types
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

try:
    from .nemotron_runtime_policy import choose_runtime
except ImportError:
    from nemotron_runtime_policy import choose_runtime


MODEL_DIR = os.environ.get(
    "NEMOTRON_MODEL_DIR", "nvidia/nemotron-3.5-asr-streaming-0.6b"
)
SERVED_NAME = os.environ.get("NEMOTRON_SERVED_NAME", "nemotron-3.5-asr")
LOOKAHEAD_TOKENS = int(os.environ.get("NEMOTRON_LOOKAHEAD_TOKENS", "13"))
DEFAULT_RUNTIME = os.environ.get("NEMOTRON_RUNTIME", "auto")
AUTO_MAX_OFFLINE_SECONDS = float(
    os.environ.get("NEMOTRON_AUTO_MAX_OFFLINE_SECONDS", "900")
)
MEMORY_RESERVE_GB = float(os.environ.get("NEMOTRON_MEMORY_RESERVE_GB", "1"))
OFFLINE_FIXED_MIB = float(os.environ.get("NEMOTRON_OFFLINE_FIXED_MIB", "256"))
OFFLINE_MIB_PER_SECOND = float(
    os.environ.get("NEMOTRON_OFFLINE_MIB_PER_SECOND", "12")
)
RELEASE_OFFLINE_CACHE = os.environ.get(
    "NEMOTRON_RELEASE_OFFLINE_CACHE", "true"
).casefold() not in {"0", "false", "no"}

app = FastAPI(title="Nemotron 3.5 timestamped ASR")
_model = None
_processor = None
_inference_lock = threading.Lock()


def get_runtime():
    global _model, _processor
    if _model is None or _processor is None:
        from transformers import AutoModelForRNNT, AutoProcessor

        _processor = AutoProcessor.from_pretrained(MODEL_DIR)
        _processor.set_num_lookahead_tokens(LOOKAHEAD_TOKENS)
        _model = AutoModelForRNNT.from_pretrained(MODEL_DIR, dtype=torch.float16)
        _model = _model.to("cuda").eval()
    return _model, _processor


def decode_upload(raw: bytes) -> tuple[np.ndarray, int]:
    try:
        audio, sample_rate = sf.read(io.BytesIO(raw), dtype="float32")
    except Exception:
        source_path: Path | None = None
        wav_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as source:
                source.write(raw)
                source_path = Path(source.name)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav:
                wav_path = Path(wav.name)
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(source_path),
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    str(wav_path),
                ],
                check=True,
                capture_output=True,
            )
            audio, sample_rate = sf.read(wav_path, dtype="float32")
        finally:
            if source_path:
                source_path.unlink(missing_ok=True)
            if wav_path:
                wav_path.unlink(missing_ok=True)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != 16000:
        import librosa

        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000
    return np.asarray(audio, dtype=np.float32), sample_rate


def streaming_inputs(audio: np.ndarray, sample_rate: int, language: str):
    model, processor = get_runtime()
    first_count = processor.num_samples_first_audio_chunk
    first_audio = audio[:first_count]
    if len(first_audio) < first_count:
        first_audio = np.pad(first_audio, (0, first_count - len(first_audio)))
    first = processor(
        first_audio,
        sampling_rate=sample_rate,
        is_streaming=True,
        is_first_audio_chunk=True,
        language=language,
        return_tensors="pt",
    ).to(model.device, dtype=model.dtype)

    def feature_generator():
        yield first.input_features[:, : processor.num_mel_frames_first_audio_chunk, :]
        mel_frame = processor.num_mel_frames_first_audio_chunk
        hop_length = processor.feature_extractor.hop_length
        n_fft = processor.feature_extractor.n_fft
        start = mel_frame * hop_length - n_fft // 2
        while start < len(audio):
            end = start + processor.num_samples_per_audio_chunk
            chunk = audio[start:end]
            if len(chunk) < processor.num_samples_per_audio_chunk:
                chunk = np.pad(chunk, (0, processor.num_samples_per_audio_chunk - len(chunk)))
            inputs = processor(
                chunk,
                sampling_rate=sample_rate,
                is_streaming=True,
                is_first_audio_chunk=False,
                language=language,
                return_tensors="pt",
            ).to(model.device, dtype=model.dtype)
            yield inputs.input_features
            mel_frame += processor.num_mel_frames_per_audio_chunk
            start = mel_frame * hop_length - n_fft // 2

    inputs = dict(first)
    inputs["input_features"] = feature_generator()
    inputs["num_lookahead_tokens"] = LOOKAHEAD_TOKENS
    return inputs


def offline_inputs(audio: np.ndarray, sample_rate: int, language: str):
    model, processor = get_runtime()
    return processor(
        audio,
        sampling_rate=sample_rate,
        language=language,
        return_tensors="pt",
    ).to(model.device, dtype=model.dtype)


def generate_with_confidence(model, inputs: dict, enabled: bool):
    if not enabled:
        with torch.inference_mode():
            return model.generate(**inputs), None

    step_confidences: list[float] = []
    original_update = model._update_model_kwargs_for_generation

    def capture_update(self, outputs, model_kwargs, *args, **kwargs):
        probabilities = outputs.logits[:, -1, :].detach().float().softmax(dim=-1)
        step_confidences.append(float(probabilities.max(dim=-1).values[0].cpu()))
        return original_update(outputs, model_kwargs, *args, **kwargs)

    model._update_model_kwargs_for_generation = types.MethodType(capture_update, model)
    try:
        with torch.inference_mode():
            output = model.generate(**inputs)
    finally:
        model._update_model_kwargs_for_generation = original_update
    return output, step_confidences


def emitted_token_confidences(model, processor, output, step_confidences):
    if step_confidences is None:
        return None
    sequence = output.sequences[0].detach().cpu().tolist()[1:]
    if len(sequence) != len(step_confidences):
        raise RuntimeError("RNNT confidence steps did not align with generated tokens")
    ignored = set(processor.tokenizer.all_special_ids) | {model.config.blank_token_id}
    return [
        confidence
        for token_id, confidence in zip(sequence, step_confidences, strict=True)
        if token_id not in ignored
    ]


def words_from_tokens(
    token_timestamps: list[dict],
    duration: float,
    token_confidences: list[float] | None = None,
) -> list[dict]:
    if token_confidences is not None and len(token_confidences) != len(token_timestamps):
        raise RuntimeError("RNNT confidence values did not align with timestamped emissions")
    text = ""
    character_times: list[tuple[float, float]] = []
    character_confidences: list[float | None] = []
    for index, item in enumerate(token_timestamps):
        token = str(item.get("token", ""))
        start = max(0.0, float(item["start"]))
        end = min(duration, float(item["end"]))
        if not token or start >= duration or end <= start:
            continue
        text += token
        character_times.extend([(start, end)] * len(token))
        confidence = token_confidences[index] if token_confidences is not None else None
        character_confidences.extend([confidence] * len(token))

    words = []
    for match in re.finditer(r"\S+", text):
        times = character_times[match.start() : match.end()]
        word = {
            "word": match.group(0),
            "start": round(min(start for start, _ in times), 3),
            "end": round(max(end for _, end in times), 3),
        }
        confidences = [
            value
            for value in character_confidences[match.start() : match.end()]
            if value is not None
        ]
        if confidences:
            word["confidence"] = round(min(confidences), 6)
        words.append(word)
    return words


def transcribe_audio(
    audio: np.ndarray,
    sample_rate: int,
    language: str,
    include_confidence: bool,
    requested_runtime: str,
    memory_limit_gb: float | None,
    max_offline_seconds: float | None,
) -> dict:
    model, processor = get_runtime()
    duration = len(audio) / sample_rate
    free_gpu_bytes, _ = torch.cuda.mem_get_info(model.device)
    decision = choose_runtime(
        requested_runtime,
        duration_seconds=duration,
        free_gpu_bytes=free_gpu_bytes,
        memory_limit_gb=memory_limit_gb,
        max_offline_seconds=max_offline_seconds,
        auto_max_offline_seconds=AUTO_MAX_OFFLINE_SECONDS,
        reserve_gb=MEMORY_RESERVE_GB,
        offline_fixed_mib=OFFLINE_FIXED_MIB,
        offline_mib_per_second=OFFLINE_MIB_PER_SECOND,
    )
    actual_runtime = decision.selected
    fallback_reason = None
    allocated_before = torch.cuda.memory_allocated(model.device)
    reserved_before = torch.cuda.memory_reserved(model.device)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(model.device)
    try:
        inputs = (
            offline_inputs(audio, sample_rate, language)
            if actual_runtime == "offline"
            else streaming_inputs(audio, sample_rate, language)
        )
        output, step_confidences = generate_with_confidence(
            model, inputs, include_confidence
        )
    except torch.OutOfMemoryError:
        if actual_runtime != "offline":
            raise
        fallback_reason = "offline_cuda_out_of_memory"
        actual_runtime = "streaming"
        if "inputs" in locals():
            del inputs
        gc.collect()
        torch.cuda.empty_cache()
        inputs = streaming_inputs(audio, sample_rate, language)
        output, step_confidences = generate_with_confidence(
            model, inputs, include_confidence
        )
    elapsed = time.perf_counter() - started
    _, timestamps = processor.decode(
        output.sequences,
        durations=output.durations,
        skip_special_tokens=True,
    )
    token_timestamps = timestamps[0] if timestamps and isinstance(timestamps[0], list) else timestamps
    token_confidences = emitted_token_confidences(
        model, processor, output, step_confidences
    )
    words = words_from_tokens(token_timestamps, duration, token_confidences)
    text = " ".join(word["word"] for word in words).strip()
    peak_allocated = torch.cuda.max_memory_allocated(model.device)
    peak_reserved = torch.cuda.max_memory_reserved(model.device)
    runtime_metadata = {
        **decision.metadata(),
        "actual": actual_runtime,
        "fallback_reason": fallback_reason,
        "elapsed_seconds": round(elapsed, 4),
        "real_time_factor": round(elapsed / duration, 6) if duration else None,
        "allocated_before_mib": round(allocated_before / (1024**2), 1),
        "reserved_before_mib": round(reserved_before / (1024**2), 1),
        "peak_allocated_mib": round(peak_allocated / (1024**2), 1),
        "peak_reserved_mib": round(peak_reserved / (1024**2), 1),
        "lookahead_tokens": LOOKAHEAD_TOKENS if actual_runtime == "streaming" else None,
        "cache_released": actual_runtime == "offline" and RELEASE_OFFLINE_CACHE,
    }
    if actual_runtime == "offline" and RELEASE_OFFLINE_CACHE:
        del inputs, output
        gc.collect()
        torch.cuda.empty_cache()
    runtime_metadata["reserved_after_mib"] = round(
        torch.cuda.memory_reserved(model.device) / (1024**2), 1
    )
    return {
        "text": text,
        "words": words,
        "duration": duration,
        "runtime_metadata": runtime_metadata,
    }


@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [
            {
                "id": SERVED_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "transformers",
                "root": MODEL_DIR,
                "parent": None,
            }
        ],
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": SERVED_NAME,
        "loaded": _model is not None,
        "word_timestamps": True,
        "word_confidence": True,
        "lookahead_tokens": LOOKAHEAD_TOKENS,
        "runtime_default": DEFAULT_RUNTIME,
        "runtime_modes": ["auto", "throughput", "streaming"],
        "auto_max_offline_seconds": AUTO_MAX_OFFLINE_SECONDS,
        "memory_reserve_gb": MEMORY_RESERVE_GB,
        "offline_memory_estimate": {
            "fixed_mib": OFFLINE_FIXED_MIB,
            "mib_per_second": OFFLINE_MIB_PER_SECOND,
        },
        "release_offline_cache": RELEASE_OFFLINE_CACHE,
    }


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form(SERVED_NAME),
    language: str = Form("auto"),
    response_format: str = Form("json"),
    timestamps: bool = Form(False),
    confidence: bool = Form(False),
    runtime: str = Form(DEFAULT_RUNTIME),
    memory_limit_gb: float | None = Form(None),
    max_offline_seconds: float | None = Form(None),
    timestamp_granularities: list[str] = Form(
        default=[], alias="timestamp_granularities[]"
    ),
):
    try:
        audio, sample_rate = decode_upload(await file.read())
        with _inference_lock:
            result = transcribe_audio(
                audio,
                sample_rate,
                language or "auto",
                include_confidence=confidence,
                requested_runtime=runtime,
                memory_limit_gb=memory_limit_gb,
                max_offline_seconds=max_offline_seconds,
            )
    except ValueError as exc:
        return JSONResponse(
            {"error": {"message": str(exc), "type": "invalid_request"}},
            status_code=400,
        )
    except Exception as exc:
        return JSONResponse(
            {"error": {"message": str(exc), "type": "asr_error"}}, status_code=503
        )
    base = {
        "text": result["text"],
        "model": SERVED_NAME,
        "task": "transcribe",
        "format": file.filename.rsplit(".", 1)[-1] if file.filename else "wav",
    }
    verbose = response_format == "verbose_json" or timestamps or "word" in timestamp_granularities
    if not verbose:
        return base
    response = {
        **base,
        "duration": round(result["duration"], 3),
        "language": language or "auto",
        "words": result["words"],
        "runtime_metadata": result["runtime_metadata"],
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": round(result["duration"], 3),
                "text": result["text"],
            }
        ]
        if result["text"]
        else [],
    }
    if confidence:
        response["confidence_metadata"] = {
            "level": "word",
            "method": "rnnt_max_softmax",
            "aggregation": "min",
            "calibrated_probability": False,
        }
    return response


if __name__ == "__main__":
    import uvicorn

    get_runtime()
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "1239")))
