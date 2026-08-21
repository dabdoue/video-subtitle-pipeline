#!/usr/bin/env python3
"""OpenAI-compatible Nemotron 3.5 ASR server with native word timestamps.

This optional provider requires current Transformers, PyTorch, FastAPI,
Uvicorn, NumPy, SoundFile, and FFmpeg. Configuration is supplied through
environment variables so private paths and deployment details stay local.
"""

from __future__ import annotations

import io
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse


MODEL_DIR = os.environ.get(
    "NEMOTRON_MODEL_DIR", "nvidia/nemotron-3.5-asr-streaming-0.6b"
)
SERVED_NAME = os.environ.get("NEMOTRON_SERVED_NAME", "nemotron-3.5-asr")
LOOKAHEAD_TOKENS = int(os.environ.get("NEMOTRON_LOOKAHEAD_TOKENS", "13"))

app = FastAPI(title="Nemotron 3.5 timestamped ASR")
_model = None
_processor = None


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
    return inputs


def words_from_tokens(token_timestamps: list[dict], duration: float) -> list[dict]:
    text = ""
    character_times: list[tuple[float, float]] = []
    for item in token_timestamps:
        token = str(item.get("token", ""))
        start = max(0.0, float(item["start"]))
        end = min(duration, float(item["end"]))
        if not token or start >= duration or end <= start:
            continue
        text += token
        character_times.extend([(start, end)] * len(token))

    words = []
    for match in re.finditer(r"\S+", text):
        times = character_times[match.start() : match.end()]
        words.append(
            {
                "word": match.group(0),
                "start": round(min(start for start, _ in times), 3),
                "end": round(max(end for _, end in times), 3),
            }
        )
    return words


def transcribe_audio(audio: np.ndarray, sample_rate: int, language: str) -> dict:
    model, processor = get_runtime()
    duration = len(audio) / sample_rate
    inputs = streaming_inputs(audio, sample_rate, language)
    with torch.inference_mode():
        output = model.generate(**inputs)
    _, timestamps = processor.decode(
        output.sequences,
        durations=output.durations,
        skip_special_tokens=True,
    )
    token_timestamps = timestamps[0] if timestamps and isinstance(timestamps[0], list) else timestamps
    words = words_from_tokens(token_timestamps, duration)
    text = " ".join(word["word"] for word in words).strip()
    return {"text": text, "words": words, "duration": duration}


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
        "lookahead_tokens": LOOKAHEAD_TOKENS,
    }


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form(SERVED_NAME),
    language: str = Form("auto"),
    response_format: str = Form("json"),
    timestamps: bool = Form(False),
    timestamp_granularities: list[str] = Form(
        default=[], alias="timestamp_granularities[]"
    ),
):
    try:
        audio, sample_rate = decode_upload(await file.read())
        result = transcribe_audio(audio, sample_rate, language or "auto")
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
    return {
        **base,
        "duration": round(result["duration"], 3),
        "language": language or "auto",
        "words": result["words"],
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


if __name__ == "__main__":
    import uvicorn

    get_runtime()
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "1239")))
