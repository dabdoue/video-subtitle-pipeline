#!/usr/bin/env python3
"""Local command-provider adapter for NVIDIA Nemotron 3.5 via Transformers."""

import argparse
import json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument(
        "--model",
        default="nvidia/nemotron-3.5-asr-streaming-0.6b",
    )
    parser.add_argument("--device", default=None, help="Examples: cuda:0, mps, cpu")
    args = parser.parse_args()

    from transformers import pipeline

    options = {"task": "automatic-speech-recognition", "model": args.model}
    if args.device:
        options["device"] = args.device
    recognizer = pipeline(**options)
    result = recognizer(args.audio)
    print(json.dumps({"text": result.get("text", "")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

