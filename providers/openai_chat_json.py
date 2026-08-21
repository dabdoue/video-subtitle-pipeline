#!/usr/bin/env python3
"""Command-provider adapter for OpenAI-compatible chat endpoints.

Reads the pipeline prompt from stdin and prints only the model's response text.
Configure LLM_BASE_URL, LLM_API_KEY, and optionally LLM_MODEL.
"""

import argparse
import json
import os
import sys
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", ""))
    args = parser.parse_args()
    base_url = os.environ.get("LLM_BASE_URL", "").rstrip("/")
    if not base_url or not args.model:
        raise SystemExit("LLM_BASE_URL and --model/LLM_MODEL are required")
    url = base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"
    body = json.dumps(
        {
            "model": args.model,
            "messages": [{"role": "user", "content": sys.stdin.read()}],
            "temperature": 0,
            "stream": False,
            "response_format": {"type": "json_object"},
        }
    ).encode()
    headers = {"Content-Type": "application/json"}
    if key := os.environ.get("LLM_API_KEY"):
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=300) as response:
        payload = json.load(response)
    print(payload["choices"][0]["message"]["content"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

