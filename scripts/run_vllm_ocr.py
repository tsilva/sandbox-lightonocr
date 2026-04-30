#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import os
import sys

import requests

from _common import image_to_png_bytes, load_input_image, positive_int, write_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send OCR requests for LightOnOCR-2-1B to a vLLM endpoint."
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--image-path", help="Path to an input image.")
    source_group.add_argument("--image-url", help="URL of an input image.")
    source_group.add_argument("--pdf-path", help="Path to an input PDF.")
    source_group.add_argument("--pdf-url", help="URL of an input PDF.")
    parser.add_argument(
        "--page",
        type=positive_int,
        default=1,
        help="1-based PDF page number. Default: 1.",
    )
    parser.add_argument(
        "--instruction",
        help="Optional text instruction appended after the image in the chat prompt.",
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:8000/v1/chat/completions",
        help="OpenAI-compatible chat completions endpoint.",
    )
    parser.add_argument(
        "--model",
        default="lightonai/LightOnOCR-2-1B",
        help="Model name sent to the endpoint.",
    )
    parser.add_argument(
        "--api-key",
        help="Bearer token for the endpoint. Defaults to OPENAI_API_KEY if set.",
    )
    parser.add_argument(
        "--max-tokens",
        type=positive_int,
        default=4096,
        help="Maximum completion tokens. Default: 4096.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature. Default: 0.2.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling. Default: 0.9.",
    )
    parser.add_argument("--output", help="Optional file path to save OCR text.")
    return parser.parse_args()


def extract_text(response_json: dict) -> str:
    content = response_json["choices"][0]["message"]["content"]
    if isinstance(content, str):
        return content.strip()

    parts = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(item.get("text", ""))
    return "".join(parts).strip()


def main() -> None:
    args = parse_args()
    image, source = load_input_image(args)
    image_bytes = image_to_png_bytes(image)
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    message_content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_base64}"},
        }
    ]
    if args.instruction:
        message_content.append({"type": "text", "text": args.instruction})

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": message_content}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }

    headers = {}
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    print(f"POST {args.endpoint}", file=sys.stderr)
    print(f"OCR source: {source}", file=sys.stderr)
    response = requests.post(args.endpoint, json=payload, headers=headers, timeout=600)
    response.raise_for_status()

    text = extract_text(response.json())
    write_output(text, args.output)
    print(text)


if __name__ == "__main__":
    main()
