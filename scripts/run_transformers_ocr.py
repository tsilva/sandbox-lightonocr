#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import tempfile

import torch
from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor

from _common import load_input_image, positive_int, write_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LightOnOCR-2-1B locally with transformers."
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
        "--model",
        default="lightonai/LightOnOCR-2-1B",
        help="Hugging Face model id. Default: lightonai/LightOnOCR-2-1B.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Execution device. Default: auto.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
        help="Torch dtype. Default: auto.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=positive_int,
        default=1024,
        help="Maximum number of generated tokens. Default: 1024.",
    )
    parser.add_argument("--output", help="Optional file path to save OCR text.")
    return parser.parse_args()


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str, requested: str) -> torch.dtype:
    if requested == "float32":
        return torch.float32
    if requested == "float16":
        return torch.float16
    if requested == "bfloat16":
        return torch.bfloat16
    if device == "cuda":
        return torch.bfloat16
    return torch.float32


def move_inputs(inputs, device: str, dtype: torch.dtype):
    moved = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            if value.is_floating_point():
                moved[key] = value.to(device=device, dtype=dtype)
            else:
                moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def main() -> None:
    args = parse_args()
    image, source = load_input_image(args)
    device = pick_device(args.device)
    dtype = pick_dtype(device, args.dtype)

    print(f"Loading {args.model} on {device} with {dtype}...", file=sys.stderr)
    model = LightOnOcrForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    processor = LightOnOcrProcessor.from_pretrained(args.model)

    with tempfile.NamedTemporaryFile(suffix=".png") as temp_image:
        image.save(temp_image.name)
        content = [{"type": "image", "path": temp_image.name}]
        if args.instruction:
            content.append({"type": "text", "text": args.instruction})

        conversation = [{"role": "user", "content": content}]
        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

    inputs = move_inputs(inputs, device, dtype)
    print(f"OCR source: {source}", file=sys.stderr)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

    generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    text = processor.decode(generated_ids, skip_special_tokens=True).strip()
    write_output(text, args.output)
    print(text)


if __name__ == "__main__":
    main()
