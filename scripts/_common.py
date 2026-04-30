from __future__ import annotations

import io
from pathlib import Path

import pypdfium2 as pdfium
import requests
from PIL import Image

DEFAULT_IMAGE_URL = (
    "https://huggingface.co/datasets/hf-internal-testing/fixtures_ocr/resolve/main/"
    "SROIE-receipt.jpeg"
)
TARGET_DPI = 200
TARGET_LONGEST_EDGE = 1540


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise ValueError("value must be >= 1")
    return parsed


def fetch_bytes(url: str) -> bytes:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return response.content


def normalize_image(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    longest_edge = max(image.size)
    if longest_edge <= TARGET_LONGEST_EDGE:
        return image

    scale = TARGET_LONGEST_EDGE / longest_edge
    new_size = (
        max(1, round(image.width * scale)),
        max(1, round(image.height * scale)),
    )
    return image.resize(new_size, Image.Resampling.LANCZOS)


def open_image_from_bytes(data: bytes) -> Image.Image:
    with Image.open(io.BytesIO(data)) as image:
        return normalize_image(image)


def render_pdf_page(pdf_bytes: bytes, page_number: int) -> Image.Image:
    pdf = pdfium.PdfDocument(pdf_bytes)
    try:
        page_count = len(pdf)
        if not 1 <= page_number <= page_count:
            raise ValueError(
                f"page {page_number} is out of range for a {page_count}-page PDF"
            )
        page = pdf[page_number - 1]
        image = page.render(scale=TARGET_DPI / 72).to_pil()
        return normalize_image(image)
    finally:
        pdf.close()


def load_input_image(args) -> tuple[Image.Image, str]:
    if args.image_path:
        path = Path(args.image_path).expanduser().resolve()
        return open_image_from_bytes(path.read_bytes()), str(path)

    if args.image_url:
        return open_image_from_bytes(fetch_bytes(args.image_url)), args.image_url

    if args.pdf_path:
        path = Path(args.pdf_path).expanduser().resolve()
        return render_pdf_page(path.read_bytes(), args.page), f"{path}#page={args.page}"

    if args.pdf_url:
        return render_pdf_page(fetch_bytes(args.pdf_url), args.page), (
            f"{args.pdf_url}#page={args.page}"
        )

    return open_image_from_bytes(fetch_bytes(DEFAULT_IMAGE_URL)), DEFAULT_IMAGE_URL


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def write_output(text: str, output_path: str | None) -> None:
    if output_path is None:
        return

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
