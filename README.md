# LightOnOCR-2-1B demo scripts

This repo contains two small Python demos for [`lightonai/LightOnOCR-2-1B`](https://huggingface.co/lightonai/LightOnOCR-2-1B):

- `scripts/run_transformers_ocr.py`: runs the model locally with `transformers`
- `scripts/run_vllm_ocr.py`: sends the same kind of request to a `vLLM` OpenAI-compatible endpoint

Both scripts accept:

- `--image-path` or `--image-url`
- `--pdf-path` or `--pdf-url`
- `--page` for PDF page selection
- `--output` to save OCR text to a file

If you do not pass an input, both scripts default to the sample receipt image used in the model card.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Local `transformers` demo

Run the default sample from the model card:

```bash
python scripts/run_transformers_ocr.py
```

Run against a local PDF:

```bash
python scripts/run_transformers_ocr.py --pdf-path /path/to/document.pdf --page 1
```

Run against an image URL and save the output:

```bash
python scripts/run_transformers_ocr.py \
  --image-url https://huggingface.co/datasets/hf-internal-testing/fixtures_ocr/resolve/main/SROIE-receipt.jpeg \
  --output output/receipt.txt
```

## `vLLM` demo

Start a local `vLLM` server with the settings from the model card:

```bash
vllm serve lightonai/LightOnOCR-2-1B \
  --limit-mm-per-prompt '{"image": 1}' \
  --mm-processor-cache-gb 0 \
  --no-enable-prefix-caching
```

Then call it with the demo script:

```bash
python scripts/run_vllm_ocr.py
```

Run it on the first page of the example paper from the model card:

```bash
python scripts/run_vllm_ocr.py \
  --pdf-url https://arxiv.org/pdf/2412.13663 \
  --page 1
```

If your endpoint requires auth, set `OPENAI_API_KEY` or pass `--api-key`.

## Notes

- The Hugging Face model card recommends rendering PDFs at 200 DPI with a target longest dimension of 1540 px. The scripts do that automatically for PDF inputs and also downscale oversized images to the same longest edge.
- The local script defaults to `bfloat16` on CUDA and `float32` on CPU or MPS for safer portability.
- `transformers>=5.0.0` is required by the model card for the LightOnOCR classes.
