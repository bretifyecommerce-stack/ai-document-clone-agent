# AI Document Clone Agent

Production-ready pipeline that recreates document pages from reference images into
vector-based PDFs with selectable text whenever possible.

## Features

- Deterministic preprocessing: deskew, perspective correction, page boundary detection.
- Layout analysis with heuristics and connected components.
- OCR (Tesseract by default, EasyOCR optional).
- PDF reconstruction with vector primitives and embedded fonts (ReportLab).
- Debug artifacts (blocks, OCR, diff heatmap, metrics).
- FastAPI service + CLI.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Note:** Tesseract OCR must be installed on the system for the default engine.

## CLI Usage

```bash
python -m docclone.cli replicate --input samples/page.png --output out.pdf
```

## API Usage

```bash
uvicorn docclone.api:app --reload
```

- `POST /replicate` with one or more images (multipart `files` field)
- `POST /replicate/async` for background processing
- `GET /job/{id}` and `GET /job/{id}/download`

## Configuration

Create a `config.yaml` or `config.json`:

```yaml
ocr_engine: tesseract
layout_engine: heuristic
fallback_threshold: 0.08
output_debug: true
font_dir: "docclone/assets/fonts"
dpi: 300
watermark_text: null
```

Use it with the CLI:

```bash
python -m docclone.cli replicate --input samples/page.png --output out.pdf --config config.yaml
```

## Output

- `out.pdf`: vector-based PDF
- `out/debug/page_01/blocks.json`: block + OCR metadata
- `out/debug/page_01/diff_heatmap.png`: visualization of differences
- `out/debug/page_01/metrics.json`: SSIM/MAE values

## Troubleshooting

- **Fonts not matching**: add matching `.ttf` files under `docclone/assets/fonts` and
  update `font_dir` in config.
- **Low SSIM**: provide higher DPI images and enable debug output to inspect blocks.
- **Complex watermarks**: OCR may fail; enable fallback by keeping non-text blocks as images.

## Allowed Use

This project is intended for legitimate document replication and restoration
workflows. Do not use it to forge or falsify identification documents.
