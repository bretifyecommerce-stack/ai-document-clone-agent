from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from reportlab.pdfgen import canvas

from .preprocess import preprocess
from .reconstruct import build_debug_payload, reconstruct_page
from .render_pdf import FontManager, render_page
from .utils import Config, ensure_dir, list_images, load_config, load_image, write_image
from .validate import compare_images, render_pdf_page, save_diff


def replicate_images(input_path: str, output_pdf: str, config_path: str | None = None) -> Path:
    config = load_config(config_path)
    input_paths = list_images(Path(input_path))
    output_path = Path(output_pdf)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    debug_root = output_path.with_suffix("")
    font_manager = FontManager(config.font_dir)

    pdf = None
    for page_index, image_path in enumerate(input_paths):
        page = load_image(image_path, config.dpi)
        prep = preprocess(page)
        rendered = reconstruct_page(prep.image, page_index, config.dpi, config.ocr_engine)
        if pdf is None:
            pdf = canvas.Canvas(str(output_path), pagesize=(rendered.width_pt, rendered.height_pt))
        else:
            pdf.setPageSize((rendered.width_pt, rendered.height_pt))
        render_page(pdf, rendered.width_pt, rendered.height_pt, rendered.blocks, prep.image, config.dpi, font_manager)
        if config.watermark_text:
            pdf.saveState()
            pdf.setFont("Helvetica", 12)
            pdf.setFillColorRGB(0.5, 0.5, 0.5, alpha=0.4)
            pdf.drawString(20, 20, config.watermark_text)
            pdf.restoreState()
        pdf.showPage()

        if config.output_debug:
            page_debug = debug_root / "debug" / f"page_{page_index + 1:02d}"
            ensure_dir(page_debug)
            write_image(page_debug / "preprocessed.png", prep.image)
            debug_payload = build_debug_payload(rendered, prep.image)
            (page_debug / "blocks.json").write_text(
                __import__("json").dumps(debug_payload, indent=2, ensure_ascii=False)
            )
    if pdf is None:
        raise ValueError("No pages found")
    pdf.save()

    if config.output_debug:
        for page_index, image_path in enumerate(input_paths):
            page = load_image(image_path, config.dpi)
            generated = render_pdf_page(output_path, page_index, config.dpi)
            validation = compare_images(page.image, generated)
            page_debug = output_path.with_suffix("") / "debug" / f"page_{page_index + 1:02d}"
            ensure_dir(page_debug)
            save_diff(page_debug / "diff_heatmap.png", validation.diff_image)
            (page_debug / "metrics.json").write_text(
                __import__("json").dumps({"ssim": validation.ssim, "mae": validation.mae}, indent=2)
            )

    return output_path
