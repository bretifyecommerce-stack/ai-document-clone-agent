from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from .utils import Block, OCRWord, extract_color, group_words_by_line, load_font_paths


class FontManager:
    def __init__(self, font_dir: str | None) -> None:
        self.fonts: Dict[str, str] = {}
        for path in load_font_paths(font_dir):
            name = path.split("/")[-1].split(".")[0]
            if name not in self.fonts:
                pdfmetrics.registerFont(TTFont(name, path))
                self.fonts[name] = name

    def pick_font(self) -> str:
        if self.fonts:
            return list(self.fonts.keys())[0]
        return "Helvetica"


def _estimate_font_size(words: List[OCRWord]) -> float:
    if not words:
        return 10.0
    heights = [w.bbox[3] - w.bbox[1] for w in words]
    return max(6.0, float(np.median(heights)))


def _estimate_space_width(words: List[OCRWord]) -> float:
    if not words:
        return 6.0
    widths = [max(1.0, (w.bbox[2] - w.bbox[0]) / max(1, len(w.text))) for w in words]
    return float(np.median(widths))


def _build_line_text(words: List[OCRWord]) -> str:
    words_sorted = sorted(words, key=lambda w: w.bbox[0])
    space_width = _estimate_space_width(words_sorted)
    text_parts: List[str] = []
    last_x2 = None
    for word in words_sorted:
        if last_x2 is None:
            text_parts.append(word.text)
        else:
            gap = max(0, word.bbox[0] - last_x2)
            spaces = int(round(gap / max(1.0, space_width)))
            text_parts.append(" " * max(1, spaces) + word.text)
        last_x2 = word.bbox[2]
    return "".join(text_parts)


def render_block_text(
    pdf: canvas.Canvas,
    block: Block,
    page_height_pt: float,
    pt_per_px: float,
    font_name: str,
    image: np.ndarray,
) -> None:
    if not block.words:
        return
    lines = group_words_by_line(block.words, y_threshold=12)
    for line in lines:
        font_size_px = _estimate_font_size(line)
        font_size_pt = font_size_px * pt_per_px
        pdf.setFont(font_name, font_size_pt)
        color = extract_color(image, block.bbox)
        pdf.setFillColorRGB(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)
        line_text = _build_line_text(line)
        x = min(word.bbox[0] for word in line)
        y = min(word.bbox[1] for word in line)
        x_pt = x * pt_per_px
        y_pt = page_height_pt - (y * pt_per_px) - font_size_pt
        pdf.drawString(x_pt, y_pt, line_text)


def render_block_line(
    pdf: canvas.Canvas,
    block: Block,
    page_height_pt: float,
    pt_per_px: float,
) -> None:
    x1, y1, x2, y2 = block.bbox
    x1_pt, x2_pt = x1 * pt_per_px, x2 * pt_per_px
    y_pt = page_height_pt - (y1 * pt_per_px)
    pdf.setLineWidth(max(0.5, (y2 - y1) * pt_per_px))
    pdf.line(x1_pt, y_pt, x2_pt, y_pt)


def render_block_image(
    pdf: canvas.Canvas,
    block: Block,
    page_height_pt: float,
    pt_per_px: float,
    image: np.ndarray,
) -> None:
    x1, y1, x2, y2 = block.bbox
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return
    img_reader = ImageReader(crop)
    width_pt = (x2 - x1) * pt_per_px
    height_pt = (y2 - y1) * pt_per_px
    x_pt = x1 * pt_per_px
    y_pt = page_height_pt - (y2 * pt_per_px)
    pdf.drawImage(img_reader, x_pt, y_pt, width=width_pt, height=height_pt)


def render_page(
    pdf: canvas.Canvas,
    page_width_pt: float,
    page_height_pt: float,
    blocks: List[Block],
    image: np.ndarray,
    dpi: int,
    font_manager: FontManager,
) -> None:
    pt_per_px = 72.0 / dpi
    font_name = font_manager.pick_font()
    for block in blocks:
        if block.block_type == "text":
            if block.words:
                render_block_text(pdf, block, page_height_pt, pt_per_px, font_name, image)
            else:
                render_block_image(pdf, block, page_height_pt, pt_per_px, image)
        elif block.block_type == "line":
            render_block_line(pdf, block, page_height_pt, pt_per_px)
        elif block.block_type == "image":
            render_block_image(pdf, block, page_height_pt, pt_per_px, image)
