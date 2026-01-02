from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .layout import detect_blocks
from .ocr import run_ocr
from .utils import Block, OCRWord, RenderedPage, extract_color, image_to_pdf_points


def assign_words_to_blocks(words: List[OCRWord], blocks: List[Block]) -> List[Block]:
    for block in blocks:
        for word in words:
            x1, y1, x2, y2 = word.bbox
            bx1, by1, bx2, by2 = block.bbox
            if x1 >= bx1 and y1 >= by1 and x2 <= bx2 and y2 <= by2:
                block.words.append(word)
    return blocks


def detect_image_blocks(image: np.ndarray, blocks: List[Block]) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    images: List[Tuple[np.ndarray, Tuple[int, int, int, int]]] = []
    for block in blocks:
        if block.block_type == "image":
            x1, y1, x2, y2 = block.bbox
            crop = image[y1:y2, x1:x2].copy()
            images.append((crop, block.bbox))
    return images


def reconstruct_page(image: np.ndarray, page_index: int, dpi: int, ocr_engine: str) -> RenderedPage:
    blocks = detect_blocks(image)
    words = run_ocr(image, ocr_engine)
    blocks = assign_words_to_blocks(words, blocks)
    width_pt, height_pt = image_to_pdf_points(image, dpi)
    images = detect_image_blocks(image, blocks)
    return RenderedPage(
        page_index=page_index,
        width_pt=width_pt,
        height_pt=height_pt,
        blocks=blocks,
        images=images,
    )


def build_debug_payload(rendered: RenderedPage, image: np.ndarray) -> dict:
    payload = {
        "page_index": rendered.page_index,
        "width_pt": rendered.width_pt,
        "height_pt": rendered.height_pt,
        "blocks": [],
    }
    for block in rendered.blocks:
        color = extract_color(image, block.bbox)
        payload["blocks"].append(
            {
                "id": block.block_id,
                "type": block.block_type,
                "bbox": block.bbox,
                "confidence": block.confidence,
                "color": color,
                "words": [
                    {"text": word.text, "bbox": word.bbox, "confidence": word.confidence}
                    for word in block.words
                ],
            }
        )
    return payload
