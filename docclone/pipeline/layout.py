from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from .utils import Block


def _binarize(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)


def detect_lines(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=150, minLineLength=50, maxLineGap=5)
    boxes: List[Tuple[int, int, int, int]] = []
    if lines is None:
        return boxes
    for line in lines:
        x1, y1, x2, y2 = line[0]
        boxes.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
    return boxes


def detect_blocks(image: np.ndarray) -> List[Block]:
    binary = _binarize(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dilated = cv2.dilate(binary, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blocks: List[Block] = []
    for idx, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < 500:
            continue
        aspect = w / float(h)
        block_type = "text"
        if aspect > 6 and h < 40:
            block_type = "line"
        elif area > (image.shape[0] * image.shape[1] * 0.05):
            block_type = "image"
        blocks.append(
            Block(
                block_id=f"block_{idx}",
                block_type=block_type,
                bbox=(x, y, x + w, y + h),
                confidence=0.6,
                words=[],
            )
        )
    return blocks
