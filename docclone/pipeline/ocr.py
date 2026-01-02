from __future__ import annotations

from typing import List

import cv2
import pytesseract

from .utils import OCRWord


def run_tesseract(image: cv2.Mat) -> List[OCRWord]:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    data = pytesseract.image_to_data(rgb, output_type=pytesseract.Output.DICT)
    words: List[OCRWord] = []
    count = len(data["text"])
    for i in range(count):
        text = data["text"][i].strip()
        conf = float(data["conf"][i])
        if not text or conf < 0:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        words.append(OCRWord(text=text, bbox=(x, y, x + w, y + h), confidence=conf / 100.0))
    return words


def run_easyocr(image: cv2.Mat) -> List[OCRWord]:
    import easyocr

    reader = easyocr.Reader(["en"], gpu=False)
    results = reader.readtext(image)
    words: List[OCRWord] = []
    for bbox, text, conf in results:
        xs = [int(point[0]) for point in bbox]
        ys = [int(point[1]) for point in bbox]
        words.append(
            OCRWord(
                text=text,
                bbox=(min(xs), min(ys), max(xs), max(ys)),
                confidence=float(conf),
            )
        )
    return words


def run_ocr(image: cv2.Mat, engine: str) -> List[OCRWord]:
    if engine == "easyocr":
        return run_easyocr(image)
    return run_tesseract(image)
