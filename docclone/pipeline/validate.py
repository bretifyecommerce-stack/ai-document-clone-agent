from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import fitz
import numpy as np
from skimage.metrics import structural_similarity


@dataclass
class ValidationResult:
    ssim: float
    mae: float
    diff_image: np.ndarray


def render_pdf_page(pdf_path: Path, page_index: int, dpi: int) -> np.ndarray:
    doc = fitz.open(str(pdf_path))
    page = doc.load_page(page_index)
    pix = page.get_pixmap(dpi=dpi)
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def compare_images(reference: np.ndarray, generated: np.ndarray) -> ValidationResult:
    ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    gen = cv2.cvtColor(generated, cv2.COLOR_BGR2GRAY)
    min_h = min(ref.shape[0], gen.shape[0])
    min_w = min(ref.shape[1], gen.shape[1])
    ref = ref[:min_h, :min_w]
    gen = gen[:min_h, :min_w]
    ssim_value, diff = structural_similarity(ref, gen, full=True)
    diff_image = (1 - diff) * 255
    diff_image = diff_image.astype(np.uint8)
    mae = float(np.mean(np.abs(ref.astype(np.float32) - gen.astype(np.float32))) / 255.0)
    diff_color = cv2.applyColorMap(diff_image, cv2.COLORMAP_JET)
    return ValidationResult(ssim=ssim_value, mae=mae, diff_image=diff_color)


def save_diff(path: Path, diff_image: np.ndarray) -> None:
    cv2.imwrite(str(path), diff_image)
