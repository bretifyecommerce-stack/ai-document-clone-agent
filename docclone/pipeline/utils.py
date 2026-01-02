from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


@dataclass
class PageImage:
    image: np.ndarray
    path: Path
    dpi: int


@dataclass
class OCRWord:
    text: str
    bbox: Tuple[int, int, int, int]
    confidence: float


@dataclass
class Block:
    block_id: str
    block_type: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    words: List[OCRWord]


@dataclass
class RenderedPage:
    page_index: int
    width_pt: float
    height_pt: float
    blocks: List[Block]
    images: List[Tuple[np.ndarray, Tuple[int, int, int, int]]]


@dataclass
class Config:
    ocr_engine: str = "tesseract"
    layout_engine: str = "heuristic"
    fallback_threshold: float = 0.08
    output_debug: bool = True
    font_dir: Optional[str] = None
    dpi: int = 300
    watermark_text: Optional[str] = None


def load_config(path: Optional[str]) -> Config:
    if not path:
        return Config()
    config_path = Path(path)
    data: Dict[str, Any]
    if config_path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        data = yaml.safe_load(config_path.read_text()) or {}
    else:
        data = json.loads(config_path.read_text())
    return Config(**data)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_image(path: Path, dpi: int) -> PageImage:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    return PageImage(image=image, path=path, dpi=dpi)


def resize_to_dpi(image: np.ndarray, current_dpi: int, target_dpi: int) -> np.ndarray:
    if current_dpi == target_dpi:
        return image
    scale = target_dpi / current_dpi
    new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)


def to_pil(image: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def normalize_bbox(bbox: Iterable[int]) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def crop_region(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = normalize_bbox(bbox)
    return image[y1:y2, x1:x2].copy()


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def list_images(input_path: Path) -> List[Path]:
    if input_path.is_dir():
        files = sorted(
            [p for p in input_path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        )
        if not files:
            raise ValueError(f"No images found in {input_path}")
        return files
    if input_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise ValueError("Input must be a PNG/JPG image or a directory of images")
    return [input_path]


def image_to_pdf_points(image: np.ndarray, dpi: int) -> Tuple[float, float]:
    height_px, width_px = image.shape[:2]
    width_pt = width_px * 72.0 / dpi
    height_pt = height_px * 72.0 / dpi
    return width_pt, height_pt


def infer_page_size(image: np.ndarray, dpi: int) -> Tuple[float, float]:
    return image_to_pdf_points(image, dpi)


def merge_word_boxes(words: List[OCRWord]) -> Tuple[int, int, int, int]:
    x1s, y1s, x2s, y2s = zip(*(w.bbox for w in words))
    return (min(x1s), min(y1s), max(x2s), max(y2s))


def group_words_by_line(words: List[OCRWord], y_threshold: int = 10) -> List[List[OCRWord]]:
    if not words:
        return []
    words_sorted = sorted(words, key=lambda w: (w.bbox[1], w.bbox[0]))
    lines: List[List[OCRWord]] = []
    current_line: List[OCRWord] = []
    current_y = words_sorted[0].bbox[1]
    for word in words_sorted:
        if abs(word.bbox[1] - current_y) > y_threshold:
            lines.append(current_line)
            current_line = [word]
            current_y = word.bbox[1]
        else:
            current_line.append(word)
    if current_line:
        lines.append(current_line)
    return lines


def extract_color(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int]:
    crop = crop_region(image, bbox)
    if crop.size == 0:
        return (0, 0, 0)
    avg = crop.mean(axis=(0, 1))
    return int(avg[2]), int(avg[1]), int(avg[0])


def to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#%02x%02x%02x" % rgb


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def write_image(path: Path, image: np.ndarray) -> None:
    cv2.imwrite(str(path), image)


def load_font_paths(font_dir: Optional[str]) -> List[str]:
    if not font_dir:
        return []
    path = Path(font_dir)
    if not path.exists():
        return []
    return [str(p) for p in path.glob("**/*.ttf")]
