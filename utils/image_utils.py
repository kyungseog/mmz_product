from __future__ import annotations

import numpy as np
from pathlib import Path
from PIL import Image


def crop_segment(image: Image.Image, mask: np.ndarray) -> Image.Image | None:
    """
    세그멘테이션 마스크로 해당 영역을 크롭.
    마스크가 너무 작거나 없으면 None 반환.
    """
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # 바운딩 박스 패딩 (10px)
    pad = 10
    h, w = image.size[1], image.size[0]
    y_min = max(0, y_min - pad)
    x_min = max(0, x_min - pad)
    y_max = min(h, y_max + pad)
    x_max = min(w, x_max + pad)

    cropped = image.crop((x_min, y_min, x_max, y_max))
    return cropped


def segment_area_ratio(mask: np.ndarray) -> float:
    """마스크가 전체 이미지에서 차지하는 비율."""
    return mask.sum() / mask.size


def is_edge_clipped(mask: np.ndarray, margin: int = 5) -> bool:
    """세그먼트가 이미지 가장자리에 붙어 잘렸는지 확인."""
    h, w = mask.shape
    edges = [
        mask[:margin, :].any(),
        mask[-margin:, :].any(),
        mask[:, :margin].any(),
        mask[:, -margin:].any(),
    ]
    return any(edges)


def normalize_url(url: str) -> str:
    """프로토콜 상대 URL을 https로 정규화."""
    if url.startswith("//"):
        return "https:" + url
    return url


def save_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(path), "JPEG", quality=90)
