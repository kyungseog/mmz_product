"""
Phase 2: SegFormer로 상품 이미지 내 모든 패션 아이템 감지 및 크롭

- 메인 상품 카테고리에 해당하는 세그먼트는 제외
- 나머지 감지된 아이템(하의/아우터/악세서리 등)을 코디 후보로 크롭
- data/segments.json에 메타데이터 저장
"""
import argparse
import json

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from config import (
    BATCH_SIZE, CATEGORY_MAIN_SEGMENTS, CROPPED_DIR,
    DATA_DIR, IMAGES_DIR, MIN_SEGMENT_AREA,
    PRODUCTS_FILE, SEGMENT_TO_CATEGORY, SEGFORMER_ID2LABEL,
    SEGFORMER_MODEL, SEGMENTS_FILE, TEST_BRAND, get_device,
)
from utils.image_utils import crop_segment, is_edge_clipped, save_image, segment_area_ratio
from utils.state import load_state, save_state


def load_model(device: str):
    processor = SegformerImageProcessor.from_pretrained(SEGFORMER_MODEL)
    model = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_MODEL)
    model.to(device).eval()
    return processor, model


def segment_image(
    image: Image.Image,
    processor,
    model,
    device: str,
) -> np.ndarray:
    """이미지를 세그멘테이션해 픽셀별 클래스 ID 배열 반환."""
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits  # [1, num_classes, H, W]

    # 원본 이미지 크기로 업샘플링
    upsampled = torch.nn.functional.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False
    )
    return upsampled.argmax(dim=1).squeeze().cpu().numpy()


HUMAN_LABELS = {"hair", "face", "sunglasses", "left_leg", "right_leg", "left_arm", "right_arm"}


def has_model(seg_map: np.ndarray) -> bool:
    """세그멘테이션 맵에 사람(모델) 신체 클래스가 하나라도 있으면 True."""
    detected = {SEGFORMER_ID2LABEL[int(i)] for i in np.unique(seg_map)}
    return bool(detected & HUMAN_LABELS)


def extract_segments(
    image: Image.Image,
    seg_map: np.ndarray,
    main_category: str,
) -> list[dict]:
    """
    세그멘테이션 맵에서 코디 후보 세그먼트 추출.
    메인 상품 클래스와 신체 관련 클래스는 제외.
    모델이 없는 상품 사진은 호출 전에 has_model()로 걸러낼 것.
    """
    main_labels = CATEGORY_MAIN_SEGMENTS.get(main_category, set())
    segments = []

    detected_labels = set(SEGFORMER_ID2LABEL[int(i)] for i in np.unique(seg_map))
    for label in detected_labels:
        if label in ("background", "hair", "face", "sunglasses",
                     "left_leg", "right_leg", "left_arm", "right_arm"):
            continue
        if label in main_labels:
            continue

        category = SEGMENT_TO_CATEGORY.get(label)
        if category is None:
            continue

        class_id = next(k for k, v in SEGFORMER_ID2LABEL.items() if v == label)
        mask = (seg_map == class_id)

        area = segment_area_ratio(mask)
        if area < MIN_SEGMENT_AREA:
            continue

        crop = crop_segment(image, mask)
        if crop is None:
            continue

        segments.append({
            "label":    label,
            "category": category,
            "area":     round(area, 4),
            "clipped":  is_edge_clipped(mask),
            "crop":     crop,
        })

    return segments


def run(brand_no: int) -> None:
    state = load_state()
    already_done = set(state.get("segmented", []))

    with open(PRODUCTS_FILE, encoding="utf-8") as f:
        products = json.load(f)

    device = get_device()
    print(f"[Phase 2] 세그멘테이션 시작 (device={device})")
    processor, model = load_model(device)

    segments_meta: dict = {}
    if SEGMENTS_FILE.exists():
        with open(SEGMENTS_FILE, encoding="utf-8") as f:
            segments_meta = json.load(f)

    new_count = skip_count = no_model_count = 0
    for p in tqdm(products, desc="세그멘테이션"):
        pid = str(p["productNo"])
        if pid in already_done:
            skip_count += 1
            continue

        image_path = IMAGES_DIR / str(brand_no) / p["category"] / f"{pid}.jpg"
        if not image_path.exists():
            continue

        image = Image.open(image_path).convert("RGB")
        seg_map = segment_image(image, processor, model, device)

        # 모델(사람)이 없는 상품 단독 사진 → 코디 분석 불가, 건너뜀
        if not has_model(seg_map):
            segments_meta[pid] = []
            already_done.add(pid)
            no_model_count += 1
            continue

        raw_segments = extract_segments(image, seg_map, p["category"])

        product_segments = []
        for idx, seg in enumerate(raw_segments):
            crop_filename = f"{pid}_{seg['category']}_{idx}.jpg"
            crop_path = CROPPED_DIR / crop_filename
            save_image(seg["crop"], crop_path)
            product_segments.append({
                "category": seg["category"],
                "label":    seg["label"],
                "area":     seg["area"],
                "clipped":  seg["clipped"],
                "crop_file": crop_filename,
            })

        segments_meta[pid] = product_segments
        already_done.add(pid)
        new_count += 1

    # 저장
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(SEGMENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(segments_meta, f, ensure_ascii=False, indent=2)

    state["segmented"] = list(already_done)
    save_state(state)

    print(f"  신규 처리: {new_count}개 / 모델 없음 제외: {no_model_count}개 / 스킵: {skip_count}개")
    print("[Phase 2] 완료")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--brand", type=int, default=TEST_BRAND["brandNo"])
    args = parser.parse_args()
    run(args.brand)
