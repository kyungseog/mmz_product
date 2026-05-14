"""
Phase 3: DINOv2로 크롭된 세그먼트 이미지 임베딩 생성

- 상품 원본 이미지 + 각 세그먼트 크롭 이미지 모두 임베딩
- data/embeddings.pkl에 저장 (재실행 시 기존 임베딩 재사용)
"""
import argparse
import json
import pickle
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from config import (
    BATCH_SIZE, CROPPED_DIR, DATA_DIR, DINO_MODEL,
    EMBEDDINGS_FILE, IMAGES_DIR, PRODUCTS_FILE,
    SEGMENTS_FILE, TEST_BRAND, get_device,
)
from utils.state import load_state, save_state


def load_model(device: str):
    processor = AutoImageProcessor.from_pretrained(DINO_MODEL)
    model = AutoModel.from_pretrained(DINO_MODEL)
    model.to(device).eval()
    return processor, model


@torch.no_grad()
def embed_images(
    image_paths: list[Path],
    processor,
    model,
    device: str,
    batch_size: int,
) -> list[list[float]]:
    """이미지 파일 경로 리스트를 받아 임베딩 벡터 리스트 반환."""
    all_vectors = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt").to(device)
        outputs = model(**inputs)
        # CLS 토큰 벡터를 임베딩으로 사용
        vecs = outputs.last_hidden_state[:, 0, :]
        vecs = F.normalize(vecs, dim=-1)
        all_vectors.extend(vecs.cpu().tolist())
    return all_vectors


def run(brand_no: int) -> None:
    state = load_state()
    already_done = set(state.get("embedded", []))

    with open(PRODUCTS_FILE, encoding="utf-8") as f:
        products = json.load(f)

    with open(SEGMENTS_FILE, encoding="utf-8") as f:
        segments_meta: dict = json.load(f)

    device = get_device()
    batch_size = BATCH_SIZE[device]
    print(f"[Phase 3] 임베딩 생성 시작 (device={device}, batch={batch_size})")
    processor, model = load_model(device)

    # 기존 임베딩 로드
    embeddings: dict = {"products": {}, "segments": {}}
    if EMBEDDINGS_FILE.exists():
        with open(EMBEDDINGS_FILE, "rb") as f:
            embeddings = pickle.load(f)

    new_count = skip_count = 0
    for p in tqdm(products, desc="임베딩"):
        pid = str(p["productNo"])
        if pid in already_done:
            skip_count += 1
            continue

        # 상품 원본 이미지 임베딩
        orig_path = IMAGES_DIR / str(brand_no) / p["category"] / f"{pid}.jpg"
        if orig_path.exists():
            vecs = embed_images([orig_path], processor, model, device, batch_size)
            embeddings["products"][pid] = vecs[0]

        # 세그먼트 크롭 임베딩
        segs = segments_meta.get(pid, [])
        seg_embeddings: dict[str, list] = {}
        for seg in segs:
            crop_path = CROPPED_DIR / seg["crop_file"]
            if not crop_path.exists():
                continue
            vecs = embed_images([crop_path], processor, model, device, batch_size)
            cat = seg["category"]
            seg_embeddings.setdefault(cat, []).append(vecs[0])

        if seg_embeddings:
            embeddings["segments"][pid] = seg_embeddings
        else:
            # 모델 없는 상품: 이전 실행에서 남은 세그먼트 임베딩 제거
            embeddings["segments"].pop(pid, None)

        already_done.add(pid)
        new_count += 1

    # 저장
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(embeddings, f)

    state["embedded"] = list(already_done)
    save_state(state)

    print(f"  신규 처리: {new_count}개 / 스킵: {skip_count}개")
    print("[Phase 3] 완료")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--brand", type=int, default=TEST_BRAND["brandNo"])
    args = parser.parse_args()
    run(args.brand)
