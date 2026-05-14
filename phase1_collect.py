from __future__ import annotations

"""
Phase 1: BigQuery에서 상품 수집 및 이미지 다운로드

- 상품명 기반 카테고리 분류 (세트/상하복 제외)
- data/state.json으로 이미 다운로드된 상품 건너뜀
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import requests
from google.cloud import bigquery
from tqdm import tqdm

from config import (
    BQ_PROJECT, BQ_DATASET, BQ_TABLE,
    DATA_DIR, IMAGES_DIR, PRODUCTS_FILE, STATE_FILE,
    TEST_BRAND,
)
from utils.category_classifier import classify
from utils.image_utils import normalize_url
from utils.state import load_state, save_state


def fetch_products(brand_no: int) -> list[dict]:
    client = bigquery.Client(project=BQ_PROJECT)
    query = f"""
        SELECT
            productNo,
            productName,
            productManagementCd,
            registerYmdt,
            listImageUrls,
            brandNo,
            brandName
        FROM `{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}`
        WHERE brandNo = {brand_no}
          AND listImageUrls IS NOT NULL
          AND saleStatusType = 'ONSALE'
        ORDER BY registerYmdt DESC
    """
    rows = client.query(query).result()
    return [dict(row) for row in rows]


def download_image(url: str, save_path: Path) -> bool:
    try:
        r = requests.get(normalize_url(url), timeout=15)
        r.raise_for_status()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes(r.content)
        return True
    except Exception as e:
        print(f"  [WARN] 다운로드 실패 {url}: {e}")
        return False


def run(brand_no: int, skip_bq: bool = False) -> None:
    state = load_state()
    already_done = set(state.get("downloaded", []))

    print(f"[Phase 1] 브랜드 {brand_no} 수집 시작")

    if skip_bq and PRODUCTS_FILE.exists():
        print("  BQ 조회 건너뜀 (data/products.json 사용)")
        with open(PRODUCTS_FILE, encoding="utf-8") as f:
            classified = json.load(f)
    else:
        products = fetch_products(brand_no)
        print(f"  전체 상품: {len(products)}개")

    if not (skip_bq and PRODUCTS_FILE.exists()):
        # 카테고리 분류 및 세트 제외
        classified = []
        excluded = 0
        for p in products:
            cat = classify(p["productName"])
            if cat is None:
                excluded += 1
                continue
            p["category"] = cat
            classified.append(p)

        print(f"  세트/상하복 제외: {excluded}개")

    print(f"  분석 대상: {len(classified)}개")

    cat_count = Counter(p["category"] for p in classified)
    for cat, cnt in sorted(cat_count.items()):
        print(f"    {cat}: {cnt}개")

    # products.json 저장 (전체 덮어쓰기 — 신규 상품 반영)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(PRODUCTS_FILE, "w", encoding="utf-8") as f:
        json.dump(classified, f, ensure_ascii=False, indent=2)

    # 이미지 다운로드
    new_count = skip_count = fail_count = 0
    for p in tqdm(classified, desc="이미지 다운로드"):
        pid = str(p["productNo"])
        if pid in already_done:
            skip_count += 1
            continue

        save_path = IMAGES_DIR / str(brand_no) / p["category"] / f"{pid}.jpg"
        if download_image(p["listImageUrls"], save_path):
            already_done.add(pid)
            new_count += 1
        else:
            fail_count += 1

        time.sleep(0.05)  # 서버 부하 방지

    state["downloaded"] = list(already_done)
    save_state(state)

    print(f"  신규 다운로드: {new_count}개 / 스킵: {skip_count}개 / 실패: {fail_count}개")
    print("[Phase 1] 완료")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--brand", type=int, default=TEST_BRAND["brandNo"])
    parser.add_argument("--skip-bq", action="store_true",
                        help="BQ 조회 건너뜀 (data/products.json 이미 존재할 때)")
    args = parser.parse_args()
    run(args.brand, skip_bq=args.skip_bq)
