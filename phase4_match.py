"""
Phase 4: 카테고리별 코사인 유사도 매칭

- 각 상품 이미지의 세그먼트 임베딩을 같은 브랜드 동일 카테고리 상품과 비교
- 유사도 > 0.85: 자동 확정 / 0.60~0.85: Claude 판별 대기 / < 0.60: 제외
- data/match_candidates.json에 결과 저장
"""
import argparse
import json
import pickle
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from config import (
    CANDIDATES_FILE, DATA_DIR, EMBEDDINGS_FILE,
    PRODUCTS_FILE, TEST_BRAND,
    THRESHOLD_AUTO, THRESHOLD_CLAUDE, TOP_K_CANDIDATES,
)
from utils.state import load_state, save_state


def cosine_similarity(vec_a: list, vec_b: list) -> float:
    a = np.array(vec_a)
    b = np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def build_category_index(products: list[dict], embeddings: dict) -> dict:
    """카테고리 → {productNo: embedding} 인덱스 생성."""
    index: dict[str, dict] = defaultdict(dict)
    for p in products:
        pid = str(p["productNo"])
        if pid in embeddings["products"]:
            index[p["category"]][pid] = embeddings["products"][pid]
    return index


def run(brand_no: int, summary: bool = False) -> None:
    state = load_state()
    already_done = set(state.get("matched", []))

    with open(PRODUCTS_FILE, encoding="utf-8") as f:
        products = json.load(f)

    with open(EMBEDDINGS_FILE, "rb") as f:
        embeddings = pickle.load(f)

    pid_to_product = {str(p["productNo"]): p for p in products}
    category_index = build_category_index(products, embeddings)

    # 기존 결과 로드
    candidates: dict = {}
    if CANDIDATES_FILE.exists():
        with open(CANDIDATES_FILE, encoding="utf-8") as f:
            candidates = json.load(f)

    new_count = skip_count = 0
    auto_match = claude_queue = no_match = 0

    for p in tqdm(products, desc="유사도 매칭"):
        pid = str(p["productNo"])
        if pid in already_done:
            skip_count += 1
            continue

        seg_embeddings = embeddings.get("segments", {}).get(pid, {})
        if not seg_embeddings:
            # 세그먼트 없음(모델 없는 사진 등) → 이전 결과가 남아있으면 제거
            candidates.pop(pid, None)
            already_done.add(pid)
            continue

        source_category = p["category"]
        product_result = {
            "source": pid,
            "source_name": p["productName"],
            "source_category": source_category,
            "matches": [],
        }

        for seg_category, seg_vecs in seg_embeddings.items():
            # 자기 카테고리와 동일한 세그먼트는 코디 후보에서 제외
            if seg_category == source_category:
                continue

            # 이 세그먼트 카테고리의 상품 풀
            pool = category_index.get(seg_category, {})
            if not pool:
                continue

            # 세그먼트 임베딩이 여러 개면 평균 사용
            if isinstance(seg_vecs[0], list):
                seg_vec = np.mean(seg_vecs, axis=0).tolist()
            else:
                seg_vec = seg_vecs

            # 전체 풀과 유사도 계산 (자기 자신 + 자기 카테고리 제외)
            scores = [
                (target_pid, cosine_similarity(seg_vec, target_vec))
                for target_pid, target_vec in pool.items()
                if target_pid != pid  # 자기 자신 제외
            ]
            scores.sort(key=lambda x: x[1], reverse=True)
            top_k = scores[:TOP_K_CANDIDATES]

            for target_pid, score in top_k:
                if score < THRESHOLD_CLAUDE:
                    no_match += 1
                    continue

                target = pid_to_product.get(target_pid, {})
                match_entry = {
                    "productNo":   int(target_pid),
                    "productName": target.get("productName", ""),
                    "category":    seg_category,
                    "score":       round(score, 4),
                    "status":      "auto" if score >= THRESHOLD_AUTO else "pending_claude",
                }
                product_result["matches"].append(match_entry)

                if score >= THRESHOLD_AUTO:
                    auto_match += 1
                else:
                    claude_queue += 1

        if product_result["matches"]:
            candidates[pid] = product_result

        already_done.add(pid)
        new_count += 1

    # 저장
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CANDIDATES_FILE, "w", encoding="utf-8") as f:
        json.dump(candidates, f, ensure_ascii=False, indent=2)

    state["matched"] = list(already_done)
    save_state(state)

    print(f"  신규 처리: {new_count}개 / 스킵: {skip_count}개")
    print(f"  자동 확정: {auto_match}쌍 / Claude 대기: {claude_queue}쌍 / 제외: {no_match}쌍")

    if summary:
        print("\n[매칭 결과 요약]")
        for pid, result in list(candidates.items())[:10]:
            print(f"  {result['source_name']} ({pid})")
            for m in result["matches"]:
                flag = "✅" if m["status"] == "auto" else "⚠️"
                print(f"    {flag} {m['category']} → {m['productName']} (score={m['score']})")

    print("[Phase 4] 완료")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--brand", type=int, default=TEST_BRAND["brandNo"])
    parser.add_argument("--summary", action="store_true", help="결과 요약 출력")
    args = parser.parse_args()
    run(args.brand, summary=args.summary)
