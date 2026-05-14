"""
Phase 6: 샵바이 API로 연관상품 등록

- match_candidates.json에서 status='auto' 또는 'confirmed' 항목을 등록
- --dry-run: 실제 등록 없이 결과만 출력
- data/results.json에 최종 등록 결과 저장
"""
import argparse
import json

import requests
from tqdm import tqdm

from config import (
    CANDIDATES_FILE, DATA_DIR, RESULTS_FILE,
    SHOPBY_API_BASE, SHOPBY_API_KEY, SHOPBY_CLIENT_ID, TEST_BRAND,
)
from utils.state import load_state, save_state

REGISTER_STATUSES = {"auto", "confirmed"}


def build_headers() -> dict:
    return {
        "Content-Type":  "application/json",
        "X-Api-Key":     SHOPBY_API_KEY,
        "X-Client-Id":   SHOPBY_CLIENT_ID,
    }


def register_related_products(source_no: int, related_nos: list[int]) -> bool:
    """샵바이 API로 연관상품 등록. 성공 시 True 반환."""
    url = f"{SHOPBY_API_BASE}/products/{source_no}/related-products"
    payload = {"relatedProductNos": related_nos}
    try:
        r = requests.put(url, json=payload, headers=build_headers(), timeout=10)
        r.raise_for_status()
        return True
    except Exception as e:
        print(f"  [WARN] 등록 실패 {source_no}: {e}")
        return False


def collect_confirmed_matches(candidates: dict) -> dict[str, list[dict]]:
    """확정된 매칭(auto/confirmed)만 추출."""
    result = {}
    for pid, data in candidates.items():
        confirmed = [
            m for m in data["matches"]
            if m["status"] in REGISTER_STATUSES
        ]
        if confirmed:
            result[pid] = confirmed
    return result


def run(brand_no: int, dry_run: bool = False) -> None:
    state = load_state()
    already_done = set(state.get("registered", []))

    with open(CANDIDATES_FILE, encoding="utf-8") as f:
        candidates: dict = json.load(f)

    confirmed = collect_confirmed_matches(candidates)
    print(f"[Phase 6] 등록 대상: {len(confirmed)}개 상품")
    if dry_run:
        print("  [DRY-RUN 모드] 실제 API 호출 없음\n")

    final_results = []
    success = skip = fail = 0

    for pid, matches in tqdm(confirmed.items(), desc="연관상품 등록"):
        if pid in already_done:
            skip += 1
            continue

        related_nos = [m["productNo"] for m in matches]
        source_name = candidates[pid]["source_name"]

        if dry_run:
            print(f"  [{pid}] {source_name}")
            for m in matches:
                print(f"    → [{m['category']}] {m['productName']} "
                      f"(score={m['score']}, status={m['status']})")
            success += 1
        else:
            if not SHOPBY_API_KEY:
                print("[Phase 6] SHOPBY_API_KEY 미설정. --dry-run으로 먼저 확인하세요.")
                break
            ok = register_related_products(int(pid), related_nos)
            if ok:
                already_done.add(pid)
                success += 1
            else:
                fail += 1

        final_results.append({
            "sourceProductNo":   int(pid),
            "sourceProductName": source_name,
            "relatedProducts":   matches,
        })

    # 결과 저장
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    if not dry_run:
        state["registered"] = list(already_done)
        save_state(state)

    print(f"  성공: {success}개 / 스킵: {skip}개 / 실패: {fail}개")
    print(f"  결과 저장: {RESULTS_FILE}")
    print("[Phase 6] 완료")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--brand",   type=int, default=TEST_BRAND["brandNo"])
    parser.add_argument("--dry-run", action="store_true", help="등록 없이 결과만 출력")
    args = parser.parse_args()
    run(args.brand, dry_run=args.dry_run)
