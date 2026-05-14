"""
전체 파이프라인 순차 실행 (Phase 1 ~ 6)

사용법:
  python run_all.py --brand 43147416          # 단일 브랜드
  python run_all.py --all-brands              # 전체 브랜드 (BQ에서 조회)
  python run_all.py --brand 43147416 --dry-run  # 등록 전 결과 확인
"""
import argparse

from google.cloud import bigquery

from config import BQ_DATASET, BQ_PROJECT, TEST_BRAND
import phase1_collect
import phase2_segment
import phase3_embed
import phase4_match
import phase5_claude_judge
import phase6_register


def fetch_all_brands() -> list[int]:
    client = bigquery.Client(project=BQ_PROJECT)
    query = f"""
        SELECT DISTINCT brandNo
        FROM `{BQ_PROJECT}.{BQ_DATASET}.products`
        WHERE saleStatusType = 'ONSALE'
          AND listImageUrls IS NOT NULL
    """
    return [row.brandNo for row in client.query(query).result()]


def run_pipeline(brand_no: int, dry_run: bool = False) -> None:
    print(f"\n{'='*50}")
    print(f"브랜드 {brand_no} 파이프라인 시작")
    print(f"{'='*50}")

    phase1_collect.run(brand_no)
    phase2_segment.run(brand_no)
    phase3_embed.run(brand_no)
    phase4_match.run(brand_no)
    phase5_claude_judge.run(brand_no)   # API 키 없으면 자동 스킵
    phase6_register.run(brand_no, dry_run=dry_run)

    print(f"\n브랜드 {brand_no} 파이프라인 완료\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--brand",      type=int, default=TEST_BRAND["brandNo"])
    parser.add_argument("--all-brands", action="store_true", help="전체 브랜드 처리")
    parser.add_argument("--dry-run",    action="store_true", help="Phase 6 등록 없이 결과만 출력")
    args = parser.parse_args()

    if args.all_brands:
        brands = fetch_all_brands()
        print(f"전체 브랜드 {len(brands)}개 처리 시작")
        for brand_no in brands:
            run_pipeline(brand_no, dry_run=args.dry_run)
    else:
        run_pipeline(args.brand, dry_run=args.dry_run)
