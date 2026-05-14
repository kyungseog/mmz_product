# mmz_product — 코디 상품 자동 연결

moomooz.co.kr 상품 이미지를 AI로 분석해 함께 착용된 상품을 찾고 샵바이 연관상품으로 자동 등록하는 파이프라인.

전체 설계: [docs/pipeline.md](docs/pipeline.md)

---

## 환경 설정

```bash
pip install -r requirements.txt
```

BigQuery 인증:
```bash
gcloud auth application-default login
```

Claude API 키 (Phase 5용):
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

샵바이 API 키 (Phase 6용):
```bash
export SHOPBY_API_KEY=...
export SHOPBY_CLIENT_ID=...
```

---

## 실행 (1브랜드 테스트)

각 Phase를 순서대로 실행. 이미 처리된 상품은 자동으로 건너뜀.

```bash
# 1. BigQuery에서 상품 수집 및 이미지 다운로드
python phase1_collect.py --brand 43147416

# 2. SegFormer로 이미지 내 아이템 세그멘테이션
python phase2_segment.py --brand 43147416

# 3. DINOv2 임베딩 생성
python phase3_embed.py --brand 43147416

# 4. 코사인 유사도 매칭
python phase4_match.py --brand 43147416

# 5. 애매한 케이스 Claude 판별 (선택)
python phase5_claude_judge.py --brand 43147416

# 6. 샵바이 연관상품 등록 (--dry-run으로 먼저 확인)
python phase6_register.py --brand 43147416 --dry-run
python phase6_register.py --brand 43147416
```

전체 한번에 실행:
```bash
python run_all.py --brand 43147416
```

---

## 스케줄링 (Linux GPU 서버)

`crontab -e`에 추가:
```
0 2 * * 1   cd /path/to/mmz_product && python run_all.py --all-brands >> logs/cron.log 2>&1
```

매주 월요일 새벽 2시 실행. `data/state.json`으로 신규 상품만 처리.

---

## 결과 확인

```bash
# 매칭 결과 요약
python phase4_match.py --brand 43147416 --summary

# 등록 전 결과 미리보기
python phase6_register.py --brand 43147416 --dry-run
```

`data/results.json` 에서 최종 매칭 결과 확인 가능.
