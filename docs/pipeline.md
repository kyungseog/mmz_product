# 코디 상품 자동 연결 파이프라인

moomooz.co.kr 상품 이미지를 분석해 같은 사진에 함께 착용된 상품을 찾아 샵바이 연관상품으로 등록하는 파이프라인.

---

## 전체 흐름

```
Phase 1  →  Phase 2  →  Phase 3  →  Phase 4  →  Phase 5  →  Phase 6
BQ 수집      세그멘테이션   임베딩 생성    유사도 매칭   Claude 판별    샵바이 등록
(Python)    (SegFormer)   (DINOv2)      (numpy)      (API, 선택)    (API)
```

각 Phase는 독립적으로 실행 가능하며, `data/state.json`으로 처리 이력을 관리해 재실행 시 이미 처리된 상품은 건너뜀.

---

## Phase 1 — 데이터 수집

**입력:** BigQuery `mmz-store.shopby.products`  
**출력:** `data/products.json`, `data/images/{brandNo}/{category}/{productNo}.jpg`

### 세트 상품 제외 (최우선)
아래 키워드 포함 시 분석 대상에서 제외:
```
상하복, 세트, SET, set, 1+1, 2종, 3종, 래쉬가드세트, 수영세트
```

### 상품명 기반 카테고리 분류
brandNo와 무관하게 상품명 키워드로 분류:

| 카테고리 | 키워드 |
|---|---|
| 상의 | 티셔츠, 니트, 블라우스, 셔츠, 탑, 베스트, 조끼, 나시 |
| 하의 | 팬츠, 바지, 레깅스, 스커트, 쇼츠, 반바지 |
| 아우터 | 자켓, 가디건, 점퍼, 코트, 집업, 후드 |
| 원피스 | 원피스, 점프수트, 올인원, 롬퍼 |
| 악세서리 | 모자, 양말, 가방, 헤어밴드, 스카프 |
| 미분류 | 위 키워드 미해당 → Phase 5에서 Claude가 판단 |

---

## Phase 2 — 멀티 아이템 세그멘테이션

**모델:** `mattmdjaga/segformer_b2_clothes`  
**입력:** `data/images/` 원본 이미지  
**출력:** `data/cropped/`, `data/segments.json`

### SegFormer 클래스 → 내부 카테고리 매핑

| SegFormer 클래스 | 내부 카테고리 | 처리 |
|---|---|---|
| upper_clothes (4) | 상의 | ✅ 크롭 |
| skirt (5) | 하의 | ✅ 크롭 |
| pants (6) | 하의 | ✅ 크롭 |
| dress (7) | 원피스 | ✅ 크롭 |
| hat (1) | 악세서리 | ✅ 크롭 |
| bag (16) | 악세서리 | ✅ 크롭 |
| scarf (17) | 악세서리 | ✅ 크롭 |
| belt (8) | 악세서리 | ✅ 크롭 |
| shoes (9, 10) | 신발 | ✅ 크롭 |
| hair / face / limbs | 신체 | ❌ 제외 |

### 메인 상품 제외 로직
DB 카테고리 기준으로 해당 상품의 주 클래스를 제외 후 나머지를 코디 후보로:
- 상의 상품 → upper_clothes 제외, 나머지(하의/아우터/악세서리) 코디 후보
- 하의 상품 → pants/skirt 제외, 나머지 코디 후보
- 원피스 상품 → dress 제외, 나머지 코디 후보

### 품질 필터
- 세그먼트 면적 < 전체의 5% → 너무 작음, 제외
- 세그먼트가 이미지 가장자리에 잘림 → 신뢰도 낮음 표시

---

## Phase 3 — 임베딩 생성

**모델:** `facebook/dinov2-large` (1024차원)  
**입력:** `data/cropped/` 크롭 이미지  
**출력:** `data/embeddings.pkl`

```python
embeddings = {
    "products": {productNo: vector},        # 상품 원본 임베딩
    "segments": {
        productNo: {
            "하의": vector,
            "아우터": vector,
            "악세서리": [vector, ...]        # 여러 개 가능
        }
    }
}
```

GPU 배치 처리: cuda=64장, mps=16장, cpu=8장

---

## Phase 4 — 유사도 매칭

**입력:** `data/embeddings.pkl`, `data/products.json`  
**출력:** `data/match_candidates.json`

각 상품 이미지에서 감지된 세그먼트를 **같은 브랜드의 동일 카테고리 상품**과 코사인 유사도 비교:

```
상품 A 이미지의 "하의" 세그먼트
    → 브랜드 내 하의 상품 전체 임베딩과 비교
    → Top-5 후보 추출
```

### 신뢰도 분기

| 유사도 | 처리 |
|---|---|
| > 0.85 | ✅ 자동 매칭 확정 |
| 0.60 ~ 0.85 | ⚠️ Phase 5 Claude 판별 대기열 |
| < 0.60 | ❌ 매칭 없음 |

---

## Phase 5 — Claude 최종 판별

**대상:** Phase 4에서 유사도 0.60~0.85 사이 케이스만  
**입력:** 원본 상품 이미지 + 후보 상품 이미지 (동시 전달)  
**출력:** `data/results.json` (확정 매칭 추가)

### 판별 프롬프트
```
이미지1(상품 착용 사진)에서 모델이 입은 [카테고리]와
이미지2([후보 상품명])가 동일한 상품인지 판별하세요.

색상, 패턴 방향(가로/세로/사선), 패턴 굵기, 핏, 길이를 근거로 제시.

출력 JSON:
{"match": true/false, "confidence": 0~1, "reason": "판별 근거"}
```

**비용 최적화:** 전체 호출의 20% 이하 목표 (DINOv2 1차 필터로 Top-3만 Claude 전달)

---

## Phase 6 — 샵바이 등록

**입력:** `data/results.json`  
**출력:** 샵바이 API 연관상품 등록

```
PUT /products/{productNo}/related-products
```

`--dry-run` 옵션으로 실제 등록 없이 결과 확인 가능.

---

## 처리 이력 관리 (재처리 방지)

`data/state.json`에 각 Phase별 처리 완료 상품 기록:

```json
{
  "downloaded": ["133561995", "133561716", ...],
  "segmented":  ["133561995", ...],
  "embedded":   ["133561995", ...],
  "matched":    ["133561995", ...],
  "judged":     ["133561995", ...],
  "registered": ["133561995", ...]
}
```

스케줄링 실행 시 각 Phase는 해당 리스트에 없는 상품만 처리.

---

## 환경별 설정

| 항목 | MacBook (테스트) | Linux GPU 서버 (프로덕션) |
|---|---|---|
| Device | cpu / mps | cuda |
| 배치 사이즈 | 8 / 16 | 64 |
| 대상 브랜드 | 1개 (테스트) | 전체 브랜드 |
| 스케줄링 | 없음 | cron 주 1회 |
| DINOv2 모델 | dinov2-base | dinov2-large |

---

## 파일 구조

```
data/
├── products.json          # BQ 수집 상품 목록
├── state.json             # 처리 이력 (재처리 방지)
├── segments.json          # 세그멘테이션 메타데이터
├── embeddings.pkl         # DINOv2 임베딩 캐시
├── match_candidates.json  # Phase 4 매칭 결과 (Claude 판별 포함)
├── results.json           # 최종 확정 코디 매칭
├── images/
│   └── {brandNo}/
│       ├── 상의/
│       ├── 하의/
│       ├── 아우터/
│       ├── 원피스/
│       ├── 악세서리/
│       └── 미분류/
└── cropped/
    └── {productNo}_{category}_{idx}.jpg
```
