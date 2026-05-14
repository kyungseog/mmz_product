import os
import torch
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── BigQuery ──────────────────────────────────────────────
BQ_PROJECT = os.getenv("BQ_PROJECT", "")
BQ_DATASET = "shopby"
BQ_TABLE   = "products"

# ── 브랜드 ────────────────────────────────────────────────
# 맥북 테스트용 단일 브랜드. run_all.py --all-brands 시 전체 조회.
TEST_BRAND = {"brandNo": 43148332, "brandName": "헤이시아"}

# ── 경로 ─────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
DATA_DIR       = BASE_DIR / "data"
IMAGES_DIR     = DATA_DIR / "images"
CROPPED_DIR    = DATA_DIR / "cropped"
PRODUCTS_FILE  = DATA_DIR / "products.json"
STATE_FILE     = DATA_DIR / "state.json"
SEGMENTS_FILE  = DATA_DIR / "segments.json"
EMBEDDINGS_FILE= DATA_DIR / "embeddings.pkl"
CANDIDATES_FILE= DATA_DIR / "match_candidates.json"
RESULTS_FILE   = DATA_DIR / "results.json"

# ── Device ───────────────────────────────────────────────
def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

BATCH_SIZE = {"cuda": 64, "mps": 16, "cpu": 8}

# ── 모델 ─────────────────────────────────────────────────
# Linux GPU 서버에서는 dinov2-large 사용 권장
DINO_MODEL = "facebook/dinov2-base"    # 맥북 테스트: base
# DINO_MODEL = "facebook/dinov2-large" # GPU 서버: large
SEGFORMER_MODEL = "mattmdjaga/segformer_b2_clothes"

# ── 세그멘테이션 ──────────────────────────────────────────
# SegFormer 클래스 ID → 레이블
SEGFORMER_ID2LABEL = {
    0: "background", 1: "hat",        2: "hair",      3: "sunglasses",
    4: "upper_clothes", 5: "skirt",   6: "pants",     7: "dress",
    8: "belt",       9: "left_shoe",  10: "right_shoe", 11: "face",
    12: "left_leg",  13: "right_leg", 14: "left_arm",  15: "right_arm",
    16: "bag",       17: "scarf",
}

# SegFormer 레이블 → 내부 카테고리
SEGMENT_TO_CATEGORY = {
    "upper_clothes": "상의",
    "skirt":         "하의",
    "pants":         "하의",
    "dress":         "원피스",
    "hat":           "악세서리",
    "bag":           "악세서리",
    "scarf":         "악세서리",
    "belt":          "악세서리",
    "left_shoe":     "신발",
    "right_shoe":    "신발",
}

# 카테고리별 메인 상품 클래스 (착용 이미지에서 제외할 세그먼트)
CATEGORY_MAIN_SEGMENTS = {
    "상의":   {"upper_clothes"},
    "하의":   {"skirt", "pants"},
    "아우터": {"upper_clothes"},
    "원피스": {"dress"},
    "악세서리": {"hat", "bag", "scarf", "belt"},
}

MIN_SEGMENT_AREA = 0.05  # 전체 픽셀의 5% 미만 세그먼트 제외

# ── 카테고리 분류 키워드 ──────────────────────────────────
EXCLUDE_KEYWORDS = [
    "상하복", "세트", "SET", "set", "1+1", "2종", "3종",
    "래쉬가드세트", "수영세트", "래쉬가드2종", "래쉬가드3종",
]

CATEGORY_KEYWORDS = {
    "상의":   ["티셔츠", "니트", "블라우스", "셔츠", "탑", "베스트", "조끼", "나시", "맨투맨", "후드티", "민소매"],
    "하의":   ["팬츠", "바지", "레깅스", "스커트", "쇼츠", "반바지", "고쟁이"],
    "아우터": ["자켓", "가디건", "점퍼", "코트", "집업", "후드집업", "점프"],
    "원피스": ["원피스", "점프수트", "올인원", "롬퍼"],
    "악세서리": ["모자", "양말", "가방", "헤어밴드", "스카프", "헤어핀"],
}

# ── 매칭 임계값 ───────────────────────────────────────────
THRESHOLD_AUTO   = 0.85  # 이상: 자동 매칭 확정
THRESHOLD_CLAUDE = 0.60  # 이상: Claude 판별 요청 / 미만: 매칭 없음
TOP_K_CANDIDATES = 5     # 유사도 상위 K개 후보 추출

# ── 샵바이 API ────────────────────────────────────────────
SHOPBY_API_BASE  = "https://api.shopby.co.kr"
SHOPBY_API_KEY   = os.getenv("SHOPBY_API_KEY", "")
SHOPBY_CLIENT_ID = os.getenv("SHOPBY_CLIENT_ID", "")

# ── Claude API ────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL      = "claude-sonnet-4-6"
