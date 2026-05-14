"""
Phase 5: Claude Vision으로 애매한 매칭 케이스 최종 판별

- match_candidates.json에서 status='pending_claude' 항목만 처리
- 두 이미지를 Claude에 동시 전달해 동일 상품 여부 판별
- 판별 근거(색상, 패턴 방향, 핏 등)를 reason 필드에 기록
"""
import argparse
import base64
import json
from pathlib import Path

import anthropic
from tqdm import tqdm

from config import (
    ANTHROPIC_API_KEY, CANDIDATES_FILE, CLAUDE_MODEL,
    DATA_DIR, IMAGES_DIR, TEST_BRAND,
)
from utils.state import load_state, save_state

JUDGE_PROMPT = """
두 이미지를 비교해 이미지1에서 모델이 착용한 [{category}]가
이미지2의 상품({product_name})과 동일한 상품인지 판별하세요.

판별 시 반드시 아래 항목을 근거로 사용하세요:
- 색상 (정확한 색조)
- 패턴 종류 (무지/줄무늬/꽃무늬/체크 등)
- 패턴 방향 (가로/세로/사선, 해당하는 경우)
- 패턴 굵기 및 간격
- 실루엣/핏 (타이트/와이드/버블 등)
- 길이 (반바지/5부/7부/긴바지 등)

아래 JSON만 출력하세요 (설명 없이):
{{"match": true 또는 false, "confidence": 0.0~1.0, "reason": "판별 근거 한 문장"}}
"""


def encode_image(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def judge_pair(
    client: anthropic.Anthropic,
    source_image_path: Path,
    target_image_path: Path,
    category: str,
    product_name: str,
) -> dict:
    """두 이미지를 Claude에 전달해 매칭 여부 반환."""
    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=256,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": encode_image(source_image_path),
                            },
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": encode_image(target_image_path),
                            },
                        },
                        {
                            "type": "text",
                            "text": JUDGE_PROMPT.format(
                                category=category,
                                product_name=product_name,
                            ),
                        },
                    ],
                }
            ],
        )
        raw = response.content[0].text.strip()
        return json.loads(raw)
    except Exception as e:
        return {"match": False, "confidence": 0.0, "reason": f"판별 오류: {e}"}


def run(brand_no: int) -> None:
    if not ANTHROPIC_API_KEY:
        print("[Phase 5] ANTHROPIC_API_KEY 미설정. 건너뜀.")
        return

    state = load_state()
    already_done = set(state.get("judged", []))

    with open(CANDIDATES_FILE, encoding="utf-8") as f:
        candidates: dict = json.load(f)

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    pending_pairs = [
        (pid, result, idx, match)
        for pid, result in candidates.items()
        for idx, match in enumerate(result["matches"])
        if match["status"] == "pending_claude" and pid not in already_done
    ]
    print(f"[Phase 5] Claude 판별 대상: {len(pending_pairs)}쌍")

    judged = confirmed = rejected = 0
    for pid, result, idx, match in tqdm(pending_pairs, desc="Claude 판별"):
        source_path = IMAGES_DIR / str(brand_no) / "상의" / f"{pid}.jpg"
        target_pid = str(match["productNo"])
        target_path = IMAGES_DIR / str(brand_no) / match["category"] / f"{target_pid}.jpg"

        if not source_path.exists() or not target_path.exists():
            candidates[pid]["matches"][idx]["status"] = "skip_no_image"
            continue

        verdict = judge_pair(
            client, source_path, target_path,
            match["category"], match["productName"],
        )

        candidates[pid]["matches"][idx].update({
            "status":           "confirmed" if verdict.get("match") else "rejected",
            "claude_confidence": verdict.get("confidence", 0.0),
            "claude_reason":    verdict.get("reason", ""),
        })

        if verdict.get("match"):
            confirmed += 1
        else:
            rejected += 1
        judged += 1

    # 저장
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CANDIDATES_FILE, "w", encoding="utf-8") as f:
        json.dump(candidates, f, ensure_ascii=False, indent=2)

    state["judged"] = list(already_done | {pid for pid, *_ in pending_pairs})
    save_state(state)

    print(f"  판별 완료: {judged}쌍 (확정={confirmed} / 거절={rejected})")
    print("[Phase 5] 완료")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--brand", type=int, default=TEST_BRAND["brandNo"])
    args = parser.parse_args()
    run(args.brand)
