from __future__ import annotations

from config import EXCLUDE_KEYWORDS, CATEGORY_KEYWORDS


def classify(product_name: str) -> str | None:
    """
    상품명으로 카테고리 분류.
    세트/상하복 상품이면 None 반환 (분석 제외).
    키워드 미해당이면 '미분류' 반환.
    """
    for kw in EXCLUDE_KEYWORDS:
        if kw in product_name:
            return None

    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in product_name:
                return category

    return "미분류"


def split_products_by_category(products: list[dict]) -> dict[str, list[dict]]:
    """
    상품 리스트를 카테고리별로 분류.
    None(세트) 상품은 결과에서 제외.
    """
    result: dict[str, list[dict]] = {}
    for p in products:
        cat = classify(p["productName"])
        if cat is None:
            continue
        result.setdefault(cat, []).append(p)
    return result
