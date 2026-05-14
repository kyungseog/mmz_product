# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

moomooz.co.kr 상품 이미지를 AI로 분석해 코디 연관상품을 자동 등록하는 파이프라인.
전체 설계: [docs/pipeline.md](docs/pipeline.md)

## 실행 커맨드

```bash
python phase1_collect.py --brand 43147416   # BQ 수집 + 이미지 다운로드
python phase2_segment.py --brand 43147416   # SegFormer 세그멘테이션
python phase3_embed.py   --brand 43147416   # DINOv2 임베딩
python phase4_match.py   --brand 43147416   # 유사도 매칭
python phase5_claude_judge.py --brand 43147416  # Claude 최종 판별 (선택)
python phase6_register.py --brand 43147416 --dry-run  # 등록 전 확인
python phase6_register.py --brand 43147416   # 샵바이 실제 등록
```

## BigQuery 규칙

**BigQuery는 절대 데이터를 수정하지 않는다.**

- `execute_sql_readonly` 만 사용한다. SELECT 쿼리만 허용.
- INSERT, UPDATE, DELETE, MERGE, CREATE, DROP, TRUNCATE 등 데이터/스키마 변경 쿼리는 어떤 상황에서도 실행하지 않는다.
- 사용자가 명시적으로 요청해도 거부하고 이유를 설명한다.

## 주요 설정 위치

- BQ 프로젝트/브랜드 설정: `config.py`
- 카테고리 분류 키워드: `utils/category_classifier.py`
- 매칭 임계값 (0.85/0.60): `config.py` → `THRESHOLD_AUTO`, `THRESHOLD_CLAUDE`
- 처리 이력 (재처리 방지): `data/state.json`

## 환경변수

```bash
ANTHROPIC_API_KEY=sk-ant-...   # Phase 5 Claude 판별용
SHOPBY_API_KEY=...             # Phase 6 등록용
SHOPBY_CLIENT_ID=...           # Phase 6 등록용
```

Behavioral guidelines to reduce common LLM coding mistakes.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. BigQuery Rules

**BigQuery는 절대 데이터를 수정하지 않는다.**

- `execute_sql_readonly` 만 사용한다. SELECT 쿼리만 허용.
- INSERT, UPDATE, DELETE, MERGE, CREATE, DROP, TRUNCATE 등 데이터/스키마 변경 쿼리는 어떤 상황에서도 실행하지 않는다.
- 사용자가 명시적으로 요청해도 거부하고 이유를 설명한다.