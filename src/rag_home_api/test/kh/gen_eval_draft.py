"""[2.2 보완판 · v1 공정 설계 확정] 전월세 블록 청크에서 LLM으로 RAGAS 호환 평가셋 초안(dev 10 + test 20) 생성.

작성자: 최기혁 (kh)
저장 위치: src/rag_home_api/test/kh/eval/kh_rent_{dev,test}_draft.json

v1 확정 설계 (/grill-me 합의):
  1) 대상 범위: 서울 25개 구 전체(6,201청크, deal_count >= 10 우선) 균등 분산
  2) 청년 타깃 면적대 가중: 33㎡이하(원룸급) ~50% + 33~60㎡(소형) ~35% + 60㎡초과(중대형) ~15%
  3) 정답(reference) 밀도: 핵심 사실 중심 2~3문장 간결형, 복합 추론 비교 포인트 최대 2개
  4) 문항 외(거절) 5문항: 제외 주제 5개 1:1 슬롯 배정 (dev 2 + test 3, 하드코딩 없이 LLM 생성)
  5) 자동 품질 게이트: 날짜 복사 치팅 · 청크 ID 실존 · dev/test 청크 겹침 0 · 분포 자동 검사
     → dev 10문항은 육안 검수 후 kh_rent_dev.json 확정, test 20문항은 자동 검증 통과 후 봉인(kh_rent_test.json)!
"""

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from rag_home_api.shared.core.vector_store import get_llm
from rag_home_api.test.kh.rent_chunker import KH_EVAL_DIR, RENT_MANIFEST_PATH

QuestionType = Literal["단순 질문", "복합 추론 질문", "용어 다르게 질문", "문항 외 질문"]
EvalCategory = Literal["직접 검색", "표현 변형·유사 조건 구분", "근거 없음"]

# ============================================================
# 개발용(10문항: 답변가능 8 + 근거없음 2) / 최종용(20문항: 직접검색 10 + 표현변형·유사조건 7 + 근거없음 3)
# ============================================================
DEV_PLAN: list[tuple[QuestionType, EvalCategory, int]] = [
    ("단순 질문", "직접 검색", 4),
    ("복합 추론 질문", "직접 검색", 2),
    ("용어 다르게 질문", "표현 변형·유사 조건 구분", 2),
    ("문항 외 질문", "근거 없음", 2),
]

TEST_PLAN: list[tuple[QuestionType, EvalCategory, int]] = [
    ("단순 질문", "직접 검색", 7),
    ("복합 추론 질문", "직접 검색", 3),
    ("용어 다르게 질문", "표현 변형·유사 조건 구분", 4),
    ("복합 추론 질문", "표현 변형·유사 조건 구분", 3),
    ("문항 외 질문", "근거 없음", 3),
]

# 청년 타깃 면적대 가중치 (33㎡이하 ~50%, 33~60㎡ ~35%, 60㎡초과 ~15%)
AREA_BAND_WEIGHTS: dict[str, float] = {
    "33이하": 0.50,
    "33초과_60이하": 0.35,
    "60초과": 0.15,
}

# 문항 외(거절) 5문항 1:1 제외 주제 슬롯 (dev 2개 + test 3개, 주제 중복 0!)
DEV_UNANSWERABLE_TOPICS: list[str] = [
    "실시간 공실 여부, 지금 바로 입주 가능한 매물 추천, 월 관리비 또는 세탁기·냉장고 등 가전 옵션",
    "위도·경도 좌표 기반 특정 지하철역 반경 거리(예: 반경 500m 이내) 매물 개수·목록 조회",
]
TEST_UNANSWERABLE_TOPICS: list[str] = [
    "서울 외 지역(경기도·인천 등) 전월세 시세 또는 아파트 매매가·분양 청약 정보",
    "개별 계약의 전세사기 확정 판단, 등기부등본 근저당 안전 여부 등 개별 법률 자문",
    "버팀목 전세자금대출·청년 전세대출 등 금융 대출 상품의 개인별 금리·최대 한도 추천",
]

DATA_SCOPE_SUMMARY = """
[제공 데이터 범위]
- 서울특별시 25개 자치구의 최근 1년 체결 완료된 전월세 실거래 요약 통계만 보유.
- 보유 항목: 자치구, 법정동, 주택유형(아파트/단독다가구/연립다세대/오피스텔), 전월세 구분, 면적대,
  보증금·월세 통계(중앙값·평균·범위), 거래건수, 신규/갱신 건수, 건축연도, 대표 건물명(단독다가구는 비공개).
""".strip()


class DraftQA(BaseModel):
    """답변 가능 문항 구조화 출력 스키마."""

    user_input: str = Field(description="사용자가 실제로 물어볼 자연스러운 한국어 질문")
    reference: str = Field(description="핵심 사실 중심 2~3문장 간결 모범 정답 (기준 기간·정확한 금액 포함)")


class DraftUnanswerableQA(BaseModel):
    """문항 외 질문(근거 없음) 구조화 출력 스키마."""

    user_input: str = Field(description="서울 상경 청년이 물어볼 법하지만 제공 데이터로는 답할 수 없는 질문")
    absence_reason: str = Field(description="제공 데이터에서 답할 수 없는 구체적 사유 1문장")
    reference: str = Field(description="'확인할 수 없음.'으로 시작하며 사유를 간결히 안내하는 1~2문장 모범 거절 답변")


def _build_instruction(qtype: QuestionType, category: EvalCategory) -> str:
    if qtype == "단순 질문":
        return (
            "근거 문서에 명시된 특정 지역의 보증금·월세 중앙값, 거래 건수, 건축연도 중 "
            "핵심 사실 1~2개만 묻는 '단순 질문'과 2문장 이내 간결한 모범 정답을 만드세요."
        )
    if qtype == "복합 추론 질문" and category == "직접 검색":
        return (
            "제공된 2개 근거 문서를 비교하는 '복합 추론 질문'을 만드세요. "
            "단, 비교 포인트는 최대 2개까지만 묻고(예: 보증금·월세 중앙값 차이 + 건축연도 비교 또는 예산 적합성), "
            "모범 정답은 핵심 수치와 결론만 2~3문장으로 간결하게 쓰세요."
        )
    if qtype == "복합 추론 질문" and category == "표현 변형·유사 조건 구분":
        return (
            "제공된 2개 근거 문서는 지역·유형·전월세·면적대 조건이 비슷해 혼동하기 쉽습니다. "
            "유사한 두 조건을 정확히 구분해 비교하는 질문을 만들되 비교 포인트는 최대 2개로 제한하고, "
            "모범 정답은 핵심 수치와 구분 결론만 2~3문장으로 간결하게 쓰세요."
        )
    return (
        "근거 문서의 딱딱한 표기(연립다세대, 33㎡이하, 보증금 중앙값, 건축 연도)를 일상 구어체로 바꾼 "
        "'용어 다르게 질문(유사도)' 1개와 2문장 이내 간결한 모범 정답을 만드세요. "
        "(예: 연립다세대→빌라, 33㎡이하→원룸급/작은 집, 33~60㎡→소형/투룸급, 60㎡초과→넓은 중대형, "
        "전세→월세 없이 보증금만 내는 집, 월세→다달이 내는 집)"
    )


ANSWERABLE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 서울 청년 주거 RAG 시스템의 평가 데이터 엔지니어입니다.\n"
            "반드시 제공된 [근거 문서] 안의 사실과 숫자만 사용해 질문과 모범 정답을 만드세요.\n"
            "[필수 작성 규칙]\n"
            "1. 5대 검색 조건 누락 금지: 질문(user_input)에는 대상 문서가 유일하게 특정되도록 "
            "자치구·법정동·주택유형·전월세 구분·면적대 5가지 조건의 의미를 빠짐없이 반영하세요.\n"
            "2. 날짜 복사 금지(치팅 차단): 질문(user_input)에는 근거 문서의 구체적 시작일·종료일"
            "(예: '2025-11-28~2026-11-06', '2025년 11월 25일부터')을 절대 복사하지 말고 "
            "'최근 1년간' 또는 '최근 실거래 기준'으로만 표현하세요.\n"
            "3. 간결한 모범 정답(RAGAS ContextRecall 잡음 최소화): 모범 정답(reference)은 "
            "데이터 기준 기간(YYYY-MM-DD~YYYY-MM-DD), 정확한 금액(만원/원), 요청된 핵심 결론만 "
            "2~3문장으로 간결히 적고, 요청하지 않은 일반 주의사항이나 부연설명은 넣지 마세요.",
        ),
        (
            "human",
            "[질문 유형 지시]: {instruction}\n\n[근거 문서]:\n{context}",
        ),
    ]
)

UNANSWERABLE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 서울 청년 주거 RAG 시스템의 '문항 외 질문(근거 없음 · 거절 평가)'을 만드는 엔지니어입니다.\n"
            "지정된 [할당 제외 주제]에 딱 맞는 현실적인 질문 1개를 만드세요.\n"
            "제공 데이터만으로는 절대 답할 수 없어야 하며, 모범 정답은 '확인할 수 없음.'으로 시작하는 1~2문장으로 쓰세요.",
        ),
        (
            "human",
            "{scope}\n\n[할당 제외 주제]: {assigned_topic}\n"
            "[참고용 보유 자치구·동 예시]: {sample_areas}\n"
            "[이미 생성된 거절 질문(중복 금지)]: {previous_questions}",
        ),
    ]
)


def _load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"먼저 rent_chunker.py를 실행해 {path}를 생성하세요!")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rich = [r for r in rows if r.get("deal_count", 0) >= 10]
    return rich if len(rich) >= 60 else rows


def _stratified_shuffle_split(
    chunks: list[dict],
    dev_ratio: float = 0.35,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """자치구별로 셔플 후 dev/test 풀을 완전 분리한다 (청크 겹침 0 보장)."""
    rng = random.Random(seed)
    by_gu: dict[str, list[dict]] = {}
    for c in chunks:
        by_gu.setdefault(c.get("district", ""), []).append(c)

    dev_pool: list[dict] = []
    test_pool: list[dict] = []
    for gu in sorted(by_gu):
        rows = by_gu[gu].copy()
        rng.shuffle(rows)
        cut = max(1, int(len(rows) * dev_ratio))
        dev_pool.extend(rows[:cut])
        test_pool.extend(rows[cut:])

    rng.shuffle(dev_pool)
    rng.shuffle(test_pool)
    return dev_pool, test_pool


def _build_band_schedule(total_slots: int, seed: int) -> list[str]:
    """청년 타깃 면적대 가중치(50% / 35% / 15%)에 맞춘 결정론적 면적대 스케줄 생성."""
    n33 = round(total_slots * AREA_BAND_WEIGHTS["33이하"])
    n60 = round(total_slots * AREA_BAND_WEIGHTS["33초과_60이하"])
    n_over = max(0, total_slots - n33 - n60)
    schedule = ["33이하"] * n33 + ["33초과_60이하"] * n60 + ["60초과"] * n_over
    random.Random(seed).shuffle(schedule)
    return schedule


def _pop_chunk(
    pool: list[dict],
    target_band: str,
    used_ids: set[str],
    used_gu_cnt: Counter,
    prefer_district: str | None = None,
) -> dict:
    """아직 안 쓴 청크 중 target_band + 덜 나온 자치구를 우선 선택한다."""
    candidates = [c for c in pool if c["chunk_id"] not in used_ids]
    if not candidates:
        candidates = pool

    if prefer_district:
        same_gu = [c for c in candidates if c["district"] == prefer_district]
        if same_gu:
            same_gu_band = [c for c in same_gu if c.get("area_band") == target_band]
            chosen = same_gu_band[0] if same_gu_band else same_gu[0]
            used_ids.add(chosen["chunk_id"])
            used_gu_cnt[chosen["district"]] += 1
            return chosen

    band_cands = [c for c in candidates if c.get("area_band") == target_band]
    active = band_cands if band_cands else candidates
    # 덜 선택된 자치구를 우선 정렬해 25개 구가 골고루 나오게 분산
    active.sort(key=lambda c: used_gu_cnt[c["district"]])
    chosen = active[0]
    used_ids.add(chosen["chunk_id"])
    used_gu_cnt[chosen["district"]] += 1
    return chosen


def generate_split(
    pool: list[dict],
    split: Literal["dev", "test"],
    plan: list[tuple[QuestionType, EvalCategory, int]],
    unanswerable_topics: list[str],
    seed: int = 42,
    used_unanswerables: list[str] | None = None,
) -> list[dict]:
    llm = get_llm()
    ans_chain = ANSWERABLE_PROMPT | llm.with_structured_output(DraftQA)
    unans_chain = UNANSWERABLE_PROMPT | llm.with_structured_output(DraftUnanswerableQA)

    prev_unans = used_unanswerables if used_unanswerables is not None else []
    sample_areas = ", ".join(sorted({f"{c['district']} {c['dong']}" for c in pool[:40]})[:10])

    # 답변 가능 문항에 필요한 총 청크 슬롯 수 계산 후 청년 면적대 가중 스케줄 생성
    ans_chunk_slots = sum(
        (2 if qt == "복합 추론 질문" else 1) * cnt for qt, _, cnt in plan if qt != "문항 외 질문"
    )
    band_schedule = _build_band_schedule(ans_chunk_slots, seed=seed)
    band_idx = 0

    used_ids: set[str] = set()
    used_gu_cnt: Counter = Counter()
    total_count = sum(c for _, _, c in plan)
    unans_idx = 0

    items: list[dict] = []
    seq = 1

    for qtype, category, count in plan:
        for _ in range(count):
            qid = f"RENT-KH-{split.upper()}-{seq:02d}"
            print(f"  [{split} {seq:02d}/{total_count:02d}] {category} / {qtype} 생성 중...", flush=True)

            if qtype == "문항 외 질문":
                topic = unanswerable_topics[unans_idx % len(unanswerable_topics)]
                unans_idx += 1
                uq: DraftUnanswerableQA = unans_chain.invoke(
                    {
                        "scope": DATA_SCOPE_SUMMARY,
                        "assigned_topic": topic,
                        "sample_areas": sample_areas,
                        "previous_questions": "\n".join(prev_unans) if prev_unans else "없음",
                    }
                )
                prev_unans.append(uq.user_input)
                items.append(
                    {
                        "question_id": qid,
                        "split": split,
                        "domain": "rent",
                        "eval_category": category,
                        "question_type": qtype,
                        "expected_status": "insufficient_evidence",
                        "assigned_exclusion_topic": topic,
                        "user_input": uq.user_input,
                        "reference": uq.reference,
                        "reference_contexts": [],
                        "allowed_chunk_ids": [],
                        "source_location": "",
                        "absence_reason": uq.absence_reason,
                    }
                )
            else:
                if qtype == "복합 추론 질문" and category == "표현 변형·유사 조건 구분":
                    c1 = _pop_chunk(pool, band_schedule[band_idx], used_ids, used_gu_cnt)
                    band_idx += 1
                    c2 = _pop_chunk(
                        pool, band_schedule[band_idx], used_ids, used_gu_cnt, prefer_district=c1["district"]
                    )
                    band_idx += 1
                    picked = [c1, c2]
                elif qtype == "복합 추론 질문":
                    c1 = _pop_chunk(pool, band_schedule[band_idx], used_ids, used_gu_cnt)
                    band_idx += 1
                    c2 = _pop_chunk(pool, band_schedule[band_idx], used_ids, used_gu_cnt)
                    band_idx += 1
                    picked = [c1, c2]
                else:
                    c1 = _pop_chunk(pool, band_schedule[band_idx], used_ids, used_gu_cnt)
                    band_idx += 1
                    picked = [c1]

                context_text = "\n\n---\n\n".join(c["page_content"] for c in picked)
                qa: DraftQA = ans_chain.invoke(
                    {
                        "instruction": _build_instruction(qtype, category),
                        "context": context_text,
                    }
                )
                items.append(
                    {
                        "question_id": qid,
                        "split": split,
                        "domain": "rent",
                        "eval_category": category,
                        "question_type": qtype,
                        "expected_status": "answered",
                        "user_input": qa.user_input,
                        "reference": qa.reference,
                        "reference_contexts": [c["page_content"] for c in picked],
                        "allowed_chunk_ids": [c["chunk_id"] for c in picked],
                        "source_location": " | ".join(c["location"] for c in picked),
                        "absence_reason": "",
                    }
                )
            seq += 1

    return items


def validate_eval_drafts(
    dev_items: list[dict],
    test_items: list[dict],
    manifest_by_id: dict[str, dict],
) -> dict:
    """자동 품질 게이트: test 문항 본문을 노출하지 않고 객관적 품질 지표만 검증한다."""
    date_pat = re.compile(r"202[56][-\.년]\s*\d{1,2}[-\.월]\s*\d{1,2}")

    def _audit(items: list[dict]) -> dict:
        missing_ids: list[str] = []
        date_leaks: list[str] = []
        band_cnt: Counter = Counter()
        gu_cnt: Counter = Counter()
        cat_cnt: Counter = Counter()

        for it in items:
            cat_cnt[it["eval_category"]] += 1
            if date_pat.search(it["user_input"]):
                date_leaks.append(it["question_id"])
            for cid in it["allowed_chunk_ids"]:
                meta = manifest_by_id.get(cid)
                if not meta:
                    missing_ids.append(cid)
                else:
                    band_cnt[meta.get("area_band", "?")] += 1
                    gu_cnt[meta.get("district", "?")] += 1

        return {
            "total": len(items),
            "categories": dict(cat_cnt),
            "missing_chunk_ids": missing_ids,
            "date_leak_qids": date_leaks,
            "unique_districts": len(gu_cnt),
            "area_bands": dict(band_cnt),
        }

    dev_ids = {cid for it in dev_items for cid in it["allowed_chunk_ids"]}
    test_ids = {cid for it in test_items for cid in it["allowed_chunk_ids"]}
    overlap = sorted(dev_ids & test_ids)

    return {
        "dev": _audit(dev_items),
        "test": _audit(test_items),
        "dev_test_chunk_overlap": overlap,
        "all_passed": (
            not overlap
            and not _audit(dev_items)["missing_chunk_ids"]
            and not _audit(test_items)["missing_chunk_ids"]
            and not _audit(dev_items)["date_leak_qids"]
            and not _audit(test_items)["date_leak_qids"]
        ),
    }


def main() -> None:
    chunks = _load_manifest(RENT_MANIFEST_PATH)
    manifest_by_id = {c["chunk_id"]: c for c in chunks}
    KH_EVAL_DIR.mkdir(parents=True, exist_ok=True)

    dev_pool, test_pool = _stratified_shuffle_split(chunks, dev_ratio=0.35, seed=42)
    used_unans: list[str] = []

    dev_items = generate_split(
        dev_pool, "dev", DEV_PLAN, DEV_UNANSWERABLE_TOPICS, seed=42, used_unanswerables=used_unans
    )
    test_items = generate_split(
        test_pool, "test", TEST_PLAN, TEST_UNANSWERABLE_TOPICS, seed=99, used_unanswerables=used_unans
    )

    dev_path = KH_EVAL_DIR / "kh_rent_dev_draft.json"
    test_path = KH_EVAL_DIR / "kh_rent_test_draft.json"
    dev_path.write_text(json.dumps(dev_items, ensure_ascii=False, indent=2), encoding="utf-8")
    test_path.write_text(json.dumps(test_items, ensure_ascii=False, indent=2), encoding="utf-8")

    report = validate_eval_drafts(dev_items, test_items, manifest_by_id)
    print("\n==================== [자동 품질 게이트 결과] ====================")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\ndev 초안 저장: {dev_path} (육안 검수 후 kh_rent_dev.json 확정!)")
    print(f"test 초안 저장: {test_path} (자동 게이트 통과 시 열지 말고 kh_rent_test.json으로 봉인!)")


if __name__ == "__main__":
    main()
