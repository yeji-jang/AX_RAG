"""
# 평가셋 빌더 (지침서 STEP 2 기준 30문항: dev 10 + test 20)
- 입력: docs/02_chunked/공공임대_공고문_preV2_chunkV1_cs800_ov100.jsonl
- 출력:
    - docs/04-1_eval_set/{DATASET_VER}/dev.json  (10문항: direct 5, paraphrase_or_similar 3, unanswerable 2)
    - docs/04-1_eval_set/{DATASET_VER}/test.json (20문항: direct 10, paraphrase_or_similar 7, unanswerable 3)
"""
import json
import random
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from rag_home_api.test.yj.core.logger import get_logger
from rag_home_api.test.yj.core.schemas.evaluation import EvalQAItem
from rag_home_api.test.yj.core.vector_store import get_llm

logger = get_logger(__name__)

#===================================================================
# 1. 경로 및 설정
#===================================================================
DATASET_VER = "chunk_V1"
OVERWRITE = False       # 이미 검수된 평가셋 덮어쓰기 방지 (새 버전은 DATASET_VER 변경 권장)

CHUNK_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "02_chunked"
    / "공공임대_공고문_preV2_chunkV1_cs800_ov100.jsonl"
)
EVAL_DIR = Path(__file__).resolve().parents[2] / "docs" / "04-1_eval_set" / DATASET_VER
DEV_PATH = EVAL_DIR / "dev.json"
TEST_PATH = EVAL_DIR / "test.json"

RANDOM_SEED = 42


#===================================================================
# 2. LLM 초안 생성용 스키마 & 프롬프트
#===================================================================
class DraftQA(BaseModel):
    question: str = Field(description="실제 사용자가 물을 법한 자연스러운 한국어 질문")
    ground_truth: str = Field(description="주어진 청크 내용만을 근거로 한 정확한 모범 정답")
    required_facts: list[str] = Field(
        description="정답 충족 여부를 채점할 때 반드시 포함되어야 할 핵심 사실 2~3개 (예: ['보증금 84,400천원', '월임대료 351,660원'])"
    )


DIRECT_PROMPT = """당신은 서울시 공공임대주택 RAG 평가 데이터셋 작성 전문가입니다.
주어진 [청크 본문]을 보고 [직접 검색(direct)] 유형의 질문 1개, 모범 정답, 핵심 사실 목록을 만드세요.

규칙:
1. 질문에 청크 앞부분의 구체적인 공고명·단지명·주택유형·계층 중 식별 가능한 핵심 정보를 반드시 포함하세요.
2. 청크 본문에 명확히 적힌 사실·숫자·조건만 묻고, 원문 단위(천원/원/㎡/%)를 정확히 유지하세요.
3. "위 표에서", "이 공고에서" 같은 지시어는 쓰지 마세요."""

PARAPHRASE_PROMPT = """당신은 서울시 공공임대주택 RAG 평가 데이터셋 작성 전문가입니다.
주어진 [청크 본문]을 보고 [표현 변형·유사 조건 구분(paraphrase_or_similar)] 유형의 질문 1개, 모범 정답, 핵심 사실 목록을 만드세요.

규칙:
1. 질문에 구체적인 공고명·단지명·주택유형은 포함하되, 본문의 딱딱한 행정 용어를 일반인의 일상 표현으로 바꿔 묻거나(예: '임대보증금 증액 전환' → '보증금을 올려서 월세를 줄이는 경우', '무주택세대구성원' → '집 없는 가족 구성원'), 비슷한 계층·순위·예외 조건 중 특정 하나를 콕 집어 구분하도록 질문하세요.
2. 모범 정답은 반드시 청크 본문에 근거한 정확한 사실·숫자만 쓰세요."""


#===================================================================
# 3. 근거 없음(unanswerable) 5문항 (dev 2 + test 3)
#===================================================================
UNANSWERABLE_DEV: list[dict] = [
    {
        "question": "경기도 성남시 판교 행복주택의 2026년 예비입주자 모집 일정과 보증금은 얼마인가요?",
        "unanswerable_reason": "수집 범위가 서울시 LH/SH 공공임대 공고문으로 한정되어 경기도 성남시 판교 공고 정보가 문서에 없음",
    },
    {
        "question": "LH 행복주택 청약 시 시중은행 버팀목 전세대출의 우대금리와 최대 한도는 얼마인가요?",
        "unanswerable_reason": "모집공고문에는 주택 입주자격·임대조건만 수록되어 있고 시중은행 대출 상품 금리·한도 정보는 없음",
    },
]

UNANSWERABLE_TEST: list[dict] = [
    {
        "question": "부산광역시 해운대구 국민임대주택 예비입주자 모집 신청 기간은 언제인가요?",
        "unanswerable_reason": "수집 문서가 서울특별시 공고문으로 한정되어 부산광역시 공고 정보가 없음",
    },
    {
        "question": "2028년 시행 예정인 서울시 공공임대주택 소득기준 개편안의 1인 가구 상한액은 얼마인가요?",
        "unanswerable_reason": "수집 기준 시점(2025-11~2026-11 공고) 이후의 2028년 미래 정책 정보는 문서에 없음",
    },
    {
        "question": "LH수서1 행복주택 36A형의 2026년 11월 기준 예상 당첨 경쟁률과 커트라인 점수는 몇 점인가요?",
        "unanswerable_reason": "모집공고문에는 모집 기준과 일정만 기재되어 있으며 실제 청약 경쟁률·커트라인 결과는 없음",
    },
]


#===================================================================
# 4. 서로 다른 공고에서 표/본문 균형 샘플링 (dev 8 + test 17 = 총 25개 청크)
#===================================================================
def sample_chunks(path: Path, total_needed: int = 25) -> list[dict]:
    rows = [json.loads(line) for line in path.open("r", encoding="utf-8") if line.strip()]
    candidates = [r for r in rows if len(r["text"]) >= 350]

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(candidates)

    table_picked: list[dict] = []
    text_picked: list[dict] = []
    used_docs: set[str] = set()
    half = total_needed // 2  # 표 12개 + 본문 13개

    for row in candidates:
        doc_id = row["doc_id"]
        if doc_id in used_docs:
            continue

        is_table = "|---|" in row["text"]
        if is_table and len(table_picked) < half:
            table_picked.append(row)
            used_docs.add(doc_id)
        elif not is_table and len(text_picked) < (total_needed - half):
            text_picked.append(row)
            used_docs.add(doc_id)

        if len(table_picked) + len(text_picked) >= total_needed:
            break

    combined = table_picked + text_picked
    rng.shuffle(combined)
    return combined


#===================================================================
# 5. dev.json(10) / test.json(20) 생성 & EvalQAItem 검증 저장
#===================================================================
def _build_split(
    split: str,
    prefix: str,
    direct_chunks: list[dict],
    para_chunks: list[dict],
    unanswerables: list[dict],
) -> list[EvalQAItem]:
    llm = get_llm().with_structured_output(DraftQA)
    items: list[EvalQAItem] = []
    seq = 1

    # 1) direct 문항 생성
    for chunk in direct_chunks:
        draft: DraftQA = llm.invoke([
            SystemMessage(content=DIRECT_PROMPT),
            HumanMessage(content=f"[청크 본문]\n{chunk['text']}"),
        ])
        items.append(EvalQAItem(
            question_id=f"{prefix}-{seq:03d}",
            split=split,
            query_type="direct",
            question=draft.question,
            ground_truth=draft.ground_truth,
            required_facts=draft.required_facts,
            allowed_doc_ids=[chunk["doc_id"]],
            allowed_chunk_ids=[chunk["chunk_id"]],
            source_location=f"{chunk['doc_id']} p.{chunk['page']}",
        ))
        logger.info("[%s-%03d] (direct) %s", prefix, seq, draft.question)
        seq += 1

    # 2) paraphrase_or_similar 문항 생성
    for chunk in para_chunks:
        draft: DraftQA = llm.invoke([
            SystemMessage(content=PARAPHRASE_PROMPT),
            HumanMessage(content=f"[청크 본문]\n{chunk['text']}"),
        ])
        items.append(EvalQAItem(
            question_id=f"{prefix}-{seq:03d}",
            split=split,
            query_type="paraphrase_or_similar",
            question=draft.question,
            ground_truth=draft.ground_truth,
            required_facts=draft.required_facts,
            allowed_doc_ids=[chunk["doc_id"]],
            allowed_chunk_ids=[chunk["chunk_id"]],
            source_location=f"{chunk['doc_id']} p.{chunk['page']}",
        ))
        logger.info("[%s-%03d] (paraphrase) %s", prefix, seq, draft.question)
        seq += 1

    # 3) unanswerable 문항 추가
    for u in unanswerables:
        items.append(EvalQAItem(
            question_id=f"{prefix}-{seq:03d}",
            split=split,
            query_type="unanswerable",
            question=u["question"],
            ground_truth="정보 부족 안내",
            required_facts=[],
            allowed_doc_ids=[],
            allowed_chunk_ids=[],
            source_location="",
            unanswerable_reason=u["unanswerable_reason"],
        ))
        logger.info("[%s-%03d] (unanswerable) %s", prefix, seq, u["question"])
        seq += 1

    return items


def build_and_save_eval_datasets() -> None:
    if not OVERWRITE and (DEV_PATH.exists() or TEST_PATH.exists()):
        raise FileExistsError(
            f"이미 평가셋이 존재합니다({EVAL_DIR}). 새로 만들려면 DATASET_VER를 바꾸거나 OVERWRITE=True로 설정하세요."
        )
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    picked = sample_chunks(CHUNK_PATH, total_needed=25)

    # dev(답변 가능 8: direct 5 + para 3) / test(답변 가능 17: direct 10 + para 7) — 공고 겹침 없음!
    dev_items = _build_split("dev", "DEV", picked[0:5], picked[5:8], UNANSWERABLE_DEV)
    test_items = _build_split("test", "TEST", picked[8:18], picked[18:25], UNANSWERABLE_TEST)

    DEV_PATH.write_text(
        json.dumps([i.model_dump() for i in dev_items], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    TEST_PATH.write_text(
        json.dumps([i.model_dump() for i in test_items], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n저장 완료! dev={len(dev_items)}문항 ({DEV_PATH.name}), test={len(test_items)}문항 ({TEST_PATH.name})")


def load_eval_dataset(split: str = "dev") -> list[EvalQAItem]:
    path = DEV_PATH if split == "dev" else TEST_PATH
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [EvalQAItem(**item) for item in raw]


if __name__ == "__main__":
    build_and_save_eval_datasets()