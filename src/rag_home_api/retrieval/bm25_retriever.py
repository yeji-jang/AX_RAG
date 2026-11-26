import re
from functools import lru_cache

from kiwipiepy import Kiwi
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from rag_home_api.config import settings
from rag_home_api.shared.core.logger import get_logger
from rag_home_api.shared.core.vector_store import get_raw_db_connection

logger = get_logger(__name__)

# Kiwi 품사 태그: 일반명사(NNG), 고유명사(NNP), 외국어(SL), 어근(XR)
# - 순수 숫자(SN)와 의존명사(NNB)는 평수 숫자('10', '18')가 무관한 날짜('03-18')·월세('10만원')와
#   충돌하는 원인이 되어 제외하고, 영숫자 코드·법정동 'N가'·법령 조문번호만 전용 정규식으로 보존한다.
KEEP_POS = {"NNG", "NNP", "SL", "XR"}
KO_STOPWORDS = frozenset(
    {
        "요즘", "보통", "얼마", "정도", "어디", "대체로", "어느", "비교", "생각",
        "경우", "사항", "관련", "기준", "내용", "확인", "안내", "전체", "이상", "이하", "초과", "미만",
    }
)
# 영문자가 최소 1개 포함된 코드만 보존 (sh_306214, 25b, 36a 등 — 순수 숫자/날짜 10, 18, 2026-03-18 배제)
_ALPHA_CODE_RE = re.compile(r"(?=[a-z0-9_\-]*[a-z])[a-z0-9][a-z0-9_\-]{1,}")
# 법정동 숫자 접미사 원형 보존 (성수동2가, 명륜3가, 문래동4가 등)
_DONG_GA_RE = re.compile(r"[가-힣]{2,}\d+가")
# 법령 조문번호 원형 보존 (제7조, 제3조의3 등)
_LAW_ARTICLE_RE = re.compile(r"제\d+조(?:의\d+)?")


@lru_cache(maxsize=1)
def _get_kiwi() -> Kiwi:
    """Kiwi 형태소 분석기는 로딩 비용이 있으므로 프로세스당 1회만 생성한다."""
    return Kiwi()


def tokenize_korean_housing(text: str) -> list[str]:
    """Kiwi 명사 형태소 + 영숫자 주택형/공고 코드(sh_306214, 25B) + 법정동 N가 + 법령 조문번호를 보존한다."""
    tokens: list[str] = []

    # ① 영문자 포함 코드 보존 (순수 숫자·날짜의 2중 추출·오폭 방지)
    for code in _ALPHA_CODE_RE.findall(text.lower()):
        tokens.append(code)
        for sep in ("_", "-"):
            if sep in code:
                tokens.extend(
                    part for part in code.split(sep) if len(part) > 1 and any(ch.isalpha() for ch in part)
                )

    # ② 법정동 N가 & 법령 조문번호 원형 보존
    tokens.extend(_DONG_GA_RE.findall(text))
    tokens.extend(_LAW_ARTICLE_RE.findall(text))

    # ③ 한국어 형태소 분석 ("대항력이랑" -> "대항력", "확정일자를" -> "확정일자")
    for tok in _get_kiwi().tokenize(text):
        if tok.tag in KEEP_POS and len(tok.form) > 1 and tok.form not in KO_STOPWORDS:
            tokens.append(tok.form.lower())

    return tokens


@lru_cache(maxsize=2)
def _load_corpus_cached(
    collection_name: str,
    index_version: str,
) -> tuple[list[Document], BM25Okapi | None]:
    """PGVector의 동일 청크 코퍼스를 1회 로드해 BM25Okapi 역색인을 메모리에 유지한다."""
    sql = """
        SELECT e.document, e.cmetadata
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
        WHERE c.name = %s
    """
    docs: list[Document] = []
    with get_raw_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (collection_name,))
            for content, meta in cur.fetchall():
                docs.append(Document(page_content=content, metadata=meta or {}))

    if not docs:
        return [], None

    logger.info("BM25 코퍼스 로드 완료: %s (%d개 청크)", collection_name, len(docs))
    tokenized_corpus = [tokenize_korean_housing(d.page_content) for d in docs]
    return docs, BM25Okapi(tokenized_corpus)


def _match_filters(meta: dict, filters: dict | None) -> bool:
    """vector_retriever._build_pg_filter와 동일한 필터 조건을 BM25에도 적용한다."""
    if not filters:
        return True
    for key in (
        "doc_type",
        "district",
        "dong",
        "housing_type",
        "lease_type",
        "area_band",
        "source",
        "category",
    ):
        if filters.get(key) and meta.get(key) != filters[key]:
            return False
    if filters.get("max_deposit_krw") is not None:
        dep = meta.get("min_deposit_krw")
        if dep is None or dep > int(filters["max_deposit_krw"]):
            return False
    if filters.get("max_monthly_rent_krw") is not None:
        rent = meta.get("min_monthly_rent_krw")
        if rent is None or rent > int(filters["max_monthly_rent_krw"]):
            return False
    return True


def search_bm25(
    query: str,
    top_k: int = 12,
    filters: dict | None = None,
    collection_name: str | None = None,
) -> list[dict]:
    """rank_bm25 + Kiwi 토크나이저로 키워드 매칭 상위 청크를 검색한다."""
    target_collection = collection_name or settings.collection_name
    docs, bm25 = _load_corpus_cached(target_collection, settings.index_version)
    if not docs or bm25 is None:
        return []

    scores = bm25.get_scores(tokenize_korean_housing(query))
    paired: list[tuple[Document, float]] = [
        (doc, float(sc))
        for doc, sc in zip(docs, scores)
        if sc > 0 and _match_filters(doc.metadata, filters)
    ]
    paired.sort(key=lambda x: x[1], reverse=True)

    return [
        {
            "chunk_id": doc.metadata["chunk_id"],
            "document": doc,
            "bm25_rank": rank,
            "bm25_score": round(sc, 4),
        }
        for rank, (doc, sc) in enumerate(paired[:top_k], start=1)
    ]