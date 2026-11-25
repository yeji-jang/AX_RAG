"""검색 A / B — 튜터 12_rag 방식, 모든 단계를 LangChain 기능으로.

A: 벡터 검색기  vectorstore.as_retriever(search_kwargs={"k": n})
B: BM25Retriever.from_documents(..., preprocess_func=토크나이저) + 벡터 검색기
   → EnsembleRetriever(retrievers=[bm25, vector], weights=[0.5, 0.5])   (안에서 RRF로 합침)

개선 1 거의 같은 글 합치기: EmbeddingsRedundantFilter → DocumentCompressorPipeline
       → ContextualCompressionRetriever(base_compressor, base_retriever)
개선 2 문서 종류 나눠 찾기: RunnableBranch(조건, 전체 검색기) / 법령·가이드 검색기(as_retriever filter)
개선 3 BM25 단어 정리: kiwi_tokenizer의 품사·제외 단어 목록

단계별 설정은 PRESETS에 모아 두고 --preset으로 골라 같은 코드로 r0 → 개선 3을 다시 돌릴 수 있게 함.
"""

import re

from kiwipiepy import Kiwi
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_classic.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_community.retrievers import BM25Retriever
from langchain_core.runnables import RunnableBranch
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

from common import DB_URL, MODELS, load_chunks

CONFIG = {
    "k": 5,  # 최종 상위 k (Hit@5)
    "candidates": 20,  # B에서 BM25·벡터 검색기가 각각 가져올 후보 수 (A도 거의 같은 글 합치기를 할 때 20에서 고름)
    "weights": [0.5, 0.5],  # EnsembleRetriever 가중치 [BM25, 벡터] — 같으면 순수 RRF와 순서가 같음
    "rrf_c": 60,  # EnsembleRetriever의 RRF 상수(기본값): 점수 = Σ 가중치 / (순위 + c)
    "bm25_pos": ["NNG", "NNP", "NR", "SN", "SL", "SH", "XR", "VV", "VA"],  # BM25에 쓰는 품사
    "bm25_stopwords": [],
    "redundant_threshold": None,  # EmbeddingsRedundantFilter 코사인 유사도 기준(기본값 0.95). None이면 안 함
    "route": False,  # 문서 종류 나눠 찾기
    "notice_words": ["공고", "모집", "행복주택", "청년안심", "국민임대", "영구임대", "매입임대", "통합공공임대", "공공임대",
                     "장기전세", "LH", "SH", "단지", "입주자격", "예비입주자"],
    "notice_patterns": [r"\d{2,3}[A-Z]형?", r"[가-힣]{2,}역(?![가-힣])"],  # 주택형(14A·36B), 역 이름(신독산역 등)
    "law_categories": ["일반 전월세 법령", "일반 전월세 가이드"],
}

# 단계별 설정 (실험기록 6절 순서)
PRESETS = {
    "r0": {},  # 기본 A/B
    "dedupe": {"redundant_threshold": 0.95},  # 개선 1: 여러 공고에 반복되는 거의 같은 안내문을 하나로
    "route": {"redundant_threshold": 0.95, "route": True},  # 개선 2: 공고·단지 말이 없으면 법령·가이드에서만
    "tok": {"redundant_threshold": 0.95, "route": True,  # 개선 3: BM25에서 일반 동사·흔한 명사 제외
            "bm25_pos": ["NNG", "NNP", "NR", "SN", "SL", "SH", "XR"],
            "bm25_stopwords": ["때", "경우", "얼마", "몇", "사항", "해당", "등", "것", "수", "후"]},
}


def apply_preset(name: str) -> None:
    base = {"bm25_pos": ["NNG", "NNP", "NR", "SN", "SL", "SH", "XR", "VV", "VA"], "bm25_stopwords": [],
            "redundant_threshold": None, "route": False}
    CONFIG.update(base | PRESETS[name])


_kiwi = Kiwi()


def kiwi_tokenizer(text: str) -> list[str]:
    """BM25 토크나이저(튜터 custom_tokenizer 자리). 공백 분리 대신 kiwipiepy 형태소 — 팀 결정."""
    return [t.form for t in _kiwi.tokenize(text)
            if t.tag in CONFIG["bm25_pos"] and t.form not in CONFIG["bm25_stopwords"]]


def route(question: str) -> str:
    """'law' = 법령·가이드만, 'all' = 전체. 공고·단지를 가리키는 말이 있거나 애매하면 전체."""
    if not CONFIG["route"]:
        return "all"
    if any(w in question for w in CONFIG["notice_words"]) or any(re.search(p, question) for p in CONFIG["notice_patterns"]):
        return "all"
    return "law"


class Retriever:
    def __init__(self, collection: str):
        self.collection = collection
        self.chunks = load_chunks(collection)
        self.by_id = {c.metadata["chunk_id"]: c for c in self.chunks}
        self.embeddings = OpenAIEmbeddings(model=MODELS["embedding"])
        self.vectorstore = PGVector(embeddings=self.embeddings, connection=DB_URL,
                                    collection_name=collection, use_jsonb=True)  # 연결만 — pre_delete_collection 쓰지 않음
        law_chunks = [c for c in self.chunks if c.metadata["category"] in CONFIG["law_categories"]]
        # 범위(all / law)마다 검색기를 하나씩 만듦. BM25Retriever에는 거르기 기능이 없어서 법령·가이드 청크로 따로 만듦
        self.bm25 = {"all": self._bm25(self.chunks), "law": self._bm25(law_chunks) if CONFIG["route"] else None}
        self.retrievers = {}  # 방식(A/B)별 최종 검색기

    @staticmethod
    def _bm25(docs) -> BM25Retriever:
        bm25 = BM25Retriever.from_documents(documents=docs, preprocess_func=kiwi_tokenizer)
        bm25.k = CONFIG["candidates"]
        return bm25

    def vector_retriever(self, k: int, scope: str):
        search_kwargs = {"k": k}
        if scope == "law":
            search_kwargs["filter"] = {"category": {"$in": CONFIG["law_categories"]}}
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)

    def hybrid_retriever(self, scope: str) -> EnsembleRetriever:
        return EnsembleRetriever(retrievers=[self.bm25[scope], self.vector_retriever(CONFIG["candidates"], scope)],
                                 weights=CONFIG["weights"], c=CONFIG["rrf_c"])

    def scoped_retriever(self, mode: str, scope: str):
        """범위 하나의 검색기. 개선 1이면 ContextualCompressionRetriever로 거의 같은 글을 거름."""
        if mode == "A":
            n = CONFIG["candidates"] if CONFIG["redundant_threshold"] else CONFIG["k"]
            base = self.vector_retriever(n, scope)
        elif mode == "B":
            base = self.hybrid_retriever(scope)
        else:
            raise ValueError(f"지원하지 않는 검색 방식: {mode}")
        if not CONFIG["redundant_threshold"]:
            return base
        pipeline = DocumentCompressorPipeline(transformers=[
            EmbeddingsRedundantFilter(embeddings=self.embeddings, similarity_threshold=CONFIG["redundant_threshold"]),
        ])
        return ContextualCompressionRetriever(base_compressor=pipeline, base_retriever=base)

    def retriever(self, mode: str):
        """최종 검색기. 개선 2면 RunnableBranch로 질문에 따라 전체 / 법령·가이드 검색기를 고름."""
        if mode not in self.retrievers:
            if CONFIG["route"]:
                self.retrievers[mode] = RunnableBranch(
                    (lambda q: route(q) == "all", self.scoped_retriever(mode, "all")),
                    self.scoped_retriever(mode, "law"),
                )
            else:
                self.retrievers[mode] = self.scoped_retriever(mode, "all")
        return self.retrievers[mode]

    def search(self, mode: str, query: str) -> list[dict]:
        """상위 k개를 로그용 dict로. 실패 원인을 보려고 BM25·벡터 각자의 순위도 함께 적음."""
        docs = self.retriever(mode).invoke(query)[:CONFIG["k"]]
        hits = [{"chunk_id": d.metadata["chunk_id"], "row_id": d.metadata["row_id"], "doc_id": d.metadata["doc_id"],
                 "rank": i + 1} for i, d in enumerate(docs)]
        if mode == "B":  # 기록용: 두 검색기 각자의 순위(합치기 전)
            scope = route(query)
            b_rank = {d.metadata["chunk_id"]: i + 1 for i, d in enumerate(self.bm25[scope].invoke(query))}
            v_rank = {d.metadata["chunk_id"]: i + 1
                      for i, d in enumerate(self.vector_retriever(CONFIG["candidates"], scope).invoke(query))}
            for h in hits:
                h["bm25_rank"], h["vector_rank"] = b_rank.get(h["chunk_id"]), v_rank.get(h["chunk_id"])
        return hits

    def text(self, chunk_id: str) -> str:
        return self.by_id[chunk_id].page_content
