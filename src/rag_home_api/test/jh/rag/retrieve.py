"""검색 A / B.

A: 벡터 검색만 (PGVector 코사인 거리, 상위 5)
B: BM25(kiwipiepy 형태소) + 벡터 → RRF로 합쳐 상위 5
"""

import pickle
import re
from difflib import SequenceMatcher

from kiwipiepy import Kiwi
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from rank_bm25 import BM25Okapi

from common import CACHE, DB_URL, MODELS, load_chunks

# 설정값 (실험 기록에 그대로 남김)
CONFIG = {
    "k": 5,  # 최종 상위 k (Hit@5)
    "vector_candidates": 20,  # RRF에 넣을 벡터 후보 수
    "bm25_candidates": 20,  # RRF에 넣을 BM25 후보 수
    "rrf_k": 60,  # RRF 상수: 점수 = Σ 1/(rrf_k + 순위)
    "bm25_pos": ["NNG", "NNP", "NR", "SN", "SL", "SH", "XR", "VV", "VA"],  # BM25에 쓰는 품사
    "dedupe": "chunk_id",  # 중복 제거 기준
    # r1: 여러 공고에 거의 같은 안내문이 반복돼 상위를 독차지 → 공백 뺀 글자 유사도 0.9 이상이면 먼저 나온 하나만 남김
    "near_dup_ratio": 0.9,
    # r1-route: 문서 종류 나눠 찾기. 공고·단지를 가리키는 말이 없으면 '일반 전월세' 질문 → 법령·가이드에서만 찾음.
    # 공고 말이 있거나 애매하면 전체에서 찾음(잘못 좁혀서 놓치는 것을 막기 위해).
    "route": True,
    "notice_words": ["공고", "모집", "행복주택", "청년안심", "국민임대", "영구임대", "매입임대", "통합공공임대", "공공임대",
                     "장기전세", "LH", "SH", "단지", "입주자격", "예비입주자"],
    "notice_patterns": [r"\d{2,3}[A-Z]형?", r"[가-힣]{2,}역(?![가-힣])"],  # 주택형(14A·36B), 역 이름(신독산역 등)
    "law_categories": ["일반 전월세 법령", "일반 전월세 가이드"],
}


def route(question: str) -> str:
    """'law' = 법령·가이드만, 'all' = 전체."""
    if not CONFIG.get("route"):
        return "all"
    if any(w in question for w in CONFIG["notice_words"]) or any(re.search(p, question) for p in CONFIG["notice_patterns"]):
        return "all"
    return "law"

_kiwi = Kiwi()


def tokenize(text: str) -> list[str]:
    return [t.form for t in _kiwi.tokenize(text) if t.tag in CONFIG["bm25_pos"]]


class Retriever:
    def __init__(self, collection: str):
        self.collection = collection
        self.chunks = load_chunks(collection)
        self.by_id = {c.metadata["chunk_id"]: c for c in self.chunks}
        self.store = PGVector(embeddings=OpenAIEmbeddings(model=MODELS["embedding"]), connection=DB_URL,
                              collection_name=collection, use_jsonb=True)  # 연결만 — pre_delete_collection 쓰지 않음
        self.bm25 = BM25Okapi(self._tokens())
        self.law_idx = [i for i, c in enumerate(self.chunks) if c.metadata["category"] in CONFIG["law_categories"]]

    def _tokens(self) -> list[list[str]]:
        CACHE.mkdir(exist_ok=True)
        path = CACHE / f"bm25_tokens_{self.collection}_{len(self.chunks)}.pkl"
        if path.exists():
            return pickle.loads(path.read_bytes())
        toks = [[t.form for t in sent if t.tag in CONFIG["bm25_pos"]]
                for sent in _kiwi.tokenize([c.page_content for c in self.chunks])]
        path.write_bytes(pickle.dumps(toks))
        return toks

    def near_dedupe(self, hits: list[dict], n: int) -> list[dict]:
        """순서대로 보며, 이미 고른 것과 거의 같은 글(유사도 ≥ near_dup_ratio)은 건너뜀. 건너뛴 것은 dup_of_me로 기록."""
        ratio = CONFIG.get("near_dup_ratio")
        if not ratio:
            return hits[:n]
        kept, norm = [], {}
        for h in hits:
            t = re.sub(r"\s+", "", self.by_id[h["chunk_id"]].page_content)
            dup = next((k for k in kept if SequenceMatcher(None, norm[k["chunk_id"]], t).quick_ratio() >= ratio
                        and SequenceMatcher(None, norm[k["chunk_id"]], t).ratio() >= ratio), None)
            if dup:
                dup.setdefault("dup_of_me", []).append(h["chunk_id"])
                continue
            norm[h["chunk_id"]] = t
            kept.append(h)
            if len(kept) == n:
                break
        return kept

    # --- A ---
    def vector(self, query: str, n: int, scope: str = "all") -> list[dict]:
        flt = {"category": {"$in": CONFIG["law_categories"]}} if scope == "law" else None
        hits = self.store.similarity_search_with_score(query, k=n, filter=flt)
        return [{"chunk_id": d.metadata["chunk_id"], "row_id": d.metadata["row_id"], "doc_id": d.metadata["doc_id"],
                 "vector_rank": i + 1, "distance": round(float(s), 4)} for i, (d, s) in enumerate(hits)]

    # --- BM25 ---
    def bm25_search(self, query: str, n: int, scope: str = "all") -> list[dict]:
        scores = self.bm25.get_scores(tokenize(query))
        pool = self.law_idx if scope == "law" else range(len(scores))
        top = sorted(pool, key=lambda i: -scores[i])[:n]
        return [{"chunk_id": self.chunks[i].metadata["chunk_id"], "row_id": self.chunks[i].metadata["row_id"],
                 "doc_id": self.chunks[i].metadata["doc_id"], "bm25_rank": r + 1, "bm25": round(float(scores[i]), 3)}
                for r, i in enumerate(top) if scores[i] > 0]

    # --- B ---
    def hybrid(self, query: str, n: int, scope: str = "all") -> list[dict]:
        merged: dict[str, dict] = {}
        for lst, key in [(self.vector(query, CONFIG["vector_candidates"], scope), "vector_rank"),
                         (self.bm25_search(query, CONFIG["bm25_candidates"], scope), "bm25_rank")]:
            for h in lst:
                m = merged.setdefault(h["chunk_id"], {k: h[k] for k in ("chunk_id", "row_id", "doc_id")} | {"rrf": 0.0})
                m.update({k: v for k, v in h.items() if k not in m})
                m["rrf"] += 1 / (CONFIG["rrf_k"] + h[key])
        out = self.near_dedupe(sorted(merged.values(), key=lambda m: -m["rrf"]), n)
        for i, m in enumerate(out):
            m["rrf"] = round(m["rrf"], 5)
            m["rrf_rank"] = i + 1
        return out

    def search(self, mode: str, query: str, scope: str = "all") -> list[dict]:
        k = CONFIG["k"]
        if mode == "A":  # 거의 같은 글 합치기를 A에도 똑같이 적용 — 비교 대상은 검색 방식만
            return self.near_dedupe(self.vector(query, CONFIG["vector_candidates"], scope), k)
        if mode == "B":
            return self.hybrid(query, k, scope)
        raise ValueError(f"지원하지 않는 검색 방식: {mode}")

    def text(self, chunk_id: str) -> str:
        return self.by_id[chunk_id].page_content
