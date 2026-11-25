"""검색 A / B.

A: 벡터 검색만 (PGVector 코사인 거리, 상위 5)
B: BM25(kiwipiepy 형태소) + 벡터 → RRF로 합쳐 상위 5
"""

import pickle

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
}

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

    def _tokens(self) -> list[list[str]]:
        CACHE.mkdir(exist_ok=True)
        path = CACHE / f"bm25_tokens_{self.collection}_{len(self.chunks)}.pkl"
        if path.exists():
            return pickle.loads(path.read_bytes())
        toks = [[t.form for t in sent if t.tag in CONFIG["bm25_pos"]]
                for sent in _kiwi.tokenize([c.page_content for c in self.chunks])]
        path.write_bytes(pickle.dumps(toks))
        return toks

    # --- A ---
    def vector(self, query: str, n: int) -> list[dict]:
        hits = self.store.similarity_search_with_score(query, k=n)
        return [{"chunk_id": d.metadata["chunk_id"], "row_id": d.metadata["row_id"], "doc_id": d.metadata["doc_id"],
                 "vector_rank": i + 1, "distance": round(float(s), 4)} for i, (d, s) in enumerate(hits)]

    # --- BM25 ---
    def bm25_search(self, query: str, n: int) -> list[dict]:
        scores = self.bm25.get_scores(tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[:n]
        return [{"chunk_id": self.chunks[i].metadata["chunk_id"], "row_id": self.chunks[i].metadata["row_id"],
                 "doc_id": self.chunks[i].metadata["doc_id"], "bm25_rank": r + 1, "bm25": round(float(scores[i]), 3)}
                for r, i in enumerate(top) if scores[i] > 0]

    # --- B ---
    def hybrid(self, query: str, n: int) -> list[dict]:
        merged: dict[str, dict] = {}
        for lst, key in [(self.vector(query, CONFIG["vector_candidates"]), "vector_rank"),
                         (self.bm25_search(query, CONFIG["bm25_candidates"]), "bm25_rank")]:
            for h in lst:
                m = merged.setdefault(h["chunk_id"], {k: h[k] for k in ("chunk_id", "row_id", "doc_id")} | {"rrf": 0.0})
                m.update({k: v for k, v in h.items() if k not in m})
                m["rrf"] += 1 / (CONFIG["rrf_k"] + h[key])
        out = sorted(merged.values(), key=lambda m: -m["rrf"])[:n]
        for i, m in enumerate(out):
            m["rrf"] = round(m["rrf"], 5)
            m["rrf_rank"] = i + 1
        return out

    def search(self, mode: str, query: str) -> list[dict]:
        k = CONFIG["k"]
        if mode == "A":
            return self.vector(query, k)
        if mode == "B":
            return self.hybrid(query, k)
        raise ValueError(f"지원하지 않는 검색 방식: {mode}")

    def text(self, chunk_id: str) -> str:
        return self.by_id[chunk_id].page_content
