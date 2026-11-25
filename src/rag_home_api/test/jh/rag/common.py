"""공통: 경로·청크 불러오기·평가셋.

청크는 PGVector에 저장할 때(chunktest.ipynb 1~5단계)와 같은 방법으로 JSONL을 다시 잘라 만든다.
→ 같은 입력·같은 설정이면 결과가 같으므로, BM25와 PGVector가 같은 청크 ID를 쓴다.
"""

import hashlib
import json
import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_home_api.config import ROOT, settings

load_dotenv(ROOT / ".env")  # OPENAI_API_KEY·DATABASE_URL (값은 출력하지 않음)
DB_URL = os.environ["DATABASE_URL"]

JH = ROOT / "src" / "rag_home_api" / "test" / "jh"
DOCS = ROOT / "src" / "rag_home_api" / "shared" / "docs"
EVAL = JH / "평가셋"
EVAL_FILES = ["법령파트_평가셋.json", "후보풀_공공임대.json"]
RESULTS = JH / "results"
CACHE = JH / "out"  # 깃 제외(.gitignore) — BM25 토큰 캐시

FILES = ["공공임대_공고문", "청년안심주택_공고문", "일반전월세_법령가이드"]

# 청크 설정 — PGVector 서랍과 반드시 같아야 함
CHUNKING = {
    "jh_500": {"chunk_size": 500, "chunk_overlap": 100, "tag": "s500"},
    "jh_1000": {"chunk_size": 1000, "chunk_overlap": 200, "tag": "s1000"},
}

MODELS = {"embedding": settings.embedding_model}


def load_documents() -> list[Document]:
    """JSONL 한 줄 = Document 하나 (NUL 제거)."""
    docs = []
    for name in FILES:
        with open(DOCS / f"{name}.jsonl", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                docs.append(Document(
                    page_content=row["text"].replace("\x00", ""),
                    metadata={"row_id": row["id"], "doc_id": row["doc_id"], "title": row["title"],
                              "source": row["source"], "category": row["category"],
                              "page": row.get("page"), "article": row.get("article")},
                ))
    return docs


def load_chunks(collection: str) -> list[Document]:
    """자르기 + 청크 ID(`<줄id>_<태그>_cNN`). PGVector 저장 때와 같은 결과."""
    cfg = CHUNKING[collection]
    splitter = RecursiveCharacterTextSplitter(chunk_size=cfg["chunk_size"], chunk_overlap=cfg["chunk_overlap"])
    chunks = splitter.split_documents(load_documents())
    count: dict[str, int] = {}
    for c in chunks:
        rid = c.metadata["row_id"]
        count[rid] = count.get(rid, 0) + 1
        c.metadata["chunk_id"] = f"{rid}_{cfg['tag']}_c{count[rid]:02d}"
    return chunks


def load_eval(split: str) -> dict:
    """두 평가셋 파일에서 split(dev/test) 문항만 모아 번호순으로."""
    items = []
    for name in EVAL_FILES:
        items += [q for q in json.loads((EVAL / name).read_text(encoding="utf-8"))["items"] if q["split"] == split]
    items.sort(key=lambda q: int(q["question_id"].split("-")[-1]))
    return {"version": f"평가셋({eval_sha()})", "items": items}


def eval_sha() -> str:
    """평가셋 두 파일 내용의 해시 — 결과 로그에 남겨 어느 평가셋으로 돌렸는지 추적."""
    h = hashlib.sha256()
    for name in EVAL_FILES:
        h.update((EVAL / name).read_bytes())
    return h.hexdigest()[:12]
