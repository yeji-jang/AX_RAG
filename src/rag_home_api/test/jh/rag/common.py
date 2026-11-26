import hashlib
import json
import os
import re
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

JH = Path(__file__).resolve().parents[1]
ROOT = JH.parents[3]
DOCS = ROOT / "src" / "rag_home_api" / "shared" / "docs"
EVAL = JH / "평가셋"
EVAL_FILES = ["법령파트_평가셋.json", "후보풀_공공임대.json"]
APPROVAL_FILE = JH / "execution.local.json"  # 팀장이 검토하여 작성, Git 제외. 자동 생성하지 않음.
API_BASE = "https://api.openai.com/v1"
RESULTS = JH / "results"
CACHE = JH / "out"  # 깃 제외(.gitignore) — BM25 토큰 캐시

FILES = ["공공임대_공고문", "청년안심주택_공고문", "일반전월세_법령가이드"]

# 청크 설정 — PGVector 서랍과 반드시 같아야 함
CHUNKING = {
    "jh_500": {"chunk_size": 500, "chunk_overlap": 100, "tag": "s500"},
    "jh_1000": {"chunk_size": 1000, "chunk_overlap": 200, "tag": "s1000"},
}

MODELS = {"embedding": "text-embedding-3-small", "generation": "gpt-6-luna"}


def runtime_config() -> dict:
    """외부 실행 승인 뒤에만 호출. 정적 검증/도움말은 .env/키/DB를 읽지 않는다."""
    import os
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    connection = os.environ.get("DATABASE_URL")
    if not connection:
        raise ValueError("[RuntimeConfigMissing] DATABASE_URL 설정이 필요합니다.")
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise ValueError("[RuntimeConfigMissing] OPENAI_API_KEY 설정이 필요합니다.")
    return {"connection": connection.replace("postgresql://", "postgresql+psycopg://", 1), "api_key": key,
            "embedding": os.environ.get("EMBEDDING_MODEL", MODELS["embedding"]),
            "generation": os.environ.get("LLM_MODEL", MODELS["generation"])}


def database_connection() -> str:
    """색인 사전 점검은 DB 주소만 필요하며 API 키/모델 클라이언트를 읽지 않는다."""
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    value = os.environ.get("DATABASE_URL", "").strip()
    if not value:
        raise ValueError("[RuntimeConfigMissing] DATABASE_URL 설정이 필요합니다.")
    return value


def corpus_fingerprint(documents) -> str:
    rows = sorted((d.metadata["chunk_id"], d.page_content, d.metadata) for d in documents)
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def law_chunks(collection: str) -> list[Document]:
    return [d for d in load_chunks(collection) if d.metadata["category"] == "일반 전월세 법령"
            and d.metadata["doc_id"].startswith("LAW_")]


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
                              "page": row.get("page"), "article": row.get("article"),
                              "source_url": row.get("source_url")},
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
    return eval_digest()[:12]


def eval_digest() -> str:
    h = hashlib.sha256()
    for name in EVAL_FILES:
        h.update((EVAL / name).read_bytes())
    return h.hexdigest()


def local_manifest(collection: str, *, documents=None) -> dict:
    docs = law_chunks(collection) if documents is None else documents
    return {"eval_sha": eval_sha(), "eval_sha256": eval_digest(), "collection": collection,
            "chunking": CHUNKING[collection], "law_chunks": len(docs),
            "corpus_sha256": corpus_fingerprint(docs), "embedding_model": MODELS["embedding"],
            "generation_model": MODELS["generation"], "api_base": API_BASE}


def require_external_scope(collection: str, *, split: str, modes: list[str], preset: str,
                           rewrite: bool, generate: bool) -> tuple[dict, list[Document]]:
    """팀장이 지정한 로컬 승인 범위를 키/DB 초기화 전에 검사한다."""
    try:
        approved = json.loads(APPROVAL_FILE.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raise PermissionError("[ApprovalRequired] 팀장이 승인한 execution.local.json 설정이 필요합니다.") from None
    except (OSError, ValueError):
        raise ValueError("[ApprovalConfigInvalid] execution.local.json을 읽을 수 없습니다.") from None
    if not isinstance(approved, dict) or type(approved.get("version")) is not int or approved["version"] != 1:
        raise ValueError("[ApprovalConfigInvalid] version=1인 JSON 객체가 필요합니다.")
    if approved.get("approved") is not True:
        raise PermissionError("[ApprovalNotGranted] 팀장의 명시적 승인이 필요합니다.")
    required = {"eval_sha", "eval_sha256", "collection", "corpus_sha256", "index_sha256",
                "embedding_model", "generation_model", "api_base", "splits", "modes",
                "preset", "rewrite", "generate"}
    if not required.issubset(approved):
        raise ValueError("[ApprovalConfigInvalid] 로컬 승인 설정에 필수 필드가 누락됐습니다.")
    documents = law_chunks(collection)
    manifest = local_manifest(collection, documents=documents)
    for field in ("eval_sha", "eval_sha256", "collection", "corpus_sha256",
                  "embedding_model", "generation_model", "api_base"):
        if approved.get(field) != manifest[field]:
            raise PermissionError(f"[ApprovalMismatch] {field} 승인값이 현재 자료/설정과 다릅니다.")
    if not re.fullmatch(r"[0-9a-f]{64}", str(approved.get("index_sha256", ""))):
        raise ValueError("[ApprovalConfigInvalid] index_sha256 승인 지문이 필요합니다.")
    splits = ["dev", "test"] if split == "all" else [split]
    for field, values, valid in (("splits", splits, {"dev", "test"}),
                                 ("modes", modes, {"A", "B", "C"})):
        allowed = approved.get(field)
        if (not isinstance(allowed, list) or not allowed or
                any(not isinstance(v, str) or v not in valid for v in allowed)):
            raise ValueError(f"[ApprovalConfigInvalid] {field} 승인 범위가 필요합니다.")
        if not set(values).issubset(allowed):
            raise PermissionError(f"[ApprovalScopeMismatch] {field} 요청이 승인 범위를 벗어났습니다.")
    if approved.get("preset") != preset:
        raise PermissionError("[ApprovalScopeMismatch] preset 요청이 승인 범위를 벗어났습니다.")
    for field, requested in (("rewrite", rewrite), ("generate", generate)):
        if type(approved.get(field)) is not bool:
            raise ValueError(f"[ApprovalConfigInvalid] {field}는 true/false여야 합니다.")
        if requested and not approved[field]:
            raise PermissionError(f"[ApprovalScopeMismatch] {field} 실행은 승인되지 않았습니다.")
    return approved, documents
