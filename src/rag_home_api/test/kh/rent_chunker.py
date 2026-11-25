"""서울 전월세 실거래 CSV(21열 중 확정 11열만 유지) → 동·유형·면적대 집계 블록 청킹 & PGVector 적재.

작성자: 최기혁 (kh)
유지 11열:
  - 그룹키(5): region_gu, region_dong, housing_type, lease_type, area_m2
  - 집계(5): deposit_manwon, monthly_rent_manwon, building_name, contract_date, new_or_renewal
  - 선택(1): build_year
제외 10열:
  id, jibun, floor, contract_period, deposit_range, rent_range,
  prev_deposit_manwon, prev_rent_manwon, renewal_right_used, source
"""

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from langchain_core.documents import Document

from rag_home_api.config import DATA_DIR, SHARED_DOCS_DIR, settings
from rag_home_api.shared.core.logger import get_logger
from rag_home_api.shared.core.vector_store import get_raw_db_connection, get_vector_store

logger = get_logger(__name__)

# kh 개인 작업 폴더 및 전월세 컬렉션 상수 (팀 3개 분야 통합 시 collection_name만 맞춰 그대로 승격)
KH_DIR = Path(__file__).resolve().parent
KH_EVAL_DIR = KH_DIR / "eval"
KH_RESULTS_DIR = KH_DIR / "results"

RENT_COLLECTION_NAME = "housing_rent_kh"
RENT_INDEX_VERSION = "rent-v2-11cols-src"
RENT_MANIFEST_PATH = DATA_DIR / "chunk_manifest_rent_kh.jsonl"


def _area_band(area_m2: float, include_pyeong: bool = False) -> tuple[str, str]:
    """면적을 청년 주거 탐색 기준 3개 구간으로 분류한다."""
    if area_m2 <= 33.0:
        return "33이하", ("33㎡이하(원룸급·10평 이하)" if include_pyeong else "33㎡이하(원룸급)")
    if area_m2 <= 60.0:
        return "33초과_60이하", ("33~60㎡(소형·10~18평)" if include_pyeong else "33~60㎡(소형)")
    return "60초과", ("60㎡초과(중대형·18평 초과)" if include_pyeong else "60㎡초과(중대형)")


def _fmt_manwon(manwon: float) -> str:
    """만원 단위 숫자를 '1억 2,000만원 (120,000,000원)' 형식으로 안전 변환한다."""
    val = int(round(manwon))
    krw = val * 10_000
    if val == 0:
        return "0원"
    eok, rem = divmod(val, 10_000)
    if eok > 0 and rem > 0:
        pretty = f"{eok}억 {rem:,}만원"
    elif eok > 0:
        pretty = f"{eok}억원"
    else:
        pretty = f"{rem:,}만원"
    return f"{pretty} ({krw:,}원)"


def _safe_year(raw: str) -> int | None:
    try:
        y = int(float(raw))
        return y if 1900 <= y <= 2030 else None
    except (ValueError, TypeError):
        return None


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_rent_block_chunks(
    districts: list[str] | None = None,
    min_deals_per_chunk: int = 1,
    include_pyeong: bool = False,
    recent_examples: int = 3,
    index_version: str = RENT_INDEX_VERSION,
) -> list[Document]:
    """전월세 CSV에서 확정 11열만 읽어 (자치구, 법정동, 주택유형, 전월세, 면적대) 블록 청크를 만든다.

    Args:
        districts: 처리할 자치구 리스트 (예: ["강남구", "관악구", "종로구"]). None이면 25개 구 전체.
        min_deals_per_chunk: 최소 거래 건수 미만 그룹 제외 시 사용 (기본 1 = 전체 보존).
        include_pyeong: True면 청크 본문에 평수 환산(10평 이하 / 10~18평 / 18평 초과, 약 N평) 병기.
        recent_examples: 청크당 포함할 최근 실거래 사례 건수 (기본 3, 실험용 0/3/5 변경 가능).
        index_version: 청크 버전 식별자 (캐시 무효화 및 청킹 전략 추적용).
    """
    files = sorted(SHARED_DOCS_DIR.glob("서울_전월세_최근1년_*.csv"))
    if districts is not None:
        target_set = set(districts)
        files = [f for f in files if f.stem.replace("서울_전월세_최근1년_", "") in target_set]

    if not files:
        raise FileNotFoundError(f"대상 전월세 CSV를 찾을 수 없습니다: districts={districts}")

    # 1단계: 확정 11열만 추출해 그룹 버킷으로 집계 (나머지 10개 열 제외)
    groups: dict[tuple[str, str, str, str, str, str], list[dict]] = defaultdict(list)
    skipped_empty_lease = 0

    for csv_path in files:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as fp:
            for row in csv.DictReader(fp):
                lease_type = (row.get("lease_type") or "").strip()
                if not lease_type:
                    skipped_empty_lease += 1
                    continue

                gu = (row.get("region_gu") or "").strip()
                dong = (row.get("region_dong") or "").strip()
                h_type = (row.get("housing_type") or "").strip()
                area = float(row["area_m2"]) if row.get("area_m2") else 0.0
                band_code, band_label = _area_band(area, include_pyeong=include_pyeong)

                groups[(gu, dong, h_type, lease_type, band_code, band_label)].append(
                    {
                        "area_m2": area,
                        "deposit_manwon": float(row["deposit_manwon"] or 0),
                        "monthly_rent_manwon": float(row["monthly_rent_manwon"] or 0),
                        "building_name": (row.get("building_name") or "").strip(),
                        "contract_date": (row.get("contract_date") or "").strip(),
                        "new_or_renewal": (row.get("new_or_renewal") or "").strip(),
                        "build_year": _safe_year((row.get("build_year") or "").strip()),
                    }
                )

    if skipped_empty_lease:
        logger.info("lease_type 공백 행 %d건 제외 완료", skipped_empty_lease)

    # 2단계: 그룹별 '표가 잘리지 않는 1개 요약 블록 청크' 생성
    docs: list[Document] = []
    for (gu, dong, h_type, lease_type, band_code, band_label), rows in sorted(groups.items()):
        if len(rows) < min_deals_per_chunk:
            continue

        deposits = [r["deposit_manwon"] for r in rows]
        rents = [r["monthly_rent_manwon"] for r in rows]
        areas = [r["area_m2"] for r in rows]
        dates = [r["contract_date"] for r in rows if r["contract_date"]]
        years = [r["build_year"] for r in rows if r["build_year"] is not None]

        nr_cnt = Counter(r["new_or_renewal"] for r in rows if r["new_or_renewal"] in {"신규", "갱신"})

        b_cnt = Counter(r["building_name"] for r in rows if r["building_name"])
        if b_cnt:
            top_buildings = ", ".join(f"{name}({cnt}건)" for name, cnt in b_cnt.most_common(5))
        else:
            top_buildings = "단독다가구 원천 데이터 특성상 건물명 비공개"

        recent_rows = sorted(rows, key=lambda r: r["contract_date"], reverse=True)[:recent_examples]
        recent_lines = []
        for rr in recent_rows:
            bname = rr["building_name"] or "건물명비공개"
            yr_txt = f"{rr['build_year']}년식" if rr["build_year"] else "연식미기재"
            nr = rr["new_or_renewal"] if rr["new_or_renewal"] in {"신규", "갱신"} else "미기재"
            if lease_type == "전세":
                price_txt = f"보증금 {_fmt_manwon(rr['deposit_manwon'])}"
            else:
                price_txt = (
                    f"보증금 {_fmt_manwon(rr['deposit_manwon'])} / "
                    f"월세 {_fmt_manwon(rr['monthly_rent_manwon'])}"
                )
            recent_lines.append(
                f"  - {rr['contract_date']} | {bname}({yr_txt}) | {rr['area_m2']:.2f}㎡ | {price_txt} ({nr})"
            )

        period_str = f"{min(dates)} ~ {max(dates)}" if dates else "기간 미기재"
        if years:
            year_str = f"{min(years)}~{max(years)}년식 (중앙값 {int(statistics.median(years))}년식)"
        else:
            year_str = "건축연도 미기재"

        source_file = f"서울_전월세_최근1년_{gu}.csv"
        med_area = statistics.median(areas)
        pyeong_suffix = f" (약 {med_area / 3.3058:.1f}평)" if include_pyeong else ""
        synonym_line = []
        if include_pyeong:
            h_alias = {"연립다세대": "빌라/다세대", "단독다가구": "단독/다가구/원룸건물", "오피스텔": "오피스텔", "아파트": "아파트"}.get(h_type, h_type)
            l_alias = "보증금만 내는 전세" if lease_type == "전세" else "다달이 임대료 내는 월세"
            synonym_line = [f"- 쉬운 설명: {gu} {dong} {h_alias}, {l_alias}, {band_label}"]

        lines = [
            f"[서울 전월세 실거래 요약 | {gu} {dong} | {h_type} | {lease_type} | {band_label}]",
            f"- 출처 파일: {source_file}",
            *synonym_line,
            f"- 기준 기간: {period_str} (최근 1년 체결 완료 실거래, 총 {len(rows):,}건)",
            f"- 면적 요약: 중앙값 {med_area:.2f}㎡{pyeong_suffix} (최소 {min(areas):.2f}㎡ ~ 최대 {max(areas):.2f}㎡)",
            (
                f"- 보증금 통계: 중앙값 {_fmt_manwon(statistics.median(deposits))} | "
                f"평균 {_fmt_manwon(statistics.mean(deposits))} | "
                f"범위 {_fmt_manwon(min(deposits))} ~ {_fmt_manwon(max(deposits))}"
            ),
        ]
        if lease_type == "월세":
            lines.append(
                f"- 월세 통계: 중앙값 {_fmt_manwon(statistics.median(rents))} | "
                f"평균 {_fmt_manwon(statistics.mean(rents))} | "
                f"범위 {_fmt_manwon(min(rents))} ~ {_fmt_manwon(max(rents))}"
            )
        else:
            lines.append("- 월세 통계: 전세 계약 (월세 0원)")

        lines.extend(
            [
                f"- 계약 구분: 신규 {nr_cnt.get('신규', 0):,}건 / 갱신 {nr_cnt.get('갱신', 0):,}건",
                f"- 건축 연도: {year_str}",
                f"- 주요 거래 건물·단지(상위): {top_buildings}",
                "- 최근 실거래 사례:",
                *recent_lines,
                "- 주의: 체결 완료된 과거 실거래 통계이며 실시간 공실·관리비·가전 옵션 정보는 포함되지 않음.",
            ]
        )

        text = "\n".join(lines)
        chunk_id = f"RENT_{gu}_{dong}_{h_type}_{lease_type}_{band_code}"

        meta = {
            "chunk_id": chunk_id,
            "doc_id": f"rent_{gu}_{dong}",
            "doc_type": "private_rent",
            "title": f"{gu} {dong} {h_type} {lease_type} {band_label} 실거래 시세",
            "source": source_file,
            "category": f"{h_type}_{lease_type}",
            "location": f"{source_file} ({dong}·{h_type}·{lease_type}·{band_label})",
            "district": gu,
            "dong": dong,
            "housing_type": h_type,
            "lease_type": lease_type,
            "area_band": band_code,
            "deal_count": len(rows),
            "min_build_year": min(years) if years else None,
            "max_build_year": max(years) if years else None,
            "min_deposit_krw": int(round(min(deposits))) * 10_000,
            "median_deposit_krw": int(round(statistics.median(deposits))) * 10_000,
            "min_monthly_rent_krw": int(round(min(rents))) * 10_000,
            "median_monthly_rent_krw": int(round(statistics.median(rents))) * 10_000,
            "index_version": index_version,
            "content_hash": _sha256(text),
        }
        docs.append(Document(page_content=text, metadata=meta))

    logger.info("전월세 블록 청킹 완료(11열): %d개 파일 -> %d개 청크", len(files), len(docs))
    return docs


def build_rent_index(
    districts: list[str] | None = None,
    collection_name: str = RENT_COLLECTION_NAME,
    embedding_model: str | None = None,
    include_pyeong: bool = False,
    recent_examples: int = 3,
    index_version: str | None = None,
    rebuild: bool = True,
    batch_size: int = 100,
) -> dict:
    """전월세 블록 청크를 매니페스트 저장 후 PGVector 전용 컬렉션에 안전하게 분리 적재한다."""
    chosen_emb = embedding_model or settings.embedding_model
    chosen_ver = index_version or (
        f"rent-v3-pyeong-r{recent_examples}-{chosen_emb.split('/')[-1]}"
        if include_pyeong or recent_examples != 3 or chosen_emb != settings.embedding_model
        else RENT_INDEX_VERSION
    )
    docs = load_rent_block_chunks(
        districts=districts,
        include_pyeong=include_pyeong,
        recent_examples=recent_examples,
        index_version=chosen_ver,
    )

    manifest_path = (
        RENT_MANIFEST_PATH
        if collection_name == RENT_COLLECTION_NAME
        else DATA_DIR / f"chunk_manifest_{collection_name}.jsonl"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as fp:
        for d in docs:
            row = {"page_content": d.page_content, **d.metadata}
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("전월세 매니페스트 저장 완료: %s (%d건)", manifest_path.name, len(docs))

    store = get_vector_store(collection_name=collection_name, embedding_model=chosen_emb)
    if rebuild:
        with get_raw_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM langchain_pg_embedding
                    WHERE collection_id IN (
                        SELECT uuid FROM langchain_pg_collection WHERE name = %s
                    )
                    """,
                    (collection_name,),
                )
            conn.commit()
        logger.info("rebuild=True: 기존 컬렉션(%s) 행 삭제 완료", collection_name)

    existing: dict[str, str] = {}
    with get_raw_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT e.id, e.cmetadata->>'content_hash'
                FROM langchain_pg_embedding e
                JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                WHERE c.name = %s
                """,
                (collection_name,),
            )
            existing = {row[0]: (row[1] or "") for row in cur.fetchall()}

    to_insert = [
        d for d in docs if existing.get(d.metadata["chunk_id"]) != d.metadata["content_hash"]
    ]
    total_batches = (len(to_insert) + batch_size - 1) // batch_size
    for idx in range(0, len(to_insert), batch_size):
        batch = to_insert[idx : idx + batch_size]
        store.add_documents(batch, ids=[d.metadata["chunk_id"] for d in batch])
        logger.info(
            "전월세 PGVector 적재 배치 %d/%d 완료 (%d건)",
            idx // batch_size + 1,
            total_batches,
            len(batch),
        )

    return {
        "collection_name": collection_name,
        "embedding_model": chosen_emb,
        "index_version": chosen_ver,
        "include_pyeong": include_pyeong,
        "recent_examples": recent_examples,
        "districts": districts or "ALL_25",
        "total_chunks": len(docs),
        "inserted_chunks": len(to_insert),
        "skipped_chunks": len(docs) - len(to_insert),
        "manifest_path": str(manifest_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="전월세 블록 청킹 & PGVector 컬렉션 분리 적재기")
    parser.add_argument("--collection", default=RENT_COLLECTION_NAME, help="적재할 PGVector 컬렉션명 (기본: housing_rent_kh)")
    parser.add_argument("--embedding-model", default=None, help="임베딩 모델 (예: text-embedding-3-small, text-embedding-3-large, nlpai-lab/KURE-v1)")
    parser.add_argument("--include-pyeong", action="store_true", help="청크 본문에 평수 환산·쉬운 별칭 1줄 추가")
    parser.add_argument("--recent-examples", type=int, default=3, help="청크당 최근 실거래 사례 수 (기본: 3)")
    parser.add_argument("--districts", nargs="*", default=None, help="일부 자치구만 적재할 때 지정 (비우면 25개 구 전체)")
    parser.add_argument("--no-rebuild", action="store_true", help="기존 컬렉션 행 유지 후 증분 적재")
    args = parser.parse_args()

    summary = build_rent_index(
        districts=args.districts,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        include_pyeong=args.include_pyeong,
        recent_examples=args.recent_examples,
        rebuild=not args.no_rebuild,
    )
    print(summary)
