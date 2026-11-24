"""구조 기반 CSV 청킹. 전처리·본문·Document 생성·청킹만 수행한다.

모듈 import 시 파일을 읽지 않는다. chunk_csv()를 호출하거나 아래처럼 직접 실행한다.
실행: .venv/bin/python src/rag_home_api/test/bc/structural_chunk.py
임베딩·외부 API·DB 연결·업로드는 포함하지 않는다.
"""

import argparse
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document


# 파일 위치를 기준으로 저장소 루트를 계산해 현재 작업 디렉터리에 의존하지 않는다.
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CSV_PATH = PROJECT_ROOT / "src/rag_home_api/shared/docs/서울_전월세_최근1년_관악구.csv"


def load_and_clean(csv_path: str | Path = DEFAULT_CSV_PATH) -> pd.DataFrame:
    """CSV를 읽어 기존 정제 규칙을 적용한다. 원본 파일·거래 행은 삭제하지 않는다."""
    # ID·지번의 앞자리 0이 사라지지 않도록 우선 문자열로 읽기
    df = pd.read_csv(
        csv_path,
        encoding="utf-8-sig",
        dtype="string",
        keep_default_na=False,
    )

    # 앞뒤 공백을 제거하고 빈 문자열만 결측값으로 변경
    df = df.apply(lambda column: column.str.strip())
    df = df.replace("", pd.NA)

    # 사용하지 않는 컬럼 제외
    df = df.drop(
        columns=["prev_deposit_manwon", "prev_rent_manwon", "renewal_right_used"],
        errors="ignore",
    )

    # 숫자형 메타데이터를 유지해 나중에 금액·면적 필터에 사용
    numeric_types = {
        "area_m2": "Float64",
        "floor": "Int64",
        "build_year": "Int64",
        "deposit_manwon": "Int64",
        "monthly_rent_manwon": "Int64",
    }
    for column, dtype in numeric_types.items():
        cleaned = df[column].str.replace(",", "", regex=False)
        numbers = pd.to_numeric(cleaned, errors="raise")

        # 정수 변환 시 소수점이 잘리는 것을 방지
        if dtype == "Int64" and (numbers.dropna() % 1 != 0).any():
            raise ValueError(f"{column}에 정수가 아닌 값이 존재합니다.")
        df[column] = numbers.astype(dtype)

    # 계약일을 날짜 타입으로 변환
    df["contract_date"] = pd.to_datetime(
        df["contract_date"], format="%Y-%m-%d", errors="raise"
    )
    return df


# 문단별 항목 정의: (컬럼명, 표시할 이름, 단위)
TEXT_GROUPS = [
    # 첫 번째 문단: 위치
    [
        ("region_gu", "자치구", ""),
        ("region_dong", "법정동", ""),
        ("jibun", "지번", ""),
        ("building_name", "건물명", ""),
    ],
    # 두 번째 문단: 주택 정보
    [
        ("housing_type", "주택 유형", ""),
        ("area_m2", "임대면적", "㎡"),
        ("floor", "층", "층"),
        ("build_year", "건축연도", "년"),
    ],
    # 세 번째 문단: 계약 정보
    [
        ("lease_type", "전월세 구분", ""),
        ("contract_date", "계약일", ""),
        ("contract_period", "계약기간", ""),
        ("new_or_renewal", "계약구분", ""),
    ],
    # 네 번째 문단: 현재 계약의 금액
    [
        ("deposit_manwon", "보증금", "만원"),
        ("monthly_rent_manwon", "월세", "만원"),
    ],
]

def row_to_text(row):
    # 완성된 문단을 저장할 목록
    paragraphs = []

    for group in TEXT_GROUPS:
        # 한 문단에 들어갈 항목들을 저장
        lines = []

        for column, label, unit in group:
            value = row[column]

            # 빈 항목은 생략하지만 실제 0은 유지
            if pd.isna(value):
                continue

            # 날짜를 YYYY-MM-DD 형태로 표시
            if column == "contract_date":
                value = value.strftime("%Y-%m-%d")

            # 금액에 천 단위 쉼표 표시
            elif column.endswith("_manwon"):
                value = f"{value:,}"

            # 항목명·값·단위를 한 줄로 구성
            lines.append(f"{label}: {value}{unit}")

        # 항목 사이는 줄바꿈으로 연결
        if lines:
            paragraphs.append("\n".join(lines))

    # 문단 사이는 빈 줄로 연결
    return "\n\n".join(paragraphs)


def create_documents(df: pd.DataFrame, source_file: str | Path) -> list[Document]:
    """거래마다 본문과 메타데이터를 묶는다. 날짜는 문자열, 금액은 숫자로 유지한다."""
    # DataFrame을 거래별 딕셔너리 목록으로 변환
    records = df.to_dict(orient="records")

    # 완성된 Document를 저장할 목록
    documents = []

    for row in records :
        # row_to_text 함수로 거래 한 건의 본문 생성
        text = row_to_text(row)

        # 검색 필터와 원본 거래 추적에 사용할 정보
        metadata = {}

        for column, value in row.items():
            # 빈 항목은 메타데이터에서도 생략
            if pd.isna(value):
                continue

            # 날짜 객체는 추후 JSON으로 저장 할 수 있도록 문자열로 변환
            if isinstance(value, pd.Timestamp):
                value = value.strftime("%Y-%m-%d")

            # 숫자는 문자열로 변환하지 않고 그대로 저장
            metadata[column] = value

        # 출처 파일명 추가
        metadata["source_file"] = Path(source_file).name

        # 거래별 Document 생성
        document = Document(
            id=row["id"],
            page_content=text,
            metadata=metadata
        )

        # 목록에 추가
        documents.append(document)
    return documents


def structural_chunk(documents):
    # 구조 기반 청킹 결과를 저장할 목록
    chunks = []

    for document in documents:
        # 원본 거래 ID로 청크 ID 생성
        transaction_id = document.metadata["id"]
        chunk_id = f"{transaction_id}_structural_000"

        # 거래 전체 본문을 유지하면서 청크 정보 추가
        chunk = Document(
            id=chunk_id,
            page_content=document.page_content,
            metadata={
                **document.metadata,       # 원본 메타데이터 복사
                "method": "structural",     # 청킹 방식
                "chunk_index": 0,           # 거래 내 청크 순번
                "chunk_count": 1,           # 거래당 청크 수
            },
        )

        chunks.append(chunk)

    return chunks


def chunk_csv(csv_path: str | Path = DEFAULT_CSV_PATH) -> list[Document]:
    """CSV 전처리부터 구조 기반 청킹까지 실행하고 결과 Document 목록을 반환한다."""
    df = load_and_clean(csv_path)
    documents = create_documents(df, csv_path)
    return structural_chunk(documents)


def main() -> None:
    """터미널에서 실행할 때 입력 경로를 받아 청킹 결과의 기본 상태를 출력한다."""
    parser = argparse.ArgumentParser(description="CSV 전처리 및 구조 기반 청킹")
    parser.add_argument("--input", type=Path, default=DEFAULT_CSV_PATH, help="원본 CSV 경로")
    args = parser.parse_args()
    chunks = chunk_csv(args.input)

    # 결과 파일을 자동 생성하지 않고 반환된 청크의 개수·길이와 예시만 확인한다.
    lengths = [len(chunk.page_content) for chunk in chunks]
    print("원본 거래 수:", len({chunk.metadata["id"] for chunk in chunks}))
    print("청크 수:", len(chunks))
    print("평균 길이:", round(sum(lengths) / len(lengths), 2) if lengths else 0)
    print("최대 길이:", max(lengths, default=0))
    if chunks:
        print("\n첫 청크 ID:", chunks[0].id)
        print(chunks[0].page_content)


# 다른 파일에서 import할 때는 실행하지 않고, 직접 실행할 때만 main을 호출한다.
if __name__ == "__main__":
    main()
