"""
# 1단계 : "공공임대_공고문.jsonl" 원본 문서 전처리

(순서)
     1. 원본 데이터 경로와 저장할 경로
        - version : 데이터의 버전 (예 : "V1", "V2") 수동으로 지정
     2. 전처리 방식
        1) LH 하단 워터마크 제거
        2) PDF 제어문자 제거
        3) 페이지 하단
        4) 불필요한 공백 & 과도한 빈 줄 정리
    3. JSONL 원본 읽고 전처리된 데이터를 저장할 경로로 새로 저장
    4. 프로그램 실행
"""


#===================================================================
# 1. 원본 데이터를 가져올 경로와 저장할 경로를 지정
#===================================================================

import json
from pathlib import Path



# 원본 데이터 경로
RAW_JSONL_PATH = Path(__file__).resolve().parents[3] / "shared" / "docs" / "공공임대_공고문.jsonl"

# 저장할 전처리 데이터 경로
version = "V1" # 데이터의 버전 (예 : "V1", "V2")

OUT_JSONL_PATH = Path(__file__).resolve().parents[1] / "docs" / "01_preprocessed" / f"{RAW_JSONL_PATH.stem}_{version}.jsonl"


#===================================================================
# 2. 원본 데이터 전처리 구현 함수
#===================================================================

import re

def processing_rawdata(text : str) -> str :
    # 1. LH 하단 워터마크 제거
    #   모든 페이지 맨 똑같은 문구
    text = re.sub(r"공사 승인없이 무단유출 시 법적책임이 있습니다\.[\s\S]*$", "", text)

    # 2. PDF 제어문자 제거
    #   영구임대주택과 \u0007영구임대주택을 다른 단어로 인식할 수 있기 때문
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

    # 3. 하단 페이지 번호 + 공고명 반복 푸터 제거
    #    예 : "\n7\n2026년 영구임대주택 예비입주자 모집공고"
    text = re.sub(r"\n\d{1,3}\n[^\n]*모집공고\s*$", "", text)

    # 4. 줄 끝 공백 & 3줄 이상 연속 빈 줄 정리
    #    (표 안의 |  | 는 건드리지 않음)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

#===================================================================
# 3. 전처리된 파일 OUT_JSONL_PATH로 저장
#===================================================================

def save_preprocessed_jsonl() -> None :
    # 1. 저장할 폴더가 없으면 자동 생성
    OUT_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)

    total = 0    # 전체 문서 수
    saved = 0    # 전처리 후 저장된 문서 수
    skipped = 0  # 전처리 후 비어있어서 건너뛴 문서 수

    # 2. 원본 읽기와 출력 파일 쓰기
    with(
        RAW_JSONL_PATH.open("r", encoding="utf-8") as f_in,
        OUT_JSONL_PATH.open("w", encoding="utf-8") as f_out
    ) :
        for line in f_in :
            line = line.strip()

            # 3. line이 비어있으면 건너뛰기
            if not line :
                skipped += 1
                continue

            total += 1
            data = json.loads(line)
            text = data.get("text", "")

            # 전처리 수행 전, text가 비어있으면 건너뛰기
            if not text :
                skipped += 1
                continue

            # 전처리 수행
            processed_text = processing_rawdata(text)

            # 전처리 수행 후, processed_text가 비어있으면 건너뛰기
            if not processed_text :
                skipped += 1
                continue

            # 전처리된 텍스트를 JSONL로 저장
            data["text"] = processed_text
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
            saved += 1

    print(f"Total: {total}, Saved: {saved}, Skipped: {skipped}")


#===================================================================
# 4. 실행
#===================================================================
if __name__ == "__main__" :
    save_preprocessed_jsonl()