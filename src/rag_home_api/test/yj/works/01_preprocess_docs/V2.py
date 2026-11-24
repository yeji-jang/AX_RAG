"""
# 1단계 : "공공임대_공고문.jsonl" 원본 문서 전처리 (V2)

- 원본 : 데이터 v1 (PR #9, 2026-11-24 병합)
    - 원본에서 이미 처리됨 : 제어문자·NBSP·쪽 번호 제거, 끊긴 문장 잇기, house_type 통일
    - 원본 칸 변경 : has_table, spacing_restored 삭제 (13칸 → 11칸)

(V1 대비 변경점)
    1) 워터마크 : 문구만 지우던 방식 → 문구가 들어 있는 줄 전체 제거
        - 새 원본은 "IP·해시·시각 + 문구"가 한 줄에 붙어 있음 (LH 1건, 32쪽)
    2) 단어 사이 공백 2칸 이상 → 1칸 (표 줄은 제외)
    3) 경로 : 폴더가 한 단계 깊어져 parents[3] → [4], parents[1] → [2]

(순서)
    1. 원본 데이터 경로와 저장할 경로
        - version : 전처리 버전 (예 : "V1", "V2") 수동으로 지정
    2. 전처리 방식
        1) LH 하단 워터마크 줄 제거
        2) PDF 제어문자 제거 (새 원본은 0건, 안전장치로 유지)
        3) 페이지 하단 쪽 번호 + 공고명 반복 푸터 제거 (새 원본은 0건, 안전장치로 유지)
        4) 줄 끝 공백 & 3줄 이상 연속 빈 줄 정리
        5) 단어 사이 2칸 이상 공백 → 1칸 (표 줄 제외)
    3. JSONL 원본 읽고 전처리된 데이터를 저장할 경로로 새로 저장
    4. 프로그램 실행
"""


#===================================================================
# 1. 원본 데이터를 가져올 경로와 저장할 경로를 지정
#===================================================================

import json
from pathlib import Path



# 원본 데이터 경로
RAW_JSONL_PATH = Path(__file__).resolve().parents[4] / "shared" / "docs" / "공공임대_공고문.jsonl"

# 저장할 전처리 데이터 경로
version = "V2" # 데이터의 버전 (예 : "V1", "V2")

OUT_JSONL_PATH = Path(__file__).resolve().parents[2] / "docs" / "01_preprocessed" / f"{RAW_JSONL_PATH.stem}_{version}.jsonl"


#===================================================================
# 2. 원본 데이터 전처리 구현 함수
#===================================================================

import re

def processing_rawdata(text : str) -> str :
    # 1. LH 하단 워터마크 제거
    #   모든 페이지 맨 똑같은 문구
    # text = re.sub(r"공사 승인없이 무단유출 시 법적책임이 있습니다\.[\s\S]*$", "", text)
    text = re.sub(r"\n?[^\n]*공사 승인없이 무단유출 시 법적책임이 있습니다\.[^\n]*", "", text)


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

    # 5. [V2 추가] 단어 사이 공백 2칸 이상 → 1칸
    #    예 : "바랍니다.  단," → "바랍니다. 단,"
    #    표 줄("|"로 시작)은 빈 칸 "|  |" 모양을 지키기 위해 건너뜀
    lines = text.split("\n")
    lines = [line if line.startswith("|") else re.sub(r" {2,}", " ", line)
             for line in lines]
    text = "\n".join(lines)

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