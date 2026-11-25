"""
<JSONL 검증 도구>
1. dictionary key 구조 일관성 검사
    - 파일 안의 모든 줄이 똑같은 키 이름 묶음을 가지고 있는지 확인
    - 키가 빠지거나 다른 줄이 있으면 몇 건인지 알려줌
2. 빈 값(결측치) 여부 검사
    - 각 키마다 값이 비어있는("" 또는 None, []) 줄이 몇 건인지 집계
"""

import json
from pathlib import Path
from collections import Counter


def validate_jsonl(jsonl_path : Path) -> None : 

    #===================================================================
    # 1. 파일이 실제로 있는지 확인
    #===================================================================
    if not jsonl_path.exists() :
        print(f"{jsonl_path} 파일을 찾을 수 없습니다.")
        return

    key_sets = Counter()        # 키 이름 묶음별로 몇 줄인지 확인
    broken_lines = []           # 깨진 줄 번호를 모아둘 리스트 
    empty_counts = Counter()    # 키별로 빈값("", None, [])이 몇번 나왔는지 셈
    all_keys : set[str] = set() # 파일 전체에서 등장한 모든 키 이름의 모음

    # 검증을 진행한 토탈 행의 수
    total_row = 0

    with jsonl_path.open("r", encoding="utf-8") as f : 
        for line_no, line in enumerate(f, start=1) :
            line = line.strip()

            if not line :
                continue

            try : 
                row = json.loads(line) 
            except json.JSONDecodeError as err :
                broken_lines.append(line_no)
                print(f"[{line_no}번째 줄] JSON 파싱 실패 : {err}")
                continue
            
            total_row += 1

            # 1) 현재 줄의 key 이름을 알파벳순으로 정렬하고 tuple 묶음으로 묶어 카운트
            keys_tuple = tuple(sorted(row.keys())) 
            key_sets[keys_tuple] += 1

            # 2) 현재 줄의 각 key를 돌며 빈값("", None, [])인지를 확인
            for key, value in row.items() :
                all_keys.add(key)
                if value in ("", None) or value == [] : 
                    empty_counts[key] += 1

    #===================================================================
    # 2. 결과 출력
    #===================================================================
    # 1) 무엇을 몇건을 검증했는지
    print(f"{'=' * 20} 검증 파일명 : {jsonl_path.name} {'=' * 20}")

    print(f"\n 1.검증한 총 행의 수 : {total_row}건")

    # 2) 결과값이 1이면 모든 행의 key가 동일
    print(f"\n 2. 키 구성 수 : {len(key_sets)}종류") 
    for keys, count in key_sets.items() :
        print(f" - {count:,}건 : \n {list(keys)}")

    # 3) key 값이 빈값이거나 None, []인 경우 
    print("\n 3. key별 빈 값('', None, [])의 건수")
    for key in sorted(all_keys) : 
        print(f" - {key} : {empty_counts[key]:,}건")

    # 4) JSON 파싱 실패
    print(f"\n 4. JSON 파싱 실패 : {len(broken_lines)}건 \n {broken_lines[:10]}") 


if __name__ == "__main__":
    target = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "01_preprocessed"
        / "공공임대_공고문_V2.jsonl"
    )
    validate_jsonl(target)



"""
실행 시 나오는 결과 예시

==================== 검증 파일명 : 공공임대_공고문_V2.jsonl ====================

 1.검증한 총 행의 수 : 2482건

 2. 키 구성 수 : 1종류
 - 2,482건 : 
 ['category', 'doc_id', 'house_type', 'id', 'notice_date', 'page', 'same_text_ids', 'source', 'source_url', 'text', 'title']

 3. key별 빈 값('', None, [])의 건수
 - category : 0건
 - doc_id : 0건
 - house_type : 0건
 - id : 0건
 - notice_date : 0건
 - page : 0건
 - same_text_ids : 2,353건
 - source : 0건
 - source_url : 0건
 - text : 0건
 - title : 0건

 4. JSON 파싱 실패 : 0건 
 []

==================== 검증 파일명 : 공공임대_공고문_V1.jsonl ====================

 1.검증한 총 행의 수 : 2543건

 2. 키 구성 수 : 1종류
 - 2,543건 : 
 ['category', 'doc_id', 'has_table', 'house_type', 'id', 'notice_date', 'page', 'same_text_ids', 'source', 'source_url', 'spacing_restored', 'text', 'title']

 3. key별 빈 값('', None, [])의 건수
 - category : 0건
 - doc_id : 0건
 - has_table : 0건
 - house_type : 0건
 - id : 0건
 - notice_date : 0건
 - page : 0건
 - same_text_ids : 2,372건 #  2543건 중 2,372건이 []
 - source : 0건
 - source_url : 0건
 - spacing_restored : 0건
 - text : 0건
 - title : 0건

 4. JSON 파싱 실패 : 0건 
 []
"""