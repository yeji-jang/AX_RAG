# jh 공유 데이터 (RAG용 텍스트)

서울 공공임대·청년안심주택 모집공고문과 일반 전월세 법령·가이드를 **JSONL(한 줄 = 한 쪽 또는 한 조문)**로 바꾼 자료입니다. 기준일 2026-11-23, 최근 1년 공고.

| 파일 | 문서 | 줄 | 내용 |
|---|---|---|---|
| `공공임대_공고문.jsonl` | 100 | 2,819 | LH(임대주택·주거복지)·SH 모집공고문, 한 줄 = 한 쪽 |
| `청년안심주택_공고문.jsonl` | 313 | 2,174 | 청년안심주택 공공·민간임대 모집공고문, 한 줄 = 한 쪽 |
| `일반전월세_법령가이드.jsonl` | 8 | 430 | 주택임대차보호법·시행령·부동산거래신고법(한 줄 = 한 조문, 본칙만) + 표준계약서·해설집·서울시 가이드북·상담사례집(한 줄 = 한 쪽) |
| `문서목록.csv` | 421 | | 문서별 출처 URL·공고일·변환된 쪽 수·빈 쪽 수·비고 (엑셀용) |

## 한 줄의 모양

```json
{"id": "LH_2015122300021098_p003", "doc_id": "LH_2015122300021098", "source": "LH",
 "category": "공공임대 모집공고", "title": "…행복주택 예비입주자 모집공고", "house_type": "행복주택",
 "notice_date": "20261119", "source_url": "https://apply.lh.or.kr/…", "page": 3,
 "has_table": true, "text": "…본문…\n\n| 단지 | 공급형별 | … |\n|---|---|…"}
```

법령은 `page` 대신 `section`(본칙)·`article`(예: 제3조)이 들어갑니다.

## 알아 둘 점

- **표는 마크다운 표**로 들어 있습니다. 칸이 합쳐진 복잡한 표는 일부 칸이 비거나 밀릴 수 있습니다.
- 그림·도면만 있는 쪽(165쪽)은 뺐고, 문서목록의 `pages_without_text`에 수를 적었습니다.
- 일부 공고문은 **원본 PDF부터 띄어쓰기가 빠져** 있습니다(예: `청년안심주택은서울시와…`).
- 청년안심주택 민간임대 공고 310건은 앞부분 "필수 확인사항" 문단이 거의 같습니다. 색인할 때 중복 정리가 필요합니다.
- 당첨자 명단 등 개인정보 첨부는 처음부터 받지 않았습니다. 주민번호·전화번호처럼 보이는 값은 서식 예시·입주지원센터 대표번호입니다.
- 쪽 = 청크가 아닙니다. 청킹은 색인 단계에서 다시 합니다.

## 다시 만들기

```powershell
uv run python -m rag_home_api.collect.notices        # LH·SH 공고문 PDF → data/raw/notices
uv run python -m rag_home_api.collect.youth_housing --download   # 청년안심주택 → data/raw/youth
uv run python -m rag_home_api.collect.rental_docs    # 법령·가이드 → data/raw/rental
uv run python -m rag_home_api.collect.export_jsonl   # PDF → 이 폴더의 JSONL
```

`.env`에 `DATA_GO_KR_SERVICE_KEY`(공공데이터포털 인증키)가 필요합니다. 원본 PDF(약 300MB)는 저장소에 올리지 않았습니다.
