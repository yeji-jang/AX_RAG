# MVP 출처와 데이터 필드

## 전월세: 원본 1종

[서울시 부동산 전월세가 정보 OA-21276](https://data.seoul.go.kr/dataList/OA-21276/S/1/datasetView.do?tab=A)의 FILE 탭에서 받은 `서울특별시_전월세가_2025.zip`입니다. 원본 ZIP은 `data/raw/rent/seoul_2025.zip`에 보관했습니다.

파일명의 2025는 접수년도 기준입니다. 계약일과 동일한 의미가 아니므로 MVP에서는 계약일이 2025-01-01~2025-12-31인 행만 사용했습니다. 자료의 최신성을 2026년 실시간 매물로 표현하지 않습니다.

정제 필드는 자치구·법정동·건물명·건물용도·전월세구분·임대면적·계약일·보증금·월세·층·건축년도·원본 ZIP 경로·CSV 레코드 번호입니다. 금액은 원, 면적은 ㎡입니다. 원문의 임대면적을 전용면적으로 임의 변경하지 않았습니다.

## 공공임대: 대표 공고 3건

- [SH 2026년 1차 청년 매입임대](https://www.i-sh.co.kr/main/lay2/program/S1T294C295/www/brd/m_241/view.do?seq=306214): 공고문 PDF와 주택목록 PDF. 개방형·분리형 원룸 구조 및 순위·계층별 보증금/월세를 확인합니다.
- [LH 2026년 3차 서울 청년 매입임대](https://apply.lh.or.kr/lhapply/apply/wt/wrtanc/selectWrtancInfo.do?aisTpCd=26&ccrCnntSysDsCd=03&mi=1026&panId=2015122300020759&uppAisTpCd=13): 공고문 PDF와 공급주택목록 XLSX. 방수·주택유형 및 기본/최대보증금 전환 조건을 확인합니다.
- [2026년 2차 청년안심주택 공공임대](https://soco.seoul.go.kr/youth/bbs/BMSR00015/view.do?boardId=6624&menuNo=40): 공고문 PDF. 단지 공급 조건·입주 예정 시점 등을 확인합니다. 이 공고의 모든 표를 별도 호실 행으로 변환한 것은 아닙니다.

`public_housing_units.csv`의 한 행은 공고별 호실 공급 기록입니다. `unit_record_id`로 `public_rent_scenarios.csv`와 결합합니다. 동일한 집의 다른 순위·보증금 조건을 서로 다른 매물로 세지 않습니다. `price_status=not_offered_in_source`는 무료가 아니라 원문에 제공되지 않은 조건입니다.

PDF 근거는 실제 PDF 페이지, XLSX 근거는 시트명·행 번호로 추적합니다. 방 1개와 개방형 원룸은 같은 개념으로 단정하지 않습니다.

## 거리: 기존 예시 재사용

[청년안심주택 단지 정보](https://soco.seoul.go.kr/youth/pgm/home/yohome/list.do?menuNo=400002)의 공개 단지 좌표와 [서울시 지하철역 좌표](https://t-data.seoul.go.kr/dataprovide/trafficdataviewfile.do?data_id=36), [서울시청 공식 지도](https://www.seoul.go.kr/seoul/map.do)를 사용했습니다. Haversine 공식으로 구한 직선거리이며 역 출입구까지의 도보거리가 아닙니다.

단지 목록은 공공·공공지원민간임대가 혼재합니다. 단지 좌표만으로 특정 공공임대 호실과 자동 결합하지 않았습니다. 실제 통근시간은 미수집입니다.

## 수집 범위의 한계

서울시 연간 공개 파일의 누락·정정 가능성, 공고의 변경·접수 종료, 옵션의 누락이 남아 있습니다. 이 MVP는 수집한 자료의 확인을 돕고, 현재 매물 존재나 신청 자격을 보장하지 않습니다.
