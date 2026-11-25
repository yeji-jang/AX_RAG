"""
<데이터 전처리, 청킹, 임베딩 로그를 남기는 함수>

관리 항목
- timestamp: 실행 시각
- run_id: 실험 버전 (예: "v1", "v2") → 이 값으로 전처리·청킹·임베딩·평가가 한 세트로 묶임
- step: 단계 ("preprocess" | "chunk" | "embed" | "eval")
- method: 어떻게 했는지 한 줄 설명 (예: "워터마크·제어문자 제거, 목차 제외")
- params: 설정값 딕셔너리 (chunk_size, model 등)
- metrics: 결과 수치 딕셔너리 (페이지 수, 청크 수, 소요 시간, 점수 등)
"""

import json
from datetime import datetime
from pathlib import Path


