"""정확 일치 응답 캐시. HTTP API와 무관하게 팀 서비스에서 호출한다."""
import hashlib
import json
import math
import threading
import time
from collections import OrderedDict
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

if __package__:
    from .query_filters import validate_question
else:
    from query_filters import validate_question

ROOT = Path(__file__).resolve().parent


@lru_cache(maxsize=32)
def _file_hash(path, size, modified_ns):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime_version():
    """코드·CSV·의존성 변경을 키에 반영. DB 갱신은 index_version으로 알린다."""
    if __package__:
        from .structural_chunk_v0 import DEFAULT_CSV_PATH, PROJECT_ROOT
    else:
        from structural_chunk_v0 import DEFAULT_CSV_PATH, PROJECT_ROOT
    files = [ROOT / name for name in ('cached_pipeline.py', 'query_filters.py',
             'metadata_filters.py', 'lesson_pipeline.py', 'structural_chunk_v0.py')]
    files += [DEFAULT_CSV_PATH, PROJECT_ROOT / 'uv.lock']
    result = {}
    for path in files:
        stat = path.stat()
        result[path.name] = _file_hash(str(path), stat.st_size, stat.st_mtime_ns)
    return result


class ResponseCache:
    def __init__(self, ttl_seconds=600, max_entries=100, clock=time.monotonic):
        if (not isinstance(ttl_seconds, (int, float)) or isinstance(ttl_seconds, bool)
                or not math.isfinite(ttl_seconds) or ttl_seconds <= 0):
            raise ValueError('TTL은 양의 유한한 숫자여야 합니다.')
        if not isinstance(max_entries, int) or isinstance(max_entries, bool) or max_entries < 1:
            raise ValueError('최대 항목 수는 양의 정수여야 합니다.')
        self.ttl_seconds, self.max_entries, self.clock = ttl_seconds, max_entries, clock
        self._items, self._lock = OrderedDict(), threading.Lock()

    def get(self, key):
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            expires, value = item
            if self.clock() >= expires:
                del self._items[key]
                return None
            self._items.move_to_end(key)
            return deepcopy(value)

    def put(self, key, value):
        with self._lock:
            now = self.clock()
            for expired in [key for key, (expiry, _) in self._items.items() if expiry <= now]:
                del self._items[expired]
            self._items[key] = (now + self.ttl_seconds, deepcopy(value))
            self._items.move_to_end(key)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)

    def clear(self):
        with self._lock:
            self._items.clear()


class CachedPipeline:
    """프로세스 안에서 인스턴스를 재사용하고 요청별 답변 체인은 공유하지 않는다."""
    def __init__(self, cache=None, extractor=None, executor=None):
        self.cache = cache if cache is not None else ResponseCache()
        self._extractor, self._executor = extractor, executor

    def answer(self, question, *, scope, index_version, system='C', use_cache=True):
        validate_question(question)
        if system not in ('A', 'B', 'C'):
            raise ValueError('검색 구성은 A/B/C 중 하나여야 합니다.')
        if any(not isinstance(v, str) or not v.strip() for v in (scope, index_version)):
            raise ValueError('사용자/세션 범위와 데이터/색인 버전이 필요합니다.')
        question = question.strip()
        started = time.perf_counter()
        settings = {'question': question, 'scope': scope, 'index_version': index_version,
                    'system': system, 'runtime': runtime_version()}
        key = hashlib.sha256(json.dumps(settings, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        cached = self.cache.get(key) if use_cache else None
        if cached is not None:
            return {**cached, 'cache_hit': True, 'cache_key': key,
                    'index_version': index_version, 'pipeline_executed': False,
                    'elapsed_seconds': time.perf_counter() - started}
        if __package__:
            from .query_filters import build_filter_extractor, execute_query, resolve_building_plan
        else:
            from query_filters import build_filter_extractor, execute_query, resolve_building_plan
        extractor = self._extractor if self._extractor is not None else build_filter_extractor()
        executor = self._executor if self._executor is not None else execute_query
        plan = resolve_building_plan(extractor.invoke(question))
        result = executor(question, plan, stage='answer', system=system)
        # Document를 JSON 형태로 바꾸고 답변과 실제 출처를 함께 저장한다.
        sources = [{'id': doc.metadata.get('chunk_id') or doc.id, 'storage_id': doc.id,
                    'page_content': doc.page_content,
                    'metadata': deepcopy(doc.metadata)} for doc in result['documents']]
        value = {'status': result['status'], 'response': result['response'],
                 'sources': sources, 'plan': plan.as_dict()}
        if (use_cache and result['status'] == 'success' and sources and result['response']):
            self.cache.put(key, value)
        return {**value, 'cache_hit': False, 'cache_key': key,
                'index_version': index_version, 'pipeline_executed': True,
                'elapsed_seconds': time.perf_counter() - started}
