"""검색 조건을 검증하고 PGVector와 로컬 BM25에 동일하게 적용한다."""
import math
import operator
from numbers import Real

STRING_FIELDS = {'region_gu', 'region_dong', 'building_name', 'housing_type', 'lease_type'}
NUMBER_FIELDS = {'deposit_manwon', 'monthly_rent_manwon', 'area_m2',
                 'area_pyeong', 'floor', 'build_year'}
OPERATORS = {'$eq': operator.eq, '$lt': operator.lt, '$lte': operator.le,
             '$gt': operator.gt, '$gte': operator.ge}

def finite_number(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def validate_filter(search_filter):
    if search_filter is None:
        return
    if not isinstance(search_filter, dict):
        raise ValueError('필터는 딕셔너리여야 합니다.')
    for field, condition in search_filter.items():
        if field in STRING_FIELDS:
            if not isinstance(condition, str) or not condition.strip():
                raise ValueError(f'{field}: 비어 있지 않은 문자열이 필요합니다.')
        elif field in NUMBER_FIELDS:
            if isinstance(condition, dict):
                if len(condition) != 1 or next(iter(condition)) not in OPERATORS:
                    raise ValueError(f'{field}: 지원하는 숫자 연산자 하나만 허용합니다.')
                condition = next(iter(condition.values()))
            if not finite_number(condition):
                raise ValueError(f'{field}: 유한한 숫자가 필요합니다.')
        else:
            raise ValueError(f'지원하지 않는 필드: {field}')


def matches_filter(metadata, search_filter):
    validate_filter(search_filter)
    for field, condition in (search_filter or {}).items():
        value = metadata.get(field)
        if field in STRING_FIELDS:
            if not isinstance(value, str) or value != condition:
                return False
        else:
            if not finite_number(value):
                return False
            op, limit = next(iter(condition.items())) if isinstance(condition, dict) else ('$eq', condition)
            if not OPERATORS[op](value, limit):
                return False
    return True
