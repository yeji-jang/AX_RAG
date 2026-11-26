"""자연어 조건 JSON 검증과 기존 검색 체인 연결을 외부 호출 없이 확인한다."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

import lesson_pipeline as pipeline
import query_filters as query
import run_query_dev as runner
from metadata_filters import MANUAL_DEV_FILTERS
from test_filtered_pipeline import document


def payload(conditions=None, unsupported=None, ambiguous=None):
    return json.dumps({'conditions': conditions or [], 'unsupported_conditions': unsupported or [],
                       'ambiguous_conditions': ambiguous or []}, ensure_ascii=False)


def condition(field, value, source, op='$eq'):
    return {'field': field, 'operator': op, 'value': value, 'source_text': source}


class QueryFilterTests(unittest.TestCase):
    def test_only_existing_dev_and_test_can_be_selected(self):
        self.assertEqual(len(runner.read_questions(split='dev')), 10)
        self.assertEqual(len(runner.read_questions(split='test')), 20)
        with self.assertRaises(ValueError):
            runner.question_path('v3')

    def test_markdown_and_existing_json_questions_are_identical(self):
        import re
        content = (runner.ROOT / 'docs/abc_evaluation_questions.md').read_text(encoding='utf-8')
        markdown = {}
        for line in content.splitlines():
            cells = [cell.strip() for cell in line.split('|')]
            if len(cells) >= 6 and re.fullmatch(r'[DT]\d{2}', cells[1]):
                markdown[cells[1]] = cells[3]
        actual = runner.read_questions(split='dev') + runner.read_questions(split='test')
        self.assertEqual(len(markdown), 30)
        self.assertEqual(markdown, {q['id']: q['question'] for q in actual})

    def test_real_prompt_json_model_parser_connection(self):
        question = '신림동 보증금 500만원 이하 월세 45만원 이하 5평 이상'
        conditions = [condition('region_dong', '신림동', '신림동'),
                      condition('lease_type', '월세', '월세'),
                      condition('deposit_manwon', 500, '보증금 500만원 이하', '$lte'),
                      condition('monthly_rent_manwon', 45, '월세 45만원 이하', '$lte'),
                      condition('area_pyeong', 5, '5평 이상', '$gte')]
        model = FakeListChatModel(responses=[payload(conditions)])
        original_bind = model.bind
        with patch.object(FakeListChatModel, 'bind', autospec=True,
                          side_effect=lambda instance, **kwargs: original_bind(**kwargs)) as bind:
            extractor = query.build_filter_extractor(model)
            plan = extractor.invoke(question)
            bind.assert_called_once_with(model, response_format={'type': 'json_object'})
        self.assertEqual(plan.search_filter, MANUAL_DEV_FILTERS['D01'])
        self.assertEqual(plan.status, 'ready')

    def test_units_and_synonyms_preserve_validated_model_output(self):
        question = '남현동 빌라 전세 2억 5천만원 이하 8평 이상'
        plan = query.parse_filter_plan(payload([
            condition('region_dong', '남현동', '남현동'),
            condition('housing_type', '연립다세대', '빌라'),
            condition('lease_type', '전세', '전세'),
            condition('deposit_manwon', 25000, '2억 5천만원 이하', '$lte'),
            condition('area_pyeong', 8, '8평 이상', '$gte'),
        ]), question)
        self.assertEqual(plan.search_filter, MANUAL_DEV_FILTERS['D03'])

    def test_invalid_json_and_schema_rejected(self):
        bad = ['not json', '[]', '{}', '{"conditions":[],"conditions":[]}',
               payload().replace('[]', 'null', 1),
               payload([{'field': 'floor'}]),
               payload([condition('id', 'x', '질문')]),
               payload([condition('floor', '5', '질문', '$gte')]),
               payload([condition('floor', True, '질문', '$gte')]),
               payload([condition('floor', float('nan'), '질문', '$gte')]),
               payload([condition('floor', float('inf'), '질문', '$gte')]),
               payload([condition('floor', 5, '질문', '$ne')]),
               payload([condition('region_dong', '신림동', '질문', '$in')])]
        for raw in bad:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                query.parse_filter_plan(raw, '질문')

    def test_duplicate_field_does_not_silently_drop_bound(self):
        with self.assertRaises(ValueError):
            query.parse_filter_plan(payload([condition('floor', 5, '5층 이상', '$gte'),
                                             condition('floor', 10, '10층 이하', '$lte')]),
                                    '5층 이상 10층 이하')

    def test_condition_sources_must_exist_in_question(self):
        for raw in (payload([condition('floor', 5, '존재하지않는조건', '$gte')]),
                    payload(unsupported=['존재하지않는조건']), payload(ambiguous=[''])):
            with self.assertRaises(ValueError):
                query.parse_filter_plan(raw, '질문')

    def test_empty_question_does_not_call_model(self):
        model = FakeListChatModel(responses=[payload()])
        extractor = query.build_filter_extractor(model)
        with patch.object(model, '_call') as call:
            for question in ('', '   ', None):
                with self.assertRaises(ValueError):
                    extractor.invoke(question)
            call.assert_not_called()

    def test_ambiguous_and_unsupported_skip_search_and_generation(self):
        for data, status in ((payload(ambiguous=['5평 후반대']), 'needs_clarification'),
                             (payload(unsupported=['관리비']), 'unsupported')):
            plan = query.parse_filter_plan(data, '5평 후반대 관리비')
            with patch.object(pipeline, 'build_retrievers') as build, patch.object(pipeline, 'llm') as llm:
                result = query.execute_query('5평 후반대 관리비', plan)
                self.assertEqual(result['status'], status)
                self.assertEqual(result['documents'], [])
                self.assertIn('현재 매물', result['response'])
                build.assert_not_called()
                llm.assert_not_called()

    def test_ready_plan_passes_question_and_filter_to_existing_chain(self):
        question = '신림동'
        plan = query.parse_filter_plan(payload([condition('region_dong', '신림동', question)]), question)
        seen = []
        retrievers = {name: RunnableLambda(lambda q: seen.append(q) or [document('allowed')])
                      for name in ('A', 'B', 'C')}
        with patch.object(pipeline, 'build_retrievers', return_value=retrievers) as build, \
                patch.object(pipeline, 'llm', return_value=FakeListChatModel(responses=['과거 사례'])):
            result = query.execute_query(question, plan)
            build.assert_called_once_with('v0', 'small', search_filter={'region_dong': '신림동'})
            self.assertEqual(result['response'], '과거 사례')
            self.assertEqual(seen, [question])

    def test_bad_model_output_propagates_without_unfiltered_retry(self):
        extractor = query.build_filter_extractor(FakeListChatModel(responses=['invalid JSON']))
        with patch.object(pipeline, 'build_retrievers') as build:
            with self.assertRaises(ValueError):
                extractor.invoke('질문')
            build.assert_not_called()

    def test_runner_extraction_failure_remains_error(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(query, 'build_filter_extractor', return_value=RunnableLambda(
                    lambda q: query.parse_filter_plan('bad', q))):
            self.assertEqual(runner.run('extract', Path(tmp), [{'id': 'D01', 'question': '질문'}]), 1)
            records = json.loads((Path(tmp) / 'results.json').read_text(encoding='utf-8'))
            self.assertEqual(records[0]['status'], 'error')
            self.assertIsNone(records[0]['plan'])

    def test_runner_reuses_one_extraction_and_one_retriever_setup(self):
        plan = query.parse_filter_plan(payload(), '질문')
        seen = []
        extractor = RunnableLambda(lambda q: seen.append(q) or plan)
        retrievers = {name: RunnableLambda(lambda q: []) for name in ('A', 'B', 'C')}
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(query, 'build_filter_extractor', return_value=extractor), \
                patch.object(pipeline, 'build_retrievers', return_value=retrievers) as build:
            self.assertEqual(runner.run('retrieve', Path(tmp), [{'id': 'D01', 'question': '질문'}]), 0)
            self.assertEqual(seen, ['질문'])
            build.assert_called_once()

    def test_runner_does_not_overwrite_existing_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'results.json'
            path.write_text('existing', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                runner.run('extract', Path(tmp), [])
            self.assertEqual(path.read_text(encoding='utf-8'), 'existing')

    def test_building_whitespace_resolves_unique_source_name(self):
        plan = query.parse_filter_plan(payload([condition('building_name', '예시 1단지', '예시 1단지')]),
                                       '예시 1단지')
        result = query.resolve_building_plan(plan, [{'building_name': '예시1단지'}])
        self.assertEqual(result.search_filter['building_name'], '예시1단지')
        self.assertEqual(plan.search_filter['building_name'], '예시 1단지')
        self.assertEqual(result.normalizations[0]['rule'], 'unique_normalized_catalog_name')

    def test_unique_numbered_abbreviation_resolves_catalog_name(self):
        plan = query.parse_filter_plan(payload([condition('building_name', '예시 1단지', '예시 1단지')]),
                                       '예시 1단지')
        result = query.resolve_building_plan(plan, [{'building_name': '예시서울대입구1단지'}])
        self.assertEqual(result.status, 'ready')
        self.assertEqual(result.search_filter['building_name'], '예시서울대입구1단지')

    def test_numbered_alias_never_matches_other_unit_number(self):
        plan = query.parse_filter_plan(payload([condition('building_name', '예시 1단지', '예시 1단지')]),
                                       '예시 1단지')
        result = query.resolve_building_plan(plan, [{'building_name': '예시서울대입구11단지'}])
        self.assertEqual(result.status, 'needs_clarification')

    def test_multiple_numbered_alias_candidates_need_clarification(self):
        plan = query.parse_filter_plan(payload([condition('building_name', '예시 2차', '예시 2차')]), '예시 2차')
        result = query.resolve_building_plan(plan, [{'building_name': '북부예시2차'},
                                                  {'building_name': '남부예시2차'}])
        self.assertEqual(result.status, 'needs_clarification')

    def test_building_resolution_filters_housing_type_before_exact_match(self):
        plan = query.parse_filter_plan(payload([
            condition('building_name', '예시타워', '예시타워'),
            condition('housing_type', '오피스텔', '오피스텔'),
        ]), '예시타워 오피스텔')
        result = query.resolve_building_plan(plan, [
            {'building_name': '예시타워', 'housing_type': '연립다세대'},
            {'building_name': '예시 타워 오피스텔', 'housing_type': '오피스텔'},
        ])
        self.assertEqual(result.search_filter['building_name'], '예시 타워 오피스텔')
        self.assertEqual(result.search_filter['housing_type'], '오피스텔')

    def test_building_type_suffix_is_not_part_of_identity(self):
        plan = query.parse_filter_plan(payload([condition('building_name', '예시 아파트', '예시 아파트')]),
                                       '예시 아파트')
        result = query.resolve_building_plan(plan, [{'building_name': '예시'}])
        self.assertEqual(result.search_filter['building_name'], '예시')

    def test_housing_punctuation_normalization(self):
        for value in ('단독·다가구', '단독 다가구', '단독/다가구'):
            plan = query.parse_filter_plan(payload([condition('housing_type', value, value)]), value)
            self.assertEqual(plan.search_filter['housing_type'], '단독다가구')

    def test_supported_information_request_does_not_block_search(self):
        question = '월세와 보증금이 궁금해요'
        plan = query.parse_filter_plan(payload(ambiguous=[question]), question)
        self.assertEqual(plan.status, 'ready')
        missing = query.parse_filter_plan(payload(unsupported=['관리비가 궁금해요']), '관리비가 궁금해요')
        self.assertEqual(missing.status, 'unsupported')

    def test_information_request_with_budget_stays_a_constraint(self):
        question = '보증금 500만원 이하인 집이 궁금해요'
        plan = query.parse_filter_plan(payload([
            condition('deposit_manwon', 500, question, '$lte')
        ]), question)
        self.assertEqual(plan.search_filter['deposit_manwon'], {'$lte': 500})

    def test_zero_deposit_request_is_not_removed_as_information_only(self):
        question = '보증금 없는 집이 궁금해요'
        plan = query.parse_filter_plan(payload([
            condition('deposit_manwon', 0, question)
        ]), question)
        self.assertEqual(plan.search_filter['deposit_manwon'], {'$eq': 0})

    def test_qualitative_budget_request_still_requires_clarification(self):
        question = '월세가 저렴한 집을 알려주세요'
        plan = query.parse_filter_plan(payload(ambiguous=[question]), question)
        self.assertEqual(plan.status, 'needs_clarification')

    def test_exact_m2_takes_precedence_over_consistent_approximate_pyeong(self):
        question = '임대면적 33.05785㎡ 약 10.00평 보증금은 어느 정도인지 궁금해요'
        plan = query.parse_filter_plan(payload([
            condition('area_m2', 33.05785, '임대면적 33.05785㎡'),
            condition('area_pyeong', 10, '약 10.00평'),
        ], ambiguous=['약 10.00평', '보증금은 어느 정도인지 궁금해요']), question)
        self.assertEqual(plan.status, 'ready')
        self.assertEqual(plan.search_filter, {'area_m2': {'$eq': 33.05785}})

    def test_exact_m2_clears_redundant_approximate_ambiguity_without_extra_condition(self):
        question = '33.05785㎡ 약 10평'
        plan = query.parse_filter_plan(payload([
            condition('area_m2', 33.05785, '33.05785㎡'),
        ], ambiguous=['약 10평']), question)
        self.assertEqual(plan.status, 'ready')

    def test_conflicting_area_values_still_need_clarification(self):
        question = '33.05785㎡ 약 20평'
        plan = query.parse_filter_plan(payload([
            condition('area_m2', 33.05785, '33.05785㎡'),
            condition('area_pyeong', 20, '약 20평'),
        ]), question)
        self.assertEqual(plan.status, 'needs_clarification')
        self.assertEqual(plan.search_filter['area_m2'], {'$eq': 33.05785})

    def test_approximate_area_alone_and_vague_range_remain_ambiguous(self):
        plan = query.parse_filter_plan(payload([condition('area_pyeong', 10, '약 10평')]), '약 10평')
        self.assertEqual(plan.status, 'needs_clarification')
        vague = query.parse_filter_plan(payload(ambiguous=['5평 후반대']), '5평 후반대')
        self.assertEqual(vague.status, 'needs_clarification')

    def test_building_resolution_uses_only_requested_region(self):
        plan = query.parse_filter_plan(payload([condition('building_name', '예시 1단지', '예시 1단지'),
                                               condition('region_dong', '신림동', '신림동')]),
                                       '신림동 예시 1단지')
        result = query.resolve_building_plan(plan, [
            {'building_name': '예시1단지', 'region_dong': '봉천동'},
        ])
        self.assertEqual(result.status, 'needs_clarification')


if __name__ == '__main__':
    unittest.main()
