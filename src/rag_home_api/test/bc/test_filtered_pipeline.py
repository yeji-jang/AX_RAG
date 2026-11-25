"""실제 LangChain 검색·프롬프트 연결, 외부 시스템만 테스트 대역 사용."""
import unittest
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda
from langchain_community.cross_encoders.base import BaseCrossEncoder
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever

import lesson_pipeline as pipeline
import run_filtered_dev as runner
from metadata_filters import MANUAL_DEV_FILTERS, matches_filter, validate_filter


class FakeEncoder(BaseCrossEncoder):
    def score(self, text_pairs):
        return [float(len(text)) for _, text in text_pairs]


def document(key, dong='신림동', deposit=500):
    return Document(id=key, page_content=f'과거 거래 월세 {dong} {key}', metadata={
        'chunk_id': key, 'source_row': 2, 'region_dong': dong, 'lease_type': '월세',
        'deposit_manwon': deposit, 'monthly_rent_manwon': 45, 'area_pyeong': 5,
    })


class FakeStore:
    def __init__(self, docs):
        self.docs = docs
        self.searches = []

    def as_retriever(self, search_kwargs):
        self.searches.append(search_kwargs)
        return RunnableLambda(lambda question: [
            doc for doc in self.docs if matches_filter(doc.metadata, search_kwargs.get('filter'))
        ][:search_kwargs['k']])


class FilteredPipelineTests(unittest.TestCase):
    def setUp(self):
        self.docs = [document('allowed'), document('wrong_dong', '봉천동'),
                     document('expensive', deposit=501), document('another')]
        self.search_filter = MANUAL_DEV_FILTERS['D01']
        self.store = FakeStore(self.docs)
        self.patches = [patch.object(pipeline, 'documents', return_value=self.docs),
                        patch.object(pipeline, 'vector_store', return_value=self.store),
                        patch.object(pipeline, 'cross_encoder_model', return_value=FakeEncoder())]
        self.mocks = [p.start() for p in self.patches]
        for p in self.patches:
            self.addCleanup(p.stop)

    def test_boundaries_zero_missing_unrounded_area(self):
        self.assertTrue(matches_filter({'monthly_rent_manwon': 0}, {'monthly_rent_manwon': 0}))
        for value in (None, True, '0', float('nan'), float('inf')):
            self.assertFalse(matches_filter({'floor': value}, {'floor': {'$gte': 0}}))
        self.assertFalse(matches_filter({}, {'floor': 0}))
        self.assertTrue(matches_filter({'area_pyeong': 5}, {'area_pyeong': {'$gte': 5}}))
        self.assertFalse(matches_filter({'area_pyeong': 4.9999}, {'area_pyeong': {'$gte': 5}}))
        self.assertTrue(matches_filter({}, None))
        self.assertTrue(matches_filter({}, {}))

    def test_invalid_filters_raise(self):
        bad = [[], False, {'id': 'x'}, {'source_row': 2}, {'floor': {'$ne': 2}},
               {'floor': {}}, {'floor': {'$gte': 0, '$lte': 5}},
               {'region_dong': ''}, {'region_dong': ' '}, {'region_dong': {'$eq': '신림동'}}]
        bad += [{'floor': value} for value in (True, '5', None, float('nan'), float('inf'))]
        bad += [{'floor': {'$gte': value}} for value in (False, '5', float('-inf'))]
        for search_filter in bad:
            with self.subTest(search_filter=search_filter), self.assertRaises(ValueError):
                validate_filter(search_filter)
        with self.assertRaises(ValueError):
            pipeline.build_retrievers(search_filter={'floor': True})
        self.mocks[1].assert_not_called()

    def test_numeric_comparisons(self):
        for op, expected in {'$eq': [False, True, False], '$lt': [True, False, False],
                             '$lte': [True, True, False], '$gt': [False, False, True],
                             '$gte': [False, True, True]}.items():
            self.assertEqual([matches_filter({'floor': value}, {'floor': {op: 5}})
                              for value in (4, 5, 6)], expected)

    def test_identical_filter_on_vectors_and_bm25(self):
        retrievers = pipeline.build_retrievers(search_filter=self.search_filter)
        self.assertEqual(self.store.searches, [{'k': 5, 'filter': self.search_filter},
                                              {'k': 20, 'filter': self.search_filter}])
        self.assertIsInstance(retrievers['B'], EnsembleRetriever)
        self.assertIsInstance(retrievers['C'], ContextualCompressionRetriever)
        self.assertEqual({doc.id for doc in retrievers['B'].retrievers[0].docs}, {'allowed', 'another'})
        for retriever in retrievers.values():
            self.assertEqual({doc.id for doc in retriever.invoke('월세')}, {'allowed', 'another'})

    def test_unfiltered_scope_unchanged(self):
        retrievers = pipeline.build_retrievers()
        self.assertEqual(self.store.searches, [{'k': 5}, {'k': 20}])
        self.assertEqual(len(retrievers['B'].retrievers[0].docs), len(self.docs))
        self.assertEqual(len(retrievers['B'].invoke('월세')), len(self.docs))

    def test_empty_candidates_skip_external_components(self):
        with patch.object(pipeline, 'BM25Retriever') as bm25, patch.object(pipeline, 'llm') as llm:
            retrievers = pipeline.build_retrievers(search_filter={'region_dong': '없는동'})
            for retriever in retrievers.values():
                self.assertEqual(retriever.invoke('월세'), [])
                chain, used = pipeline.build_chain(retriever)
                self.assertIn('이번 검색 결과', chain.invoke('월세'))
                self.assertEqual(used, [])
            for mock in (self.mocks[1], self.mocks[2], bm25, llm):
                mock.assert_not_called()

    def test_only_filtered_context_reaches_llm(self):
        seen = []
        model = RunnableLambda(lambda prompt: seen.append(prompt.to_string()) or '과거 참고 사례 [allowed]')
        with patch.object(pipeline, 'llm', return_value=model):
            for retriever in pipeline.build_retrievers(search_filter=self.search_filter).values():
                chain, used = pipeline.build_chain(retriever)
                self.assertIn('[allowed]', chain.invoke('월세 질문'))
                self.assertTrue(all(matches_filter(doc.metadata, self.search_filter) for doc in used))
        self.assertEqual(len(seen), 3)
        for prompt in seen:
            self.assertIn('월세 질문', prompt)
            self.assertIn('현재 매물', prompt)
            self.assertIn('allowed', prompt)
            self.assertNotIn('wrong_dong', prompt)
            self.assertNotIn('expensive', prompt)

    def test_context_cleared_between_questions_and_top_k(self):
        docs = [document(f'doc{i}') for i in range(8)]
        retriever = RunnableLambda(lambda q: docs if q == 'first' else [])
        with patch.object(pipeline, 'llm', return_value=FakeListChatModel(responses=['응답'])) as llm:
            chain, used = pipeline.build_chain(retriever)
            self.assertEqual(chain.invoke('first'), '응답')
            self.assertEqual(used, docs[:pipeline.K])
            self.assertIn('이번 검색 결과', chain.invoke('second'))
            self.assertEqual(used, [])
            llm.assert_called_once()


class FilteredRunnerTests(unittest.TestCase):
    def test_check_uses_only_standard_library(self):
        result = subprocess.run([sys.executable, '-S', str(runner.ROOT / 'run_filtered_dev.py')],
                                capture_output=True, encoding='utf-8',
                                env={**__import__('os').environ, 'PYTHONIOENCODING': 'utf-8'})
        self.assertEqual(result.returncode, 0, result.stderr)
        for expected in ('D01: 1,713', 'D02: 30', 'D03: 23', 'D04: 50'):
            self.assertIn(expected, result.stdout)

    def test_hash_mismatch_rejected_and_inputs_projected(self):
        with patch.object(runner, 'csv_hashes', return_value={'raw': 'bad', 'lf': 'bad'}):
            with self.assertRaises(ValueError):
                runner.read_inputs()
        for question in runner.read_inputs():
            self.assertEqual(set(question), {'id', 'question'})

    def test_existing_results_are_never_overwritten(self):
        for name in ('config.json', 'results.json'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp)
                (output / name).write_text('existing', encoding='utf-8')
                with self.assertRaises(FileExistsError):
                    runner.run('retrieve', output, [])
                self.assertEqual((output / name).read_text(encoding='utf-8'), 'existing')
                self.assertEqual(len(list(output.iterdir())), 1)

    def test_failed_setup_records_errors_and_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(pipeline, 'build_retrievers', side_effect=RuntimeError('offline')):
            self.assertEqual(runner.run('retrieve', Path(tmp), [{'id': 'D01', 'question': '질문'}]), 1)
            results = json.loads((Path(tmp) / 'results.json').read_text(encoding='utf-8'))
            self.assertEqual(len(results), 3)
            self.assertTrue(all(r['status'] == 'error' and r['all_match'] is None for r in results))

    def test_success_records_only_top_k_context(self):
        docs = [document(f'doc{i}') for i in range(8)]
        retrievers = {name: RunnableLambda(lambda q: docs) for name in ('A', 'B', 'C')}
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(pipeline, 'build_retrievers', return_value=retrievers):
            self.assertEqual(runner.run('retrieve', Path(tmp), [{'id': 'D01', 'question': '질문'}]), 0)
            results = json.loads((Path(tmp) / 'results.json').read_text(encoding='utf-8'))
            self.assertTrue(all(r['returned_count'] == 5 and r['all_match'] for r in results))
            config = json.loads((Path(tmp) / 'config.json').read_text(encoding='utf-8'))
            self.assertEqual(config['csv_hashes'], runner.csv_hashes())
            self.assertIn('lesson_pipeline.py', config['code_sha256'])


if __name__ == '__main__':
    unittest.main()
