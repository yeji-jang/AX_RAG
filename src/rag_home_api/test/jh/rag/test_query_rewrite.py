"""공유 리라이트 규칙과 기존 CLI 함수의 오프라인 검증. 실제 API/DB 호출 없음."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from query_rewrite import rewrite_query


class QueryRewriteTests(unittest.TestCase):
    def test_original_anchor_preserves_conditions_and_literals(self):
        question = '서울에서 2026년 5월 1일까지 보증금 1,000만원을 못 받으면?'
        chain = RunnableLambda(lambda _: SimpleNamespace(content='서울 2026년 5월 1일까지 보증금 1,000만원 미반환 조건'))
        plan = rewrite_query(question, chain)
        self.assertIn(question, plan['query'])
        self.assertIsNone(plan['fallback'])

    def test_missing_added_changed_numbers_or_article_use_original(self):
        original = '서울 2026년 5월 보증금 100만원 조건'
        for rewritten in ['', '서울 보증금 조건', original + ' 14일', original + ' 제99조',
                          original.replace('100만원', '200만원')]:
            with self.subTest(rewritten=rewritten):
                result = rewrite_query(original, RunnableLambda(lambda _, text=rewritten: SimpleNamespace(content=text)))
                self.assertEqual(result['query'], original)
                self.assertIsNotNone(result['fallback'])

    def test_failure_uses_original(self):
        def broken(_): raise TimeoutError()
        result = rewrite_query('원래 질문', RunnableLambda(broken))
        self.assertEqual(result['query'], '원래 질문')
        self.assertEqual(result['fallback'], 'TimeoutError')

    def test_existing_prepare_entry_only_sends_question_and_preserves_log_contract(self):
        import run_rewrite_c as entry
        observed = []; rows = []
        original = '서울 2026년 보증금 100만원 반환 조건은?'
        def model(prompt):
            observed.append(prompt.to_string())
            return SimpleNamespace(content='서울 2026년 보증금 100만원 반환 조건 검색')
        question = {'question_id': 'SYNTHETIC-01', 'question': original,
                    'key_facts': ['LOCAL_ONLY_GOLD_MARKER'], 'allowed_chunk_ids': ['LOCAL_ONLY_ID_MARKER']}
        with patch.object(entry, 'init_chat_model', return_value=RunnableLambda(model)), \
                patch.object(entry, 'eval_sha', return_value='synthetic-local'), patch.object(entry, 'append', side_effect=rows.append), \
                patch('builtins.print'):
            entry.prepare([question], 'synthetic-offline', None)
        self.assertEqual(len(observed), 1)
        self.assertIn(original, observed[0])
        self.assertNotIn('LOCAL_ONLY_', observed[0])
        self.assertEqual(rows[0]['record_type'], 'rewrite_plan')
        self.assertIn(original, rows[0]['query_used'])
        self.assertTrue(rows[0]['original_anchor_preserved'])
        self.assertIsNone(rows[0]['fallback'])

    def test_existing_generate_entry_uses_original_and_verified_context_only(self):
        import run_rewrite_c as entry
        original = '가상 법령 질문 원문'; requests = []
        doc = Document(page_content='공개 법령의 합성 발췌', metadata={'chunk_id': 'SYN_CID', 'row_id': 'SYN_ROW'})
        def create(**kwargs):
            requests.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
                'status': 'answered', 'answer': '모의 응답', 'citations': ['S1']})))],
                usage=SimpleNamespace(model_dump=lambda: {}))
        api = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        response = entry.generate(api, original, [doc], {'SYN_CID': doc}, {'SYN_ROW': 'https://www.law.go.kr/example'})
        self.assertEqual(response['status'], 'answered')
        self.assertEqual(requests[0]['messages'][1]['content'], original + '\n\n[참고 발췌]\n[S1] ' + doc.page_content)
        self.assertEqual(response['invalid_citations'], [])
        with self.assertRaises(ValueError):
            entry.generate(api, original, [doc], {}, {})
        self.assertEqual(len(requests), 1)

    def test_existing_c_reranks_full_b_candidates_before_top_five(self):
        import run_c
        docs = [Document(page_content='합성 법령 본문', metadata={'chunk_id': f'SYN_{i}', 'row_id': f'ROW_{i}'})
                for i in range(6)]
        seen = []
        class Reranker:
            def compress_documents(self, candidates, query):
                seen.append((list(candidates), query))
                return [candidates[-1], *candidates[:4]]
        question = {'question_id': 'SYN-01', 'question': '가상 임대차 질문',
                    'allowed_chunk_ids': ['SYN_5'], 'expected_status': 'answered'}
        result = run_c.run_pair(RunnableLambda(lambda _: docs), Reranker(), question)
        self.assertEqual(seen, [(docs, question['question'])])
        self.assertEqual(result['outcome'], 'C_rescue')
        self.assertEqual(len(result['C_top_ids']), 5)


if __name__ == '__main__': unittest.main()
