"""서비스 경계 단위 테스트. DEV·TEST 평가 질문 파일을 읽지 않고 외부 연결을 대체함."""

import unittest
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from law_router import build_law_handler, build_question_router, classify_question, law_response
from retrieve import CONFIG, Retriever, apply_preset


def statute():
    return Document(page_content="주거용 건물의 전부 또는 일부의 임대차에 관하여 적용한다.",
                    metadata={"chunk_id": "LAW_SAMPLE_c01", "row_id": "LAW_SAMPLE",
                              "doc_id": "LAW_SAMPLE", "category": "일반 전월세 법령",
                              "title": "주택임대차보호법", "article": "제2조",
                              "source_url": "https://www.law.go.kr/법령/주택임대차보호법"})


class IntentTests(unittest.TestCase):
    def test_boundaries(self):
        cases = {
            "법령": [
                "집주인이 월세를 막 올려도 돼?", "임대인의 차임 인상에 법정 제한이 있습니까?",
                "LH 단지에 살아도 집주인이 월세 올릴 수 있는 한도가 있어?",
                "집주인이 보일러 안 고쳐 주면 수리비를 돌려받을 수 있어?",
                "임대인은 고장난 주택 설비를 고쳐줘야 하나요?",
                "월세 계약 해지해도 돼?", "주택 임대료 상한이 궁금해요",
                "주택의 세입자가 주장할 수 있는 권리가 있나요?",
                "임차권등기 절차의 법적 조건은 무엇인가요?",
                "이사한 원룸의 전입신고 늦게 해도 돼?",
                "전세 계약의 확정일자는 언제 받아야 하나요?",
                "행복주택 공고를 봤는데 주택임대차 의무는 법으로 어떻게 정해요?",
                "월세 연체했다고 집주인이 쫓아내도 돼?",
                "집주인이 허락 없이 집에 들어와도 돼?",
                "빌린 집 일부를 못 쓰게 됐는데 월세 깎을 수 있어?",
                "집주인이 평균보다 월세를 더 올려도 돼?",
                "월세 계약의 갱신 조건이 궁금해요",
            ],
            "비법령": [
                "LH 모집공고 보여줘", "청년안심주택 입주신청 자격 알려줘",
                "우리 단지 근처 병원 있어?", "공고의 신청 기한은 언제야?",
                "이 원룸 월세 얼마야?", "아파트 보증금 얼마인가요?",
                "법원역 근처 월세 매물 있어?", "최근 월세 인상률 평균을 보여줘",
                "법령 말고 임대주택 모집공고만 찾아줘", "조문은 필요없고 공고의 임대조건 표만 볼래",
                "행복주택 평면과 옵션 알려줘", "단지에서 지하철까지 거리 알려줘",
                "최근 평균 월세 인상률은 몇 퍼센트야?",
            ],
            "미지원": ["", "   ", "회사 연차 사용 권리 알려줘", "방학이 언제야?", "민법상 상속 순서는?", "추천해줘"],
        }
        routes = {"법령": "law", "비법령": "non_law", "미지원": "unsupported"}
        for label, questions in cases.items():
            for question in questions:
                with self.subTest(question=question):
                    self.assertEqual(classify_question(question), routes[label])

    def test_non_law_contract_is_preserved(self):
        seen = []
        original = {"existing_payload": ["team handler"], "status": "custom"}
        branch = build_question_router(RunnableLambda(lambda q: self.fail("법령 호출 금지")),
                                       RunnableLambda(lambda q: seen.append(q) or original))
        query = "  모집 공고 보여줘  "
        self.assertIs(branch.invoke(query), original)
        self.assertEqual(seen, [query])

    def test_missing_non_law_handler_is_unsupported(self):
        branch = build_question_router(RunnableLambda(lambda q: "law"))
        self.assertEqual(branch.invoke("공고 알려줘")["status"], "unsupported")
        self.assertEqual(branch.invoke("  ")["status"], "unsupported")
        self.assertEqual(branch.invoke("집주인이 월세 막 올려도 돼?"), "law")


class CitationTests(unittest.TestCase):
    def test_only_cited_statutes_are_published(self):
        first = statute()
        second = Document(page_content="다른 본문", metadata={**first.metadata, "chunk_id": "unused"})
        result = law_response([first, second], {"status": "answered", "answer": "확인된 답변",
                                               "cited_chunk_ids": ["LAW_SAMPLE_c01"]})
        self.assertEqual(result["status"], "answered")
        self.assertEqual(len(result["citations"]), 1)
        self.assertEqual(result["citations"][0]["law_name"], "주택임대차보호법")
        self.assertEqual(result["citations"][0]["article"], "제2조")
        self.assertEqual(result["citations"][0]["source_url"], first.metadata["source_url"])

    def test_insufficient_clears_fabricated_article(self):
        result = law_response([statute()], {"status": "insufficient_evidence", "answer": "임의 조문",
                                            "cited_chunk_ids": ["invented"]})
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["citations"], [])
        self.assertNotIn("임의", result["answer"])

    def test_invalid_citations_fail_closed(self):
        for change in ({"chunk_id": "wrong"}, {"category": "일반 전월세 가이드"},
                       {"article": None}, {"source_url": "https://example.com/"}):
            with self.subTest(change=change):
                doc = statute()
                doc.metadata.update(change)
                result = law_response([doc], {"status": "answered", "answer": "답변",
                                              "cited_chunk_ids": ["LAW_SAMPLE_c01"]})
                self.assertEqual(result["status"], "error")
                self.assertEqual(result["citations"], [])

    def test_empty_and_non_law_documents_never_reach_generator(self):
        generator = RunnableLambda(lambda x: self.fail("생성 호출 금지"))
        empty = build_law_handler(RunnableLambda(lambda q: []), generator)
        self.assertEqual(empty.invoke("질문")["status"], "insufficient_evidence")
        guide = statute()
        guide.metadata["category"] = "일반 전월세 가이드"
        handler = build_law_handler(RunnableLambda(lambda q: [guide]), generator)
        self.assertEqual(handler.invoke("질문")["status"], "error")
        incomplete = statute()
        incomplete.metadata.pop("source_url")
        handler = build_law_handler(RunnableLambda(lambda q: [incomplete]), generator)
        self.assertEqual(handler.invoke("질문")["status"], "error")

    def test_original_question_and_errors(self):
        query = "임대인의 법적 의무가 궁금해요"
        seen = []
        generator = RunnableLambda(lambda data: seen.append(data) or {
            "status": "answered", "answer": "답변", "cited_chunk_ids": ["LAW_SAMPLE_c01"]})
        handler = build_law_handler(RunnableLambda(lambda q: [statute()]), generator)
        self.assertEqual(handler.invoke(query)["status"], "answered")
        self.assertEqual(seen[0]["question"], query)
        def broken(question):
            raise RuntimeError("local test")
        handler = build_law_handler(RunnableLambda(broken), generator)
        self.assertEqual(handler.invoke(query)["status"], "error")


class RetrievalBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.config = dict(CONFIG)
        apply_preset("r0")

    def tearDown(self):
        CONFIG.clear()
        CONFIG.update(self.config)

    def test_dev_bypasses_classifier_and_filters_statutes(self):
        law = statute()
        guide = Document(page_content="guide", metadata={**law.metadata, "chunk_id": "GUIDE_c1",
                                                          "doc_id": "GUIDE", "category": "일반 전월세 가이드"})
        filters = []
        class Store:
            def as_retriever(self, search_kwargs):
                filters.append(search_kwargs)
                return RunnableLambda(lambda q: [Document(page_content=law.page_content,
                                                            metadata={"chunk_id": law.metadata["chunk_id"]})])
        with patch("retrieve.load_chunks", return_value=[law, guide]), \
                patch("retrieve.OpenAIEmbeddings"), patch("retrieve.PGVector", return_value=Store()), \
                patch("retrieve.route", side_effect=AssertionError("분류 호출 금지")):
            retriever = Retriever("jh_1000", law_only=True)
            for question in ("생활 말투 질문", "LH 공고 단어가 포함된 질문"):
                docs = retriever.documents("A", question)
                self.assertEqual(docs[0].metadata["source_url"], law.metadata["source_url"])
                self.assertEqual(retriever.search_scope(question), "law_statutes")
            self.assertEqual(list(retriever.bm25["law_statutes"].docs), [law])
            self.assertEqual(filters[0]["filter"], {"$and": [
                {"category": {"$eq": "일반 전월세 법령"}}, {"doc_id": {"$in": ["LAW_SAMPLE"]}}]})
            self.assertEqual(retriever.search("B", "생활 말투 질문")[0]["doc_id"], "LAW_SAMPLE")

    def test_legacy_scope_and_stale_index(self):
        law = statute()
        class Store:
            def as_retriever(self, search_kwargs):
                return RunnableLambda(lambda q: [Document(page_content="outdated", metadata=law.metadata)])
        with patch("retrieve.load_chunks", return_value=[law]), patch("retrieve.OpenAIEmbeddings"), \
                patch("retrieve.PGVector", return_value=Store()):
            retriever = Retriever("jh_1000", law_only=True)
            with self.assertRaises(ValueError):
                retriever.documents("A", "질문")
            legacy = Retriever("jh_1000")
            self.assertEqual(legacy.search_scope("질문"), "all")


if __name__ == "__main__":
    unittest.main()
