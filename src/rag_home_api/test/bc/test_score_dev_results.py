import unittest
from score_dev_results import score_records


class ScoreDevTests(unittest.TestCase):
    def setUp(self):
        self.questions = [{'id': 'D01', 'question': '질문1', 'reference_answerable': True},
                          {'id': 'D02', 'question': '질문2', 'reference_answerable': True}]
        self.accepted = {'D01': {'tx1'}, 'D02': {'tx2'}}

    def record(self, **changes):
        return {'system': 'A', 'question_id': 'D01', 'question': '질문1',
                'status': 'success', 'retrieved_ids': ['chunk1'], 'response': '', **changes}

    def test_errors_and_clarifications_stay_in_denominator(self):
        records = [self.record(), self.record(question_id='D02', question='질문2',
                   status='needs_clarification', retrieved_ids=[])]
        score = score_records(records, self.questions, self.accepted, {'chunk1': 'tx1'})
        self.assertEqual(score['systems']['A']['hit_at_5'], 0.5)
        self.assertEqual(score['systems']['A']['clarification_count'], 1)
        self.assertEqual(score['systems']['B']['error_or_missing_count'], 2)

    def test_failed_call_does_not_count_as_hit(self):
        score = score_records([self.record(status='error')], self.questions, self.accepted, {'chunk1': 'tx1'})
        self.assertEqual(score['systems']['A']['hit_at_5_count'], 0)

    def test_duplicates_and_unknown_chunks_rejected(self):
        for records in ([self.record(), self.record()], [self.record(retrieved_ids=['unknown'])]):
            with self.assertRaises(ValueError):
                score_records(records, self.questions, self.accepted, {'chunk1': 'tx1'})

    def test_only_first_five_chunks_count(self):
        ids = ['bad'] * 5 + ['chunk1']
        score = score_records([self.record(retrieved_ids=ids)], self.questions,
                              self.accepted, {'chunk1': 'tx1', 'bad': 'wrong'})
        self.assertEqual(score['systems']['A']['hit_at_5_count'], 0)


if __name__ == '__main__':
    unittest.main()
