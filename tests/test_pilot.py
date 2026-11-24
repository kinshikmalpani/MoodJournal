import json
import unittest
from research.evaluate_pilot import trend_metrics, DATA


class PilotEvaluationTests(unittest.TestCase):
    def test_calendar_gaps_are_not_adjacent_transitions(self):
        rows = [{'person': 'p', 'day_index': day, 'intended_passages': [{'label': label}], 'passage_labels': [label]} for day, label in [(1, 'joy'), (3, 'sadness')]]
        result = trend_metrics(rows)
        self.assertEqual(result['changed_class_transitions'], 0)
        self.assertEqual(result['unchanged_class_transitions'], 0)
        self.assertEqual(result['daily_frequency_mae'], 0)

    def test_equal_entry_weight_and_direction(self):
        rows = [
            {'person': 'p', 'day_index': 1, 'intended_passages': [{'label': 'joy'}], 'passage_labels': ['joy']},
            {'person': 'p', 'day_index': 1, 'intended_passages': [{'label': 'sadness'}] * 3, 'passage_labels': ['sadness'] * 3},
            {'person': 'p', 'day_index': 2, 'intended_passages': [{'label': 'joy'}], 'passage_labels': ['sadness']},
        ]
        result = trend_metrics(rows)
        self.assertEqual(result['daily'][0]['intended_frequency']['joy'], .5)
        self.assertEqual(result['changed_class_transitions'], 2)
        self.assertEqual(result['changed_direction_accuracy'], 0)
        self.assertEqual(result['unchanged_direction_accuracy'], 1)

    def test_synthetic_segment_labels_align_with_runtime_segmentation(self):
        from journal_core import passages
        for row in map(json.loads, (DATA / 'synthetic.jsonl').read_text().splitlines()):
            self.assertEqual(passages(row['text']), [p['text'] for p in row['intended_passages']])
