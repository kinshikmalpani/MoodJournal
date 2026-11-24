import unittest
from unittest.mock import patch
from model_catalog import recommended_model, validation_score


class ModelSelectionTests(unittest.TestCase):
    def test_selection_uses_validation_not_test(self):
        reports = {
            'a': {'validation_trials': [{'validation_macro_f1': .8}], 'test': {'macro_f1': .99}},
            'b': {'best_validation_macro_f1': .9, 'test': {'macro_f1': .7}},
        }
        with patch('model_catalog.read_report', side_effect=reports.__getitem__):
            self.assertEqual(recommended_model(['a', 'b']), 'b')
        self.assertEqual(validation_score(reports['a']), .8)
