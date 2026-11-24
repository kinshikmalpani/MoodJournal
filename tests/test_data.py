import unittest
from train_baseline import prepare_data


class DataTests(unittest.TestCase):
    def test_splits_have_no_normalized_text_overlap(self):
        splits, audit = prepare_data()
        normalized = {name: {' '.join(text.lower().split()) for text, _ in rows} for name, rows in splits.items()}
        self.assertFalse(normalized['train'] & normalized['val'])
        self.assertFalse(normalized['train'] & normalized['test'])
        self.assertFalse(normalized['val'] & normalized['test'])
        self.assertEqual(audit['test']['retained_rows'], 2000)
        for name in splits:
            self.assertEqual(len(splits[name]), len(normalized[name]))


if __name__ == '__main__':
    unittest.main()
