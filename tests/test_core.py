import datetime as dt
import tempfile
import unittest
from pathlib import Path
from journal_core import save_entry, load_entries, daily_summary, negative_streak, delete_entry, passages
from journal_core import analyse
import numpy as np


class JournalTests(unittest.TestCase):
    def test_mixed_predictions_use_model_label_order(self):
        class Predictor:
            classes_ = np.array(['joy', 'sadness'])
            def predict_proba(self, texts):
                return np.array([[.95, .05], [.1, .9]])
        result = analyse('A happy morning. A sad evening.', {'pipeline': Predictor(), 'threshold': .5, 'version': 'test'})
        self.assertEqual(result['label'], 'mixed')
        self.assertEqual(result['detected_emotions'], ['joy', 'sadness'])
        self.assertEqual([p['label'] for p in result['passages']], ['joy', 'sadness'])
        self.assertAlmostEqual(sum(result['scores'].values()), 1.)

    def test_rejects_empty_or_nonverbal_input(self):
        for text in ['', '   ', '12345 !!!']:
            with self.assertRaises(ValueError):
                analyse(text, {})

    def test_multiple_entries_are_one_day_and_calendar_gaps_break_streak(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'test.sqlite3'
            end = dt.date(2026, 11, 11)
            result = {'scores': {'sadness': .8, 'joy': .2}}
            for day in [end, end, end, end - dt.timedelta(days=2)]:
                save_entry('example', result, day=day, path=path)
            rows = load_entries(days=2, end=end, path=path)
            self.assertEqual(len(rows), 3)
            self.assertEqual(len(daily_summary(rows)), 1)
            self.assertEqual(negative_streak(daily_summary(load_entries(path=path)), end=end), 1)
            save_entry('yesterday', result, day=end-dt.timedelta(days=1), path=path)
            self.assertEqual(negative_streak(daily_summary(load_entries(path=path)), end=end), 3)
            delete_entry(rows[0]['id'], path=path)
            self.assertEqual(len(load_entries(path=path)), 4)

    def test_chunking_keeps_beginning_and_end(self):
        text = ' '.join(f'word{i}' for i in range(201))
        chunks = passages(text)
        self.assertEqual(' '.join(chunks), text)
        self.assertTrue(all(len(c.split()) <= 80 for c in chunks))


if __name__ == '__main__':
    unittest.main()
