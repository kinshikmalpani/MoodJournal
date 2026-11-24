import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
import journal_core


class AppTests(unittest.TestCase):
    @unittest.skipUnless((journal_core.ROOT / 'artifacts' / 'distilbert' / 'metadata.json').exists(), 'Transformer not finalized')
    def test_switch_model_preserves_saved_predictions(self):
        real_connect = journal_core.connect
        with tempfile.TemporaryDirectory() as tmp:
            database = Path(tmp) / 'models.sqlite3'
            with patch('journal_core.connect', side_effect=lambda *args, **kwargs: real_connect(database)):
                app = AppTest.from_file(str(journal_core.ROOT / 'app.py')).run(timeout=180)
                selector = next(item for item in app.selectbox if item.label == 'Analysis model')
                selector.set_value('baseline').run(timeout=180)
                app.text_area[0].set_value('I feel happy and excited about today.')
                app.button[0].click().run(timeout=180)
                original = journal_core.load_entries()[0]
                from model_catalog import recommended_model
                selected = recommended_model()
                next(item for item in app.selectbox if item.label == 'Analysis model').set_value(selected).run(timeout=180)
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(journal_core.load_entries()[0]['result'], original['result'])
                app.text_area[0].set_value('I am furious about the unfair decision. I feel loved by my family.')
                app.button[0].click().run(timeout=180)
                self.assertEqual(len(app.exception), 0)
                rows = journal_core.load_entries()
                self.assertEqual(len(rows), 2)
                self.assertNotEqual(rows[0]['result']['version'], rows[1]['result']['version'])
                self.assertEqual(rows[0]['result'], original['result'])
                from model_catalog import read_report
                self.assertEqual(rows[1]['result']['version'], read_report(selected)['model_version'])

    def test_save_persist_export_and_delete(self):
        real_connect = journal_core.connect
        with tempfile.TemporaryDirectory() as tmp:
            database = Path(tmp) / 'ui.sqlite3'
            with patch('journal_core.connect', side_effect=lambda *args, **kwargs: real_connect(database)):
                app = AppTest.from_file(str(journal_core.ROOT / 'app.py')).run(timeout=60)
                self.assertEqual(len(app.exception), 0)
                app.text_area[0].set_value('I feel happy and excited about my wonderful day.')
                app.button[0].click().run(timeout=60)
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(len(journal_core.load_entries()), 1)
                self.assertEqual(len(app.get('download_button')), 1)
                app.run(timeout=60)
                self.assertEqual(len(journal_core.load_entries()), 1)
                delete = next(button for button in app.button if button.label == 'Delete this entry')
                delete.click().run(timeout=60)
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(journal_core.load_entries(), [])


if __name__ == '__main__':
    unittest.main()
