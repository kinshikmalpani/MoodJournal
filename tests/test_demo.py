import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import prepare_demo
from journal_core import load_entries, save_entry


class DemoTests(unittest.TestCase):
    def test_seed_is_isolated_and_repeat_launch_preserves_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            personal = root / 'local_data' / 'journal.sqlite3'
            save_entry('private sentinel', {'version': 'original'}, path=personal)
            before = personal.read_bytes()
            with patch.object(prepare_demo, 'ROOT', root), patch.object(prepare_demo, 'recommended_model', return_value='demo'), patch.object(prepare_demo, 'load_artifact', return_value={}), patch.object(prepare_demo, 'analyse', return_value={'version': 'demo'}):
                prepare_demo.main()
                first = load_entries(path=root / 'local_data' / 'demo.sqlite3')
                self.assertEqual(len(first), 7)
                prepare_demo.main()
                self.assertEqual(load_entries(path=root / 'local_data' / 'demo.sqlite3'), first)
            self.assertEqual(personal.read_bytes(), before)
