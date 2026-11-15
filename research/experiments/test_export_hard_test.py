from pathlib import Path
import tempfile
import unittest
import zipfile

from export_hard_test import load_hard_test


SOURCE = Path(__file__).resolve().parents[2] / 'data/hard/hard_v0.xlsx'


class HardTestExportTests(unittest.TestCase):
    def test_read_preserves_source_and_label_contract(self):
        original = SOURCE.read_bytes()
        rows = load_hard_test(SOURCE)
        self.assertTrue(rows)
        self.assertEqual(len(rows), len({r['row_id'] for r in rows}))
        self.assertTrue(all(r['mcId'] == 101 and set(r['labels']) <= set(range(102, 107)) for r in rows))
        self.assertEqual(SOURCE.read_bytes(), original)

    def test_formula_is_rejected_instead_of_using_stale_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'formula.xlsx'
            with zipfile.ZipFile(SOURCE) as source, zipfile.ZipFile(target, 'w') as dest:
                for entry in source.infolist():
                    content = source.read(entry.filename)
                    if entry.filename == 'xl/worksheets/sheet1.xml':
                        content = content.replace(b'<v>0</v>', b'<f>1-1</f><v>0</v>', 1)
                    dest.writestr(entry, content)
            with self.assertRaisesRegex(ValueError, 'Formula'):
                load_hard_test(target)


if __name__ == '__main__':
    unittest.main()
