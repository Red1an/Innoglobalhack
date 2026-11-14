import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

from bench import digest, jsonl, write
from publish_clearml import publish


class ExportTests(unittest.TestCase):
    def test_uploads_completed_result_without_description_column(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, run = root / 'data', root / 'run'
            data.mkdir()
            run.mkdir()
            rows = [{'description': 'test', 'mcId': 101, 'labels': [105]}]
            jsonl(data / 'test.jsonl', rows)
            manifest = {'hashes': {'test': digest(rows)}}
            write(data / 'manifest.json', manifest)
            write(run / 'run.json', {'dataset': manifest, 'config': {'model': 'avibe'}, 'signature': 'test'})
            write(run / 'metrics.json', {'micro_f1': 1.0, 'per_category': {'105': {'f1': 1.0}}})
            jsonl(run / 'predictions.jsonl', [{'labels': [105], 'row_hash': digest(rows[0]), 'valid': True, 'seconds': .1}])
            (run / 'protocol.py').write_text('# test', encoding='utf-8')
            task_api = MagicMock()
            task = task_api.init.return_value
            with patch.dict(sys.modules, {'clearml': types.SimpleNamespace(Task=task_api)}):
                publish(run, data, 'test-project')
            self.assertEqual(task.upload_artifact.call_count, 5)
            table = task.get_logger.return_value.report_table.call_args.kwargs['table_plot']
            self.assertNotIn('description', table[0])
            self.assertEqual(table[1][2], '[105]')
            task.close.assert_called_once()
            task.mark_failed.assert_not_called()
            self.assertFalse(task_api.init.call_args.kwargs['auto_connect_streams'])

    def test_invalid_test_is_rejected_before_sdk_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {'hashes': {'test': 'wrong'}}
            write(root / 'run.json', {'dataset': manifest})
            write(root / 'manifest.json', manifest)
            write(root / 'metrics.json', {})
            jsonl(root / 'test.jsonl', [])
            jsonl(root / 'predictions.jsonl', [])
            with self.assertRaisesRegex(ValueError, 'modified test'):
                publish(root, root, 'test-project')


if __name__ == '__main__':
    unittest.main()
