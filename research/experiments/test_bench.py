import argparse
import json
from pathlib import Path
import tempfile
import unittest

import bench


class ProtocolTests(unittest.TestCase):
    def test_gold_category_columns(self):
        row = {"description": "Только сантехника", **{f"y_{x}": int(x == 104) for x in range(102, 107)}}
        self.assertEqual(bench.canonical(row, 101)["labels"], [104])

    def test_prepare_parquet_train_and_gold(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = [{"description": f"Объявление {i}", "labels": [104]} for i in range(60)]
            pq.write_table(pa.Table.from_pylist(rows), root / "train.parquet")
            test = [{"description": "Отдельное тестовое", **{f"y_{x}": int(x == 104) for x in range(102, 107)}}]
            pq.write_table(pa.Table.from_pylist(test), root / "test.parquet")
            bench.prepare(argparse.Namespace(train=str(root / "train.parquet"),
                test=str(root / "test.parquet"), out=str(root / "prepared"), assume_source=101))
            self.assertEqual(bench.read(root / "prepared/test.jsonl")[0]["labels"], [104])

    def test_interrupted_record_recovery(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'predictions.jsonl'
            path.write_bytes(b'{"labels": []}\n{"labels": [10')
            self.assertEqual(bench.read_predictions(path), [{'labels': []}])
            self.assertEqual(path.read_bytes(), b'{"labels": []}\n')

    def test_clean_preserves_negation_and_lines(self):
        self.assertEqual(bench.clean('  Плитку\u00a0не кладём.\r\n Только сантехника. '),
                         'Плитку не кладём.\nТолько сантехника.')

    def test_strict_output(self):
        for text in ['{"labels":[101]}', '{"labels":[105,105]}',
                     '{"labels":[true]}', '```json\n{"labels":[105]}\n```', 'null']:
            self.assertEqual(bench.parse_response(text, 101), ([], False))
        self.assertEqual(bench.parse_response('{"labels":[105]}', 101), ([105], True))

    def test_source_required(self):
        with self.assertRaises(ValueError):
            bench.canonical({'description': 'abc', 'labels': []})

    def test_prepare_excludes_leakage_and_conflicts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = [{'description': f'text {i}', 'mcId': 101, 'labels': [105]} for i in range(40)]
            rows += [rows[0], {**rows[1], 'labels': []}]
            bench.jsonl(root / 'train.jsonl', rows)
            bench.jsonl(root / 'test.jsonl', [rows[2], {**rows[2], 'labels': []}])
            args = argparse.Namespace(train=root / 'train.jsonl', test=root / 'test.jsonl',
                                      out=root / 'v1', assume_source=None)
            bench.prepare(args)
            train, dev = [bench.read(root / 'v1' / f'{split}.jsonl') for split in ['train', 'dev']]
            descriptions = {r['description'] for r in train + dev}
            self.assertNotIn('text 1', descriptions)
            self.assertNotIn('text 2', descriptions)
            self.assertEqual(len(descriptions), len(train) + len(dev))
            manifest = json.loads((root / 'v1/manifest.json').read_text())
            self.assertEqual(manifest['conflicting_test_groups_retained'], 1)
            with self.assertRaises(ValueError):
                bench.prepare(args)

    def test_metric_false_positive_and_empty(self):
        gold = [{'labels': [105], 'description': 'не'}, {'labels': [], 'description': 'abc'}]
        pred = [{'labels': [105], 'valid': True, 'seconds': .1},
                {'labels': [104], 'valid': True, 'seconds': .2}]
        result = bench.metrics(gold, pred)
        self.assertAlmostEqual(result['micro_f1'], 2 / 3)
        self.assertEqual(result['exact_match'], .5)


if __name__ == '__main__':
    unittest.main()
