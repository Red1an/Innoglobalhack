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


class OllamaTests(unittest.TestCase):
    def test_pull_progress(self):
        import io
        from contextlib import redirect_stdout
        from unittest.mock import patch
        lines = [b'{"status": "pulling manifest"}\n', b'{"status": "pulling x", "total": 100, "completed": 0}\n',
                 b'{"status": "pulling x", "total": 100, "completed": 100}\n', b'{"status": "success"}\n']

        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        output = io.StringIO()
        with patch('urllib.request.urlopen', return_value=Response(b''.join(lines))), redirect_stdout(output):
            bench.ollama_pull('model', every=0)
        self.assertIn('(100%)', output.getvalue())

    def test_pull_error(self):
        import io
        from unittest.mock import patch

        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        with patch('urllib.request.urlopen', return_value=Response(b'{"error": "not found"}\n')):
            with self.assertRaises(RuntimeError):
                bench.ollama_pull('model')

    def test_missing_gguf_fails_before_download(self):
        from unittest.mock import patch
        with patch.object(bench, 'ollama_request', return_value={'models': []}):
            with self.assertRaises(FileNotFoundError):
                bench.ollama_model(bench.OLLAMA_MODELS['t-lite-q8'], '/nonexistent/folder')

    def fake(self, prompt_tokens):
        tag = bench.OLLAMA_MODELS['t-lite-q8']['name']

        def request(path, payload=None, timeout=600):
            if path == '/api/version':
                return {'version': '0.0.0'}
            if path == '/api/tags':
                return {'models': [{'name': tag, 'digest': 'abc', 'details': {'quantization_level': 'Q8_0'}}]}
            if path == '/api/generate':
                self.assertEqual(payload['keep_alive'], 0)
                return {}
            self.assertEqual(payload['options']['temperature'], 0)
            return {'message': {'content': '{"labels": [105]}'}, 'prompt_eval_count': prompt_tokens, 'eval_count': 7}
        return request

    def args(self):
        return argparse.Namespace(model='t-lite-q8', max_length=4096, max_new_tokens=96, gguf=None)

    def test_prediction_and_metadata(self):
        from unittest.mock import patch
        metadata = {}
        with patch.object(bench, 'ollama_request', self.fake(500)):
            records = list(bench.ollama(self.args(), [{'mcId': 101, 'description': 'Электрика'}], metadata))
        self.assertEqual(records[0]['labels'], [105])
        self.assertTrue(records[0]['valid'])
        self.assertEqual(metadata['quantization'], 'Q8_0')

    def test_truncated_context_fails(self):
        from unittest.mock import patch
        with patch.object(bench, 'ollama_request', self.fake(4050)):
            with self.assertRaises(ValueError):
                list(bench.ollama(self.args(), [{'mcId': 101, 'description': 'Электрика'}], {}))


class EmbeddingTests(unittest.TestCase):
    def test_head_trains_and_predicts(self):
        import types
        from unittest.mock import patch
        import torch

        class Tokenizer:
            def __call__(self, texts, padding=False, return_tensors=None):
                if isinstance(texts, str):
                    return {'input_ids': list(range(len(texts.split())))}
                width = max(len(t.split()) for t in texts)
                ids = torch.tensor([[hash(w) % 997 for w in t.split()] + [0] * (width - len(t.split())) for t in texts])
                mask = torch.tensor([[1] * len(t.split()) + [0] * (width - len(t.split())) for t in texts])
                return types.SimpleNamespace(to=lambda device: {'input_ids': ids, 'attention_mask': mask})

        class Model(torch.nn.Module):
            config = types.SimpleNamespace(hidden_size=8)

            def __init__(self):
                super().__init__()
                torch.manual_seed(0)
                self.table = torch.nn.Embedding(997, 8)

            def forward(self, input_ids, attention_mask):
                return types.SimpleNamespace(last_hidden_state=self.table(input_ids))

        rows = lambda n: [{'mcId': 101, 'description': f'Объявление {i} ' + ('электрика проводка' if i % 2 else 'плитка кафель'),
                           'labels': [105] if i % 2 else [103]} for i in range(n)]
        args = argparse.Namespace(max_length=4096, batch_size=4)
        with patch('transformers.AutoTokenizer.from_pretrained', return_value=Tokenizer()), \
                patch('transformers.AutoModel.from_pretrained', return_value=Model()), \
                patch('torch.cuda.is_available', return_value=False), \
                patch('torch.backends.mps.is_available', return_value=False):
            metadata = {}
            records = list(bench.embeddings(args, rows(40), rows(10), rows(4), metadata))
        self.assertEqual([r['labels'] for r in records], [[103], [105], [103], [105]])
        self.assertEqual(metadata['resolved_revision'], bench.GIGA_REVISION)


if __name__ == '__main__':
    unittest.main()
