import argparse
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from urllib.error import HTTPError
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import sys
import threading
import types

import api_bench as api
import bench


class APIBenchmarkTests(unittest.TestCase):
    def test_default_system_prompt_is_exact_local_baseline(self):
        self.assertEqual(api.baseline_prompt(), bench.PROMPT)
        self.assertTrue(api.baseline_prompt().endswith('\n'))

    def row(self):
        return {'description': '  Не делаю электрику.\nСантехника!  ',
                'mcId': 101, 'labels': [104], 'row_id': 7}

    def response(self, content='{"labels": [104]}', finish='stop'):
        return {'model': 'test-model', 'choices': [{'message': {'content': content}, 'finish_reason': finish}],
                'usage': {'prompt_tokens': 10, 'completion_tokens': 5}}

    def test_no_gold_or_id_in_request(self):
        request = api.payload_for({'model': 'test-model'}, 'system', self.row())
        self.assertEqual(json.loads(request['messages'][1]['content']),
                         {'mcId': 101, 'description': self.row()['description']})

    def test_preserve_dirty_text_and_empty_gold(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'test.jsonl'
            bench.jsonl(path, [self.row(), {**self.row(), 'labels': []}])
            rows = api.load_test(path, 101)
            self.assertEqual(rows[0]['description'], self.row()['description'])
            self.assertEqual(rows[1]['labels'], [])

    def test_unknown_or_ambiguous_outputs_are_not_guessed(self):
        for text in ('{}', '{"labels":[999]}', '{"labels":[true]}', '{"labels":["104"]}',
                     '{"labels":[104.0]}', '{"labels":[104],"extra":"value"}'):
            with self.subTest(text=text):
                record = api.prediction(self.response(text), self.row(), .1, 1)
                self.assertFalse(record['valid'])
                self.assertEqual(record['labels'], [])

    def test_complete_fenced_json_and_missing_finish_reason_are_usable(self):
        record = api.prediction(self.response('```json\n{"labels":[101,104,104]}\n```', finish=None), self.row(), .1, 1)
        self.assertTrue(record['valid'])
        self.assertEqual(record['labels'], [104])
        self.assertFalse(record['strict_valid'])
        self.assertEqual(set(record['format_issues']), {'markdown_fence_removed', 'source_category_removed',
                                                       'duplicate_ids_removed', 'missing_finish_reason'})
        self.assertIsNone(record['parse_error'])

    def test_missing_finish_reason_does_not_accept_incomplete_json_or_refusal(self):
        for text in ('{"labels":[104', 'Я не могу обсуждать эту тему.'):
            record = api.prediction(self.response(text, finish=None), self.row(), .1, 1)
            self.assertFalse(record['valid'])
        response = self.response()
        response['choices'][0]['message']['refusal'] = 'refusal'
        self.assertFalse(api.prediction(response, self.row(), .1, 1)['valid'])

    def test_truncated_valid_json_is_invalid(self):
        record = api.prediction(self.response(finish='length'), self.row(), .1, 1)
        self.assertFalse(record['valid'])
        self.assertEqual(record['labels'], [])

    def test_retry_and_caila_authorization(self):
        from unittest.mock import MagicMock
        opener = MagicMock()
        success = MagicMock()
        success.__enter__.return_value = io.StringIO(json.dumps(self.response()))
        opener.open.side_effect = [HTTPError('https://example.org', 429, 'rate limit', {}, None), success]
        with patch.object(api.time, 'sleep'):
            response, _, attempts = api.request_json(opener, 'https://example.org', 'chat/completions', 'secret', {})
        self.assertEqual(attempts, 2)
        self.assertEqual(response['model'], 'test-model')
        self.assertEqual(opener.open.call_args.args[0].get_header('Authorization'), 'secret')

    def test_forbidden_not_retried_or_body_leaked(self):
        from unittest.mock import MagicMock
        opener = MagicMock()
        opener.open.side_effect = HTTPError('https://example.org', 403, 'secret data', {}, None)
        with self.assertRaises(api.APIError) as caught:
            api.request_json(opener, 'https://example.org', 'models', 'secret')
        self.assertNotIn('secret', str(caught.exception))
        self.assertEqual(opener.open.call_count, 1)

    def test_quality_gate_uses_optimistic_final_f1_and_warmup(self):
        rows = [{**self.row(), 'labels': [104]} for _ in range(100)]
        bad = [{'labels': [102, 103, 105, 106], 'valid': True} for _ in range(30)]
        policy = {'min_rows': 30, 'target_f1': .8, 'max_invalid_rate': .2}
        self.assertIsNone(api.quality_stop(rows, bad[:29], policy))
        stop = api.quality_stop(rows, bad, policy)
        self.assertEqual(stop['rule'], 'unreachable_target_f1')
        self.assertAlmostEqual(stop['optimistic_final_f1'], 140 / 290)
        good = [{'labels': [104], 'valid': True} for _ in range(30)]
        self.assertIsNone(api.quality_stop(rows, good, policy))
        # A low prefix score alone is insufficient while final 0.8 remains possible.
        missed = [{'labels': [], 'valid': True} for _ in range(30)]
        self.assertIsNone(api.quality_stop(rows, missed, policy))
        self.assertIsNone(api.quality_stop(rows, bad, {**policy, 'min_rows': 0}))

    def test_format_failures_alone_do_not_stop_a_good_classifier(self):
        rows = [{**self.row(), 'labels': [] if i < 10 else [104]} for i in range(100)]
        records = [{'labels': [], 'valid': False} for _ in range(10)] + [
            {'labels': [104], 'valid': True} for _ in range(20)]
        self.assertIsNone(api.quality_stop(rows, records, {'min_rows': 30, 'target_f1': .8, 'max_invalid_rate': .2}))

    def test_budget_screen_rejects_weak_prefix_while_final_target_is_still_reachable(self):
        rows = [{**self.row(), 'row_id': i} for i in range(100)]
        records = [api.prediction(self.response(), row, .1, 1) for row in rows[:30]]
        records[0]['labels'] = []
        records[1]['labels'] = []
        records[2]['labels'] = []
        records[3]['labels'] = []
        policy = {'min_rows': 30, 'target_f1': .93, 'screen_f1': .93}
        self.assertIsNone(api.quality_stop(rows, records[:29], policy))
        stop = api.quality_stop(rows, records, policy)
        self.assertEqual(stop['rule'], 'budget_screen_f1')
        self.assertAlmostEqual(stop['observed_prefix_f1'], 52 / 56)
        self.assertIsNone(api.quality_stop(rows, records, {**policy, 'screen_f1': None}))
        # The screen concerns the initial warm-up, not every later fluctuation.
        good = [api.prediction(self.response(), row, .1, 1) for row in rows[:30]]
        later_misses = [api.prediction(self.response('{"labels":[]}'), row, .1, 1) for row in rows[30:35]]
        self.assertIsNone(api.quality_stop(rows, good + later_misses, policy))

    def test_reuses_matching_raw_prefix_without_paying_again_and_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = [self.row(), {**self.row(), 'row_id': 8}]
            bench.jsonl(root / 'test.jsonl', rows)
            args = argparse.Namespace(out=root / 'old', base_url='https://example.org', timeout=5,
                                      retries=0, proxy=None, direct=True, limit=None)
            entry = {'model': 'test-model'}
            with patch.object(api, 'request_json', side_effect=[
                    (self.response('```json\n{"labels":[104]}\n```', finish=None), .1, 1), api.APIError('offline')]):
                with self.assertRaises(api.APIError):
                    api.evaluate(args, 'clean', root / 'test.jsonl', rows, entry, 'prompt', None, 'secret')
            old = api.run_folder(args.out, 'clean', entry)
            # Simulate a record from the previous strict adapter.
            record = bench.read(old / 'predictions.jsonl')[0]
            record.update(labels=[], valid=False)
            bench.jsonl(old / 'predictions.jsonl', [record])
            args.out, args.reuse_from = root / 'new', root / 'old'
            with patch.object(api, 'request_json', return_value=(self.response(), .1, 1)) as call:
                scores = api.evaluate(args, 'clean', root / 'test.jsonl', rows, entry, 'prompt', None, 'secret')
                self.assertEqual(call.call_count, 1)
            self.assertEqual(scores['micro_f1'], 1)
            self.assertEqual(scores['reused_rows'], 1)
            new = api.run_folder(args.out, 'clean', entry)
            self.assertEqual(bench.read(new / 'predictions.jsonl')[0]['labels'], [104])
            self.assertTrue((new / 'response_schema.json').exists())
            with self.assertRaisesRegex(ValueError, 'Reuse source'):
                api.reuse_predictions(old, rows, entry, 'different prompt', args.base_url)

    def test_early_stop_exports_screening_only_and_resumes_without_requests(self):
        from publish_clearml import publish
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = [{**self.row(), 'row_id': i} for i in range(40)]
            bench.jsonl(root / 'test.jsonl', rows)
            args = argparse.Namespace(out=root / 'runs', base_url='https://example.org', timeout=5,
                                      retries=0, proxy=None, direct=True, limit=None,
                                      early_stop_min_rows=30, target_f1=.8, max_invalid_rate=.2)
            entry = {'model': 'test-model', 'name': 'test'}
            with patch.object(api, 'request_json', return_value=(self.response('{}'), .1, 1)) as call:
                scores = api.evaluate(args, 'clean', root / 'test.jsonl', rows, entry, 'prompt', None, 'secret')
                self.assertEqual(call.call_count, 30)
            self.assertEqual(scores['evaluation_status'], 'early_stopped')
            self.assertEqual(scores['rows'], 30)
            self.assertEqual(scores['planned_rows'], 40)
            with patch.object(api, 'request_json') as call:
                api.evaluate(args, 'clean', root / 'test.jsonl', rows, entry, 'prompt', None, 'secret')
                call.assert_not_called()
            run = api.run_folder(args.out, 'clean', entry)
            self.assertEqual(len(bench.read(run / 'test.jsonl')), 40)
            self.assertEqual(len(bench.read(run / 'predictions.jsonl')), 30)
            task_api = MagicMock()
            task_api.init.return_value.id = 'screen-task'
            task_api.init.return_value.get_output_log_web_page.return_value = 'https://clearml.example/screen'
            with patch.dict(sys.modules, {'clearml': types.SimpleNamespace(Task=task_api)}):
                api.export_result(run, 'project')
            task = task_api.init.return_value
            task.add_tags.assert_called_once_with(['early-stopped', 'screening-only'])
            names = {c.kwargs['name'] for c in task.get_logger.return_value.report_single_value.call_args_list}
            self.assertIn('screening/micro_f1', names)
            self.assertNotIn('micro_f1', names)
            self.assertIn('early-stopped', task_api.init.call_args.kwargs['task_name'])
            table = task.get_logger.return_value.report_table.call_args.kwargs['table_plot']
            self.assertEqual(len(table), 31)
            evaluation = json.loads((run / 'evaluation.json').read_text(encoding='utf-8'))
            evaluation['status'] = 'completed'
            bench.write(run / 'evaluation.json', evaluation)
            with self.assertRaisesRegex(ValueError, 'screening'):
                api.export_result(run, 'project')

    def test_incomplete_run_without_explicit_screening_marker_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = [self.row(), {**self.row(), 'row_id': 8}]
            bench.jsonl(root / 'test.jsonl', rows)
            args = argparse.Namespace(out=root / 'runs', base_url='https://example.org', timeout=5,
                                      retries=0, proxy=None, direct=True, limit=None)
            entry = {'model': 'test-model'}
            with patch.object(api, 'request_json', return_value=(self.response(), .1, 1)):
                api.evaluate(args, 'clean', root / 'test.jsonl', rows, entry, 'prompt', None, 'secret')
            run = api.run_folder(args.out, 'clean', entry)
            bench.jsonl(run / 'predictions.jsonl', bench.read(run / 'predictions.jsonl')[:1])
            (run / 'evaluation.json').unlink()
            scores = json.loads((run / 'metrics.json').read_text(encoding='utf-8'))
            scores['rows'] = 1
            bench.write(run / 'metrics.json', scores)
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                api.export_result(run, 'project')

    def test_resume_and_changed_dataset_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / 'test.jsonl'
            rows = [self.row(), {**self.row(), 'row_id': 8, 'labels': []}]
            bench.jsonl(path, rows)
            args = argparse.Namespace(out=root / 'runs', base_url='https://example.org', timeout=5,
                                      retries=0, proxy=None, direct=True, limit=None)
            entry = {'model': 'test-model'}
            with patch.object(api, 'request_json', side_effect=[(self.response(), .1, 1), api.APIError('offline')]):
                with self.assertRaises(api.APIError):
                    api.evaluate(args, 'dirty', path, rows, entry, 'prompt', None, 'secret')
            with patch.object(api, 'request_json', return_value=(self.response('{"labels":[]}'), .2, 1)) as call:
                scores = api.evaluate(args, 'dirty', path, rows, entry, 'prompt', None, 'secret')
                self.assertEqual(call.call_count, 1)
                self.assertEqual(scores['micro_f1'], 1)
            with patch.object(api, 'request_json') as call:
                api.evaluate(args, 'dirty', path, rows, entry, 'prompt', None, 'secret')
                call.assert_not_called()
                with self.assertRaisesRegex(ValueError, 'changed'):
                    api.evaluate(args, 'dirty', path, rows, entry, 'new prompt', None, 'secret')
                call.assert_not_called()
            for output in args.out.rglob('*'):
                if output.is_file():
                    self.assertNotIn('secret', output.read_text(encoding='utf-8'))

    def test_missing_clean_not_silently_substituted(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'test.jsonl'
            bench.jsonl(path, [self.row()])
            argv = ['api_bench.py', '--dry-run', '--dirty', str(path), '--clean', str(path.parent / 'missing.jsonl')]
            with patch('sys.argv', argv), self.assertRaises(FileNotFoundError):
                api.main()

    def test_real_http_cli_exports_two_datasets_and_resumes_without_paid_calls(self):
        requests = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"data":[{"id":"test-model"}]}')

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append(payload)
                user = json.loads(payload['messages'][1]['content'])
                assert set(user) == {'description', 'mcId'}
                content = json.dumps({'labels': [104] if 'Сантехника' in user['description'] else []})
                body = {'model': 'resolved-test', 'choices': [
                    {'message': {'content': content}, 'finish_reason': 'stop'}],
                    'usage': {'prompt_tokens': 10, 'completion_tokens': 5}}
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(body).encode())

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                bench.jsonl(root / 'dirty.jsonl', [self.row(), {**self.row(), 'row_id': 8,
                              'description': 'Только ремонт', 'labels': []}])
                bench.jsonl(root / 'clean.jsonl', [self.row()])
                bench.write(root / 'models.json', [{'model': 'test-model', 'name': 'test'}])
                task_api = MagicMock()
                tasks = [MagicMock(), MagicMock()]
                for i, task in enumerate(tasks):
                    task.id = f'task-{i}'
                    task.get_output_log_web_page.return_value = f'https://clearml.example/task-{i}'
                task_api.init.side_effect = tasks
                argv = ['api_bench.py', '--config', str(root / 'models.json'),
                        '--dirty', str(root / 'dirty.jsonl'), '--clean', str(root / 'clean.jsonl'),
                        '--out', str(root / 'runs'), '--base-url', f'http://127.0.0.1:{server.server_port}', '--direct']
                with patch.dict(sys.modules, {'clearml': types.SimpleNamespace(Task=task_api)}), \
                        patch.dict(api.os.environ, {'CAILA_API_KEY': 'not-a-real-key'}), patch('sys.argv', argv):
                    api.main()
                    api.main()
                self.assertEqual(len(requests), 3)
                self.assertEqual(task_api.init.call_count, 2)
                self.assertEqual({p['messages'][0]['content'] for p in requests}, {bench.PROMPT})
                for task in tasks:
                    task.close.assert_called_once()
                    uploaded = {c.kwargs['name'] for c in task.upload_artifact.call_args_list}
                    self.assertTrue({'test.jsonl', 'prompt.txt', 'bench.py', 'run.json', 'predictions.jsonl'} <= uploaded)
                    self.assertNotIn('train.jsonl', uploaded)
                    self.assertIn('system_prompt', [c.kwargs.get('name') for c in task.connect.call_args_list])
                summaries = json.loads((root / 'runs/summary.json').read_text(encoding='utf-8'))
                self.assertEqual({r['dataset'] for r in summaries}, {'dirty', 'clean'})
                self.assertTrue(all(r['micro_f1'] == 1 for r in summaries))
                before = len(requests)
                with patch.dict(sys.modules, {'clearml': types.SimpleNamespace(Task=task_api)}), \
                        patch.dict(api.os.environ, {'CAILA_API_KEY': ''}), \
                        patch('sys.argv', ['api_bench.py', '--export-only', '--out', str(root / 'runs')]):
                    api.main()
                self.assertEqual(len(requests), before)
                self.assertEqual(task_api.init.call_count, 2)
                # Clean-only must not load or send the default dirty dataset.
                clean_only = ['api_bench.py', '--config', str(root / 'models.json'),
                              '--skip-dirty', '--dirty', str(root / 'does-not-exist.jsonl'),
                              '--clean', str(root / 'clean.jsonl'), '--out', str(root / 'clean-only'),
                              '--base-url', f'http://127.0.0.1:{server.server_port}', '--direct', '--no-upload']
                with patch.dict(api.os.environ, {'CAILA_API_KEY': 'not-a-real-key'}), patch('sys.argv', clean_only):
                    api.main()
                self.assertEqual(len(requests), 4)
                self.assertFalse((root / 'clean-only/dirty').exists())
                clean_scores = json.loads((root / 'clean-only/summary.json').read_text(encoding='utf-8'))
                self.assertEqual([s['dataset'] for s in clean_scores], ['clean'])
                result_folder = next((root / 'runs').glob('dirty/*/metrics.json')).parent
                scores = json.loads((result_folder / 'metrics.json').read_text(encoding='utf-8'))
                scores['micro_f1'] = .123
                bench.write(result_folder / 'metrics.json', scores)
                with self.assertRaisesRegex(ValueError, 'metrics'):
                    api.export_result(result_folder, 'avito-microcategories')
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_export_failure_can_retry_and_does_not_leave_success_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = [self.row()]
            bench.jsonl(root / 'test.jsonl', rows)
            args = argparse.Namespace(out=root / 'runs', base_url='https://example.org', timeout=5,
                                      retries=0, proxy=None, direct=True, limit=None)
            entry = {'model': 'test-model'}
            with patch.object(api, 'request_json', return_value=(self.response(), .1, 1)):
                api.evaluate(args, 'dirty', root / 'test.jsonl', rows, entry, bench.PROMPT, None, 'secret')
            result_folder = api.run_folder(args.out, 'dirty', entry)
            with patch('publish_clearml.publish', side_effect=RuntimeError('offline')):
                with self.assertRaises(RuntimeError):
                    api.export_result(result_folder, 'project')
            self.assertFalse((result_folder / 'clearml.json').exists())
            with patch('publish_clearml.publish', return_value={'task_id': 'id', 'url': 'https://example.org/task'}) as publish:
                api.export_result(result_folder, 'project')
                api.export_result(result_folder, 'project')
                self.assertEqual(publish.call_count, 1)


if __name__ == '__main__':
    unittest.main()
