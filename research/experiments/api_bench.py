"""Caila/OpenAI-compatible zero-shot evaluation; no training or gold in requests."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import importlib.metadata
import re
import socket
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

import bench
import response_contract

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE_URL = 'https://caila.io/api/adapters/openai'


class APIError(RuntimeError):
    """Sanitized transport error; never includes headers or provider body."""


def api_opener(proxy=None, direct=False):
    if proxy and direct:
        raise ValueError('Choose --proxy or --direct, not both')
    # Explicit direct bypasses HTTP proxy settings, but cannot bypass a VPN tunnel.
    handler = ProxyHandler({'http': proxy, 'https': proxy}) if proxy else ProxyHandler({}) if direct else ProxyHandler()
    return build_opener(handler)


def request_json(opener, base_url, path, api_key, payload=None, timeout=60, retries=2):
    headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
    if api_key:
        # Caila documents the key itself in Authorization (no automatic Bearer).
        headers['Authorization'] = api_key
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode('utf-8')
    start = time.perf_counter()
    for attempt in range(retries + 1):
        req = Request(base_url.rstrip('/') + '/' + path, data=body, headers=headers)
        try:
            with opener.open(req, timeout=timeout) as response:
                try:
                    value = json.load(response)
                except (ValueError, UnicodeError):
                    raise APIError('API returned non-JSON data; check endpoint and access') from None
            if not isinstance(value, dict) or 'error' in value:
                raise APIError('API returned an error or unexpected envelope')
            return value, time.perf_counter() - start, attempt + 1
        except HTTPError as exc:
            status = exc.code
            exc.close()
            if status not in (408, 429, 500, 502, 503, 504) or attempt == retries:
                raise APIError(f'HTTP {status}; check access, model, parameters and quota') from None
        except (URLError, TimeoutError, socket.timeout):
            if attempt == retries:
                raise APIError('Connection failed or timed out; check VPN/proxy and endpoint') from None
        time.sleep(min(2 ** attempt, 8))
    raise AssertionError('unreachable')


def load_test(path, source):
    if path.suffix.lower() == '.parquet':
        import pyarrow.parquet as pq
        raw = pq.read_table(path).to_pylist()
    else:
        raw = bench.read(path)
    rows = []
    for index, item in enumerate(raw):
        row = bench.canonical(item, source)
        # Preserve original dirty/clean input exactly, including whitespace.
        row['description'] = item['description']
        row['row_id'] = item.get('row_id', item.get('id', index))
        rows.append(row)
    if not rows:
        raise ValueError('Test dataset is empty')
    return rows


def load_models(path):
    models = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(models, list) or not models:
        raise ValueError('Model config must be a nonempty list')
    seen = set()
    for entry in models:
        model = entry.get('model')
        if not isinstance(model, str) or not model.strip() or model in seen:
            raise ValueError('Model IDs must be nonempty and unique')
        params = entry.get('params', {})
        if not isinstance(params, dict) or {'model', 'messages', 'stream'} & params.keys():
            raise ValueError('params must be an object without model/messages/stream')
        seen.add(model)
    return models


def payload_for(entry, prompt, row):
    return {**entry.get('params', {}), 'model': entry['model'], 'stream': False,
            'messages': [{'role': 'system', 'content': prompt},
                         {'role': 'user', 'content': json.dumps(
                             {'mcId': row['mcId'], 'description': row['description']}, ensure_ascii=False)}]}


def prediction(response, row, seconds, attempts):
    choices = response.get('choices')
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise APIError('API response has no completion choice; row was not committed')
    choice = choices[0]
    message = choice.get('message') or {}
    content = message.get('content') if isinstance(message, dict) else None
    strict_labels, strict_valid = bench.parse_response(content, row['mcId']) if isinstance(content, str) else ([], False)
    finish = choice.get('finish_reason')
    if finish != 'stop':
        strict_labels, strict_valid = [], False
    labels, issues, error = response_contract.parse_labels(content, row['mcId'])
    if finish is None:
        issues.append('missing_finish_reason')
    elif finish != 'stop':
        error = 'unfinished_or_filtered_response'
    if isinstance(message, dict) and message.get('refusal'):
        error = 'provider_refusal'
    valid = error is None
    if not valid:
        labels = []
    return {'row_hash': bench.digest(row), 'row_id': row['row_id'], 'labels': labels,
            'valid': valid, 'seconds': seconds, 'attempts': attempts,
            'strict_labels': strict_labels, 'strict_valid': strict_valid,
            'response_contract': response_contract.VERSION, 'format_issues': issues, 'parse_error': error,
            'raw': content, 'finish_reason': choice.get('finish_reason'),
            'refusal': message.get('refusal') if isinstance(message, dict) else None,
            'response_model': response.get('model'), 'request_id': response.get('id'),
            'usage': response.get('usage') or {}}


def run_folder(out, name, entry):
    slug = re.sub(r'[^a-zA-Z0-9_.-]', '_', entry.get('name', entry['model']))
    return Path(out) / name / (slug + '-' + bench.digest(entry['model'])[:8])


def baseline_prompt(path=None):
    # Keep the exact text (including its trailing newline) used by local LLMs.
    return bench.PROMPT if path is None else Path(path).read_text(encoding='utf-8-sig')


def quality_policy(args):
    return {'min_rows': getattr(args, 'early_stop_min_rows', 30),
            'target_f1': getattr(args, 'target_f1', .93),
            'screen_f1': getattr(args, 'screen_f1', None)}


def quality_stop(rows, records, policy):
    """Stop only after warm-up; F1 bound assumes perfect remaining predictions."""
    count = len(records)
    if not policy['min_rows'] or count < policy['min_rows'] or count >= len(rows):
        return None
    screen_f1 = policy.get('screen_f1')
    if screen_f1 is not None:
        screen_rows = policy['min_rows']
        observed = float(bench.metrics(rows[:screen_rows], records[:screen_rows])['micro_f1'])
        if observed < screen_f1:
            return {'rule': 'budget_screen_f1', 'observed_prefix_f1': observed,
                    'threshold': screen_f1, 'screen_rows': screen_rows,
                    'note': 'Budget heuristic on a partial test; not a full-test score'}
    tp = fp = fn = 0
    for row, record in zip(rows, records):
        gold, predicted = set(row['labels']), set(record['labels'])
        tp += len(gold & predicted)
        fp += len(predicted - gold)
        fn += len(gold - predicted)
    best_tp = tp + sum(len(row['labels']) for row in rows[count:])
    denominator = 2 * best_tp + fp + fn
    upper = 2 * best_tp / denominator if denominator else 0.0
    if upper < policy['target_f1']:
        return {'rule': 'unreachable_target_f1', 'optimistic_final_f1': upper,
                'threshold': policy['target_f1']}
    return None


def reuse_predictions(source_folder, rows, entry, prompt, base_url):
    """Reparse an exact compatible saved prefix without requesting it again."""
    folder = Path(source_folder)
    metadata = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
    saved_rows, records = bench.read(folder / 'test.jsonl'), bench.read(folder / 'predictions.jsonl')
    if (metadata['model'] != entry or metadata['prompt'] != prompt
            or metadata['base_url'] != base_url or saved_rows != rows
            or not 0 < len(records) <= len(rows)
            or any(q['row_hash'] != bench.digest(r) for r, q in zip(rows, records))):
        raise ValueError('Reuse source does not match model, parameters, prompt or dataset')
    converted = []
    for row, record in zip(rows, records):
        if not {'raw', 'finish_reason', 'usage', 'seconds', 'attempts'} <= record.keys():
            raise ValueError('Reuse source lacks raw response evidence')
        response = {'model': record.get('response_model'), 'id': record.get('request_id'),
                    'usage': record['usage'], 'choices': [{'finish_reason': record['finish_reason'],
                    'message': {'content': record['raw'], 'refusal': record.get('refusal')}}]}
        converted.append(prediction(response, row, record['seconds'], record['attempts']))
    provenance = {'source_folder': str(folder.resolve()), 'source_signature': metadata['signature'],
                  'source_predictions_hash': bench.digest(records), 'reused_rows': len(converted)}
    return converted, provenance


def evaluate(args, name, path, rows, entry, prompt, opener, api_key):
    folder = run_folder(args.out, name, entry)
    protocol = Path(__file__).read_bytes()
    manifest = {'kind': 'api-zero-shot', 'test': len(rows), 'hashes': {'test': bench.digest(rows)},
                'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'source_assumption': getattr(args, 'assume_source', 101),
                'preprocess': 'preserve-description-v1', 'training': 'none'}
    versions = {}
    for package in ('numpy', 'scikit-learn', 'pyarrow', 'clearml', 'pydantic'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    metadata = {'dataset_name': name, 'dataset_path': str(path.resolve()),
                'dataset_file_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'test_hash': bench.digest(rows), 'rows': len(rows), 'model': entry,
                'base_url': args.base_url, 'prompt': prompt,
                'protocol_hash': hashlib.sha256(protocol).hexdigest(),
                'shared_bench_hash': hashlib.sha256(Path(bench.__file__).read_bytes()).hexdigest(),
                'timeout': args.timeout, 'retries': args.retries,
                'transport': 'explicit-proxy' if args.proxy else 'direct' if args.direct else 'system',
                'training': 'none; zero-shot', 'limit': args.limit,
                'dataset': manifest, 'config': {'model': entry['model'],
                    'params': entry.get('params', {}), 'name': entry.get('name', entry['model']),
                    'origin': entry.get('origin', ''), 'baseline': 'system-prompt-only',
                    'participant': getattr(args, 'participant', 'member-api'),
                    'run_group': getattr(args, 'run_group', 'baseline-v1'),
                    'response_contract': response_contract.VERSION,
                    'response_contract_hash': hashlib.sha256(Path(response_contract.__file__).read_bytes()).hexdigest(),
                    'reuse_from': str(Path(args.reuse_from).resolve()) if getattr(args, 'reuse_from', None) else None,
                    'dataset': name, 'limit': args.limit, 'quality_policy': quality_policy(args)},
                'versions': versions, 'python': platform.python_version()}
    signature = bench.digest(metadata)
    metadata['signature'] = signature
    folder.mkdir(parents=True, exist_ok=True)
    config_path = folder / 'run.json'
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding='utf-8'))
        if old['signature'] != signature:
            raise ValueError('Configuration/data/code changed; choose a new --out directory')
    else:
        bench.write(config_path, metadata)
        (folder / 'protocol.py').write_bytes(protocol)
        (folder / 'bench.py').write_bytes(Path(bench.__file__).read_bytes())
        (folder / 'response_contract.py').write_bytes(Path(response_contract.__file__).read_bytes())
        bench.write(folder / 'response_schema.json', response_contract.response_schema(args.assume_source if hasattr(args, 'assume_source') else 101))
        bench.jsonl(folder / 'test.jsonl', rows)
        bench.write(folder / 'manifest.json', manifest)
        (folder / 'prompt.txt').write_text(prompt, encoding='utf-8')
        selected_catalog = getattr(args, 'catalog', {}).get(entry['model'])
        if selected_catalog:
            bench.write(folder / 'model_catalog.json', selected_catalog)
        notebook = HERE / 'api_colab.ipynb'
        if notebook.is_file():
            (folder / 'notebook.ipynb').write_bytes(notebook.read_bytes())
    prediction_path = folder / 'predictions.jsonl'
    records = bench.read_predictions(prediction_path)
    if not records and getattr(args, 'reuse_from', None):
        source_folder = run_folder(args.reuse_from, name, entry)
        if (source_folder / 'predictions.jsonl').is_file():
            records, provenance = reuse_predictions(source_folder, rows, entry, prompt, args.base_url)
            bench.jsonl(prediction_path, records)
            bench.write(folder / 'reuse.json', provenance)
            print(f'Reused and revalidated {len(records)} saved responses; no repeat API calls', flush=True)
    if len(records) > len(rows) or any(r['row_hash'] != bench.digest(row) for row, r in zip(rows, records)):
        raise ValueError('Saved predictions do not match this test')
    stop = quality_stop(rows, records, quality_policy(args))
    with prediction_path.open('a', encoding='utf-8') as stream:
        for row in rows[len(records):]:
            if stop:
                break
            response, seconds, attempts = request_json(
                opener, args.base_url, 'chat/completions', api_key,
                payload_for(entry, prompt, row), args.timeout, args.retries)
            record = prediction(response, row, seconds, attempts)
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            stream.flush()
            records.append(record)
            if len(records) == 1 or len(records) % 20 == 0 or len(records) == len(rows):
                print(f"{name} | {entry['model']} | {len(records)}/{len(rows)}", flush=True)
            stop = quality_stop(rows, records, quality_policy(args))
    evaluated = rows[:len(records)]
    status = 'early_stopped' if stop else 'completed'
    evaluation = {'status': status, 'planned_rows': len(rows), 'evaluated_rows': len(evaluated),
                  'planned_test_hash': bench.digest(rows), 'evaluated_test_hash': bench.digest(evaluated),
                  'policy': quality_policy(args), 'stop_reason': stop}
    bench.write(folder / 'evaluation.json', evaluation)
    if stop:
        print(f"Early stop at {len(records)}/{len(rows)}: {json.dumps(stop)}", flush=True)
    scores = bench.metrics(evaluated, records)
    strict_records = [{**r, 'labels': r['strict_labels'], 'valid': r['strict_valid']} for r in records]
    strict_scores = bench.metrics(evaluated, strict_records)
    scores.update(dataset=name, model=entry['model'], test_hash=metadata['test_hash'],
                  legacy_strict_micro_f1=strict_scores['micro_f1'], legacy_strict_invalid_rate=strict_scores['invalid_rate'],
                  normalized_rate=sum(r['valid'] and not r['strict_valid'] for r in records) / len(records),
                  missing_finish_reason_rate=sum(r['finish_reason'] is None for r in records) / len(records),
                  reused_rows=json.loads((folder / 'reuse.json').read_text(encoding='utf-8'))['reused_rows'] if (folder / 'reuse.json').exists() else 0,
                  evaluation_status=status, early_stopped=bool(stop),
                  evaluated_rows=len(evaluated), planned_rows=len(rows),
                  prompt_hash=bench.digest(prompt), limited=args.limit is not None,
                  prompt_tokens=sum(r['usage'].get('prompt_tokens', 0) or 0 for r in records),
                  completion_tokens=sum(r['usage'].get('completion_tokens', 0) or 0 for r in records),
                  retried_rows=sum(r['attempts'] > 1 for r in records))
    # Cost units vary across gateways; do not pretend tokens are billed currency.
    bench.write(folder / 'metrics.json', scores)
    bench.jsonl(folder / 'errors.jsonl', [{'row': row, 'prediction': record}
               for row, record in zip(evaluated, records) if row['labels'] != record['labels'] or not record['strict_valid']])
    bench.write(folder / 'drafts.json', [
        {'row_id': row['row_id'], 'valid': record['valid'], 'drafts': [
            {'mcId': label, 'title': None, 'text': None} for label in record['labels']]}
        for row, record in zip(evaluated, records)])
    return scores


def export_result(folder, project, include_dataset=True, force=False):
    """Retryable ClearML export; a completed identical upload is not duplicated."""
    from publish_clearml import publish, evaluated_gold
    folder = Path(folder)
    metadata = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
    scores = json.loads((folder / 'metrics.json').read_text(encoding='utf-8'))
    rows, records = bench.read(folder / 'test.jsonl'), bench.read(folder / 'predictions.jsonl')
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    if (metadata['dataset'] != manifest or bench.digest(rows) != manifest['hashes']['test']
            or scores['rows'] != len(records)
            or any(record['row_hash'] != bench.digest(row) for row, record in zip(rows, records))):
        raise ValueError('Cannot upload incomplete or modified API result')
    evaluated = evaluated_gold(folder, rows, records, scores)
    expected_scores = json.loads(json.dumps(bench.metrics(evaluated, records)))
    if any(scores.get(key) != value for key, value in expected_scores.items()):
        raise ValueError('Saved metrics do not match API predictions')
    export_identity = {'metadata': metadata, 'predictions': records, 'scores': scores,
                       'project': project, 'include_dataset': include_dataset}
    if (folder / 'evaluation.json').is_file():
        export_identity['evaluation'] = json.loads((folder / 'evaluation.json').read_text(encoding='utf-8'))
    identity = bench.digest(export_identity)
    receipt = folder / 'clearml.json'
    if receipt.is_file() and not force:
        saved = json.loads(receipt.read_text(encoding='utf-8'))
        if saved.get('identity') == identity:
            print('Already uploaded to ClearML: ' + saved['url'])
            return saved
    result = publish(folder, folder, project, include_dataset)
    saved = {**result, 'identity': identity, 'project': project}
    bench.write(receipt, saved)
    return saved


def check_clearml():
    from clearml import Task
    # Read-only check: do not create a dummy task in the team's leaderboard.
    Task.get_projects()
    print('ClearML connection checked')


def validate_url(url, proxy=False):
    parts = urlsplit(url)
    if parts.scheme not in ('http', 'https') or not parts.netloc:
        raise ValueError('Expected an http(s) URL')
    if not proxy and (parts.username or parts.password or parts.query or parts.fragment):
        raise ValueError('Base URL must not contain credentials, query or fragment')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default=BASE_URL)
    parser.add_argument('--config', type=Path, default=HERE / 'api_models.json')
    parser.add_argument('--prompt', type=Path, help='Default: exact bench.PROMPT used by local LLMs')
    parser.add_argument('--dirty', type=Path, default=ROOT / 'data/raw/public-test-dataset.parquet')
    parser.add_argument('--skip-dirty', action='store_true', help='Evaluate only --clean; omit the default dirty dataset')
    parser.add_argument('--clean', type=Path, help='Explicit reviewed test; omitted until ready')
    parser.add_argument('--assume-source', type=int, choices=list(bench.LABELS), default=101)
    parser.add_argument('--out', type=Path, default=ROOT / 'research/api-runs/ru-cn-baseline-v1')
    parser.add_argument('--reuse-from', type=Path, help='Previous model run root; revalidate matching raw responses without repeat calls')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--timeout', type=float, default=60)
    parser.add_argument('--retries', type=int, default=2)
    parser.add_argument('--early-stop-min-rows', type=int, default=30, help='Warm-up before quality checks; 0 disables early stop')
    parser.add_argument('--target-f1', type=float, default=.93, help='Stop if even perfect remaining answers cannot reach this F1')
    parser.add_argument('--screen-f1', type=float, help='Budget heuristic: require this F1 on the warm-up prefix; may reject a model that could recover')
    parser.add_argument('--max-invalid-rate', type=float, default=None, help=argparse.SUPPRESS)
    transport = parser.add_mutually_exclusive_group()
    transport.add_argument('--proxy', default=os.environ.get('CAILA_PROXY'), help='http(s) proxy URL; never saved')
    transport.add_argument('--direct', action='store_true', help='Bypass system HTTP proxies, not VPN tunnels')
    parser.add_argument('--list-models', action='store_true')
    parser.add_argument('--filter', default='', help='Substring filter for --list-models')
    parser.add_argument('--dry-run', action='store_true', help='Validate inputs without requests or output files')
    parser.add_argument('--check', action='store_true', help='Check catalog and ClearML access without inference or tasks')
    parser.add_argument('--clearml-project', default='avito-microcategories')
    parser.add_argument('--participant', default='member-api')
    parser.add_argument('--run-group', default='baseline-v1')
    parser.add_argument('--no-upload', action='store_true', help='Explicitly disable automatic ClearML export')
    parser.add_argument('--no-dataset-upload', action='store_true', help='Omit separate test.jsonl and description column')
    parser.add_argument('--export-only', action='store_true', help='Retry ClearML export of completed runs without Caila calls')
    parser.add_argument('--force-upload', action='store_true', help='Create a new ClearML task even if receipt exists')
    args = parser.parse_args()
    if args.direct:
        args.proxy = None
    if args.timeout <= 0 or args.retries < 0 or (args.limit is not None and args.limit <= 0):
        parser.error('Timeout and limit must be positive; retries must be nonnegative')
    if (args.early_stop_min_rows < 0 or not 0 <= args.target_f1 <= 1
            or (args.screen_f1 is not None and not 0 <= args.screen_f1 <= 1)):
        parser.error('Quality thresholds must be in [0, 1]; warm-up rows must be nonnegative')
    if args.max_invalid_rate is not None:
        print('Note: --max-invalid-rate is obsolete; format errors are diagnostic and do not independently stop a model')
    validate_url(args.base_url)
    if args.proxy:
        validate_url(args.proxy, proxy=True)
    opener = api_opener(args.proxy, args.direct)
    api_key = os.environ.get('CAILA_API_KEY', '')
    if args.list_models:
        catalog, _, _ = request_json(opener, args.base_url, 'models', api_key, timeout=args.timeout, retries=args.retries)
        for model in catalog.get('data', []):
            if args.filter.casefold() in model['id'].casefold():
                print(model['id'])
        return
    models = load_models(args.config)
    prompt = baseline_prompt(args.prompt)
    if not prompt.strip():
        raise ValueError('Prompt is empty')
    if args.export_only:
        if args.no_upload or args.dry_run:
            parser.error('--export-only cannot be combined with --no-upload or --dry-run')
        folders = sorted(path.parent for path in args.out.glob('*/*/metrics.json'))
        if not folders:
            raise ValueError('No completed API results under --out')
        check_clearml()
        for folder in folders:
            export_result(folder, args.clearml_project, not args.no_dataset_upload, args.force_upload)
        return
    datasets = []
    if args.skip_dirty and args.clean is None:
        parser.error('--skip-dirty requires an explicit --clean dataset')
    for name, path in [('dirty', None if args.skip_dirty else args.dirty), ('clean', args.clean)]:
        if path is None:
            print(f'{name}: not supplied; skipped')
            continue
        rows = load_test(path, args.assume_source)
        rows = rows[:args.limit] if args.limit else rows
        datasets.append((name, path, rows))
        print(f'{name}: {len(rows)} rows; hash={bench.digest(rows)[:12]}')
    if args.dry_run:
        print(f'{len(models)} models; {sum(len(rows) for _, _, rows in datasets) * len(models)} planned requests; zero-shot')
        print('Prompt: bench.PROMPT (local baseline)' if args.prompt is None else 'Prompt: custom file')
        return
    if not api_key:
        raise ValueError('Set CAILA_API_KEY in your environment; do not store the key in files')
    # Fail before paid requests if a proposed ID is absent from this account's catalog.
    catalog, _, _ = request_json(opener, args.base_url, 'models', api_key, timeout=args.timeout, retries=args.retries)
    args.catalog = {model['id']: model for model in catalog.get('data', [])}
    available = set(args.catalog)
    missing = [model['model'] for model in models if model['model'] not in available]
    if missing:
        raise ValueError('Model IDs absent from catalog; use --list-models and update config: ' + ', '.join(missing))
    if not args.no_upload:
        check_clearml()
    if args.check:
        print(f'Ready: {len(models)} catalog model IDs verified; no inference requests sent')
        return
    args.out.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, path, rows in datasets:
        for model in models:
            scores = evaluate(args, name, path, rows, model, prompt, opener, api_key)
            summary.append({key: value for key, value in scores.items() if not isinstance(value, dict)})
            # Keep completed work visible even if the next model fails.
            bench.write(args.out / 'summary.json', summary)
            with (args.out / 'summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
                writer.writeheader()
                writer.writerows(summary)
            print(f"micro-F1={scores['micro_f1']:.4f}; precision={scores['precision']:.4f}; invalid={scores['invalid_rate']:.3f}")
            if not args.no_upload:
                export_result(run_folder(args.out, name, model), args.clearml_project,
                              not args.no_dataset_upload, args.force_upload)


if __name__ == '__main__':
    try:
        main()
    except (APIError, ValueError, OSError) as exc:
        # OSError messages can contain proxy credentials; only expose their class.
        message = type(exc).__name__ if isinstance(exc, OSError) else str(exc)
        raise SystemExit(message) from None
