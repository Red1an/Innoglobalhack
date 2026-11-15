"""Compare both train versions on the same, deduplicated guide_v2 evaluation sets."""

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from build_guide_v2 import ROOT, TEST_OUT, TRAIN_OUT, TRAIN_PARENT, json_rows, normalized

sys.path.insert(0, str(ROOT / 'scripts'))
from eval_stress_suite import build_leak_free_train, predict_char_lr, score_slice
from bench import canonical, key
sys.path.pop(0)

PUBLIC_PATH = ROOT / 'data/raw/public-test-dataset.parquet'
THRESHOLD = 0.5
VERSION_PATHS = {'guide_v1': TRAIN_PARENT / 'train.parquet', 'guide_v2': TRAIN_OUT / 'train.parquet'}
OUTPUT = ROOT / 'output/guide_v2_comparison.json'


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def to_eval(row, slice_name):
    return {**canonical(row, 101), 'row_id': row['row_id'], 'slice': slice_name}


def evaluation_groups(test_path):
    gold = [to_eval(row, 'gold') for row in pq.read_table(test_path / 'test_gold.parquet').to_pylist()]
    hard = [to_eval(row, 'hard') for row in json_rows(test_path / 'hard.jsonl')]
    stress = json_rows(test_path / 'stress_suite.jsonl')
    all_rows = gold + hard + stress

    groups = defaultdict(list)
    seen = {}
    for index, row in enumerate(all_rows):
        groups[row['slice']].append(index)
        if index >= len(gold) + len(hard):
            groups['stress_all'].append(index)
        text_key = (row['mcId'], normalized(row['description']))
        if text_key in seen:
            if row['labels'] != seen[text_key]['labels']:
                raise ValueError(f'Conflicting evaluated duplicates: {row["row_id"]}')
            continue
        groups['unique_all'].append(index)
        seen[text_key] = row
    return all_rows, groups, stress


def compare(test_path=TEST_OUT, version_paths=VERSION_PATHS):
    evaluation, groups, stress = evaluation_groups(test_path)
    public = pq.read_table(PUBLIC_PATH, columns=['description']).to_pylist()
    extra_reserved = [{'description': row['description'], 'origin_split': 'external'}
                      for row in public] + [
        {'description': evaluation[i]['description'], 'origin_split': 'external'} for i in groups['hard']
    ]
    reserved = {normalized(row['description']) for row in evaluation + public}

    fit_texts = None
    result = {
        'threshold': THRESHOLD,
        'test_sha256': {p: fingerprint(test_path / p) for p in ('test_gold.parquet', 'hard.jsonl', 'stress_suite.jsonl')},
        'public_sha256': fingerprint(PUBLIC_PATH),
        'test_rows': {name: len(ids) for name, ids in groups.items()},
        'duplicate_evaluation_rows': len(evaluation) - len(groups['unique_all']),
        'runs': {},
    }

    for version, path in version_paths.items():
        source_rows = pq.read_table(path).to_pylist()
        by_description = {key(row['description']): row for row in source_rows}
        source_keys = {normalized(row['description']) for row in source_rows}
        fit = build_leak_free_train(stress + extra_reserved, path, test_path / 'test_gold.parquet')
        fit_keys = [key(row['description']) for row in fit]
        fit_ids = [by_description[text]['item_id'] for text in fit_keys]
        if fit_texts is not None and (fit_ids, fit_keys) != fit_texts:
            raise ValueError('Train versions have different fit rows')
        fit_texts = (fit_ids, fit_keys)
        if reserved & {normalized(row['description']) for row in fit}:
            raise ValueError('Normalized overlap between fit and held-out rows')
        # Stress dev examples are the only intended overlaps at the dataset level.
        overlap_ids = [row['row_id'] for row in stress if normalized(row['description']) in source_keys]
        expected_ids = [row['row_id'] for row in stress if row['origin_split'] == 'dev']
        if overlap_ids != expected_ids or source_keys & {normalized(row['description']) for row in extra_reserved}:
            raise ValueError('Unexpected dataset-level overlap')

        predictions = predict_char_lr(fit, evaluation, threshold=THRESHOLD)
        result['runs'][version] = {
            'train_sha256': fingerprint(path),
            'fit_rows': len(fit),
            'fit_item_ids_sha256': hashlib.sha256(json.dumps(fit_ids).encode()).hexdigest(),
            'train_full_stress_overlaps': overlap_ids,
            'fit_by_origin': dict(Counter(by_description[text]['origin'] for text in fit_keys)),
            'fit_eval_overlap': 0,
            'metrics': {
                name: score_slice([evaluation[i]['labels'] for i in indices],
                                  [predictions[i] for i in indices],
                                  [evaluation[i]['mcId'] for i in indices])
                for name, indices in groups.items()
            },
        }

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = compare()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(args.out)
    print(args.out)
