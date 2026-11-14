"""Собирает train_relabel_v2_synth_v1 из актуальных реальных и проверенных синтетических объявлений."""

import argparse
import json
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from build_train_dedup_v1 import duplicate_key
from build_train_relabel_v1 import LABELS, labels_for, validated, write_csv
from build_train_text_v1 import ROOT, sha256

sys.path.insert(0, str(ROOT / 'research/experiments'))
from bench import gold_overlap_key, key
sys.path.pop(0)

REAL = ROOT / 'data/processed/train_relabel_v2'
SYNTH = ROOT / 'data/processed/train_synth_v1'
DESTINATION = ROOT / 'data/processed/train_relabel_v2_synth_v1'
TESTS = {
    'test_gold': ROOT / 'data/test/test_gold.parquet',
    'public_test': ROOT / 'data/raw/public-test-dataset.parquet',
}
DATASET_FILENAME = 'train_relabel_v2_synth_v1.parquet'
ALL_FIVE_CAP = 800
SEED = 42


def artifact(path, records, staged_path=None):
    display_path = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    return {'path': display_path.as_posix(), 'sha256': sha256(staged_path or path), 'records': records}


def label_counts(rows):
    counts = Counter(label for row in rows for label in labels_for(row))
    return {str(label): counts[label] for label in LABELS}


def build(destination):
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f'Версия уже существует: {destination}')

    real_manifest = json.loads((REAL / 'manifest.json').read_text(encoding='utf-8'))
    synth_manifest = json.loads((SYNTH / 'manifest.json').read_text(encoding='utf-8'))
    synth_path = SYNTH / 'train.parquet'
    review_path = SYNTH / 'synth_review.jsonl'
    real = pq.read_table(validated(real_manifest['dataset']))
    synth_source = pq.read_table(synth_path)
    if len(real) != real_manifest['dataset']['records'] or len(synth_source) != synth_manifest['records']:
        raise ValueError('Число строк в источниках расходится с манифестами')
    real_rows = real.to_pylist()
    real_lines = {row['item_id']: row['source_line'] for row in real_rows}
    with review_path.open(encoding='utf-8') as stream:
        reviews = [json.loads(line) for line in stream]
    accepted = {row['task_id']: row for row in reviews if row['accepted']}
    synth_rows = [row for row in synth_source.to_pylist() if row['origin'] == 'synthetic']
    if len(synth_rows) != len(accepted) or len(synth_rows) != synth_manifest['synthetic']:
        raise ValueError('Число синтетических строк расходится с результатами проверки')
    for row in synth_rows:
        review = accepted.pop(row['synth_task_id'])
        source_id = review['source_item_id']
        if (row['source_line'] != -source_id or row['synth_checked'] is not True
                or review['checked'] is not True or row['description'] != review['text']
                or labels_for(row) != review['target_labels'] or labels_for(row) != review['check_labels']):
            raise ValueError(f'Расхождение с результатами проверки синтеза: {row["synth_task_id"]}')
        row['source_item_id'] = source_id
        row['source_line'] = real_lines[source_id]

    normalize = gold_overlap_key()
    test_inputs = {}
    test_keys = {}
    for name, path in TESTS.items():
        texts = pq.read_table(path, columns=['description']).column('description').to_pylist()
        test_inputs[name] = artifact(path, len(texts))
        test_keys[name] = ({key(text) for text in texts}, {normalize(text) for text in texts} - {''})

    real_kept, excluded = [], []
    overlap_ids = set()
    for row in real_rows:
        text = row['description']
        text_key, normalized = key(text), normalize(text)
        matches = [name for name, (exact, cleaned) in test_keys.items()
                   if text_key in exact or normalized in cleaned]
        if matches:
            overlap_ids.add(row['item_id'])
            excluded.append({'item_id': row['item_id'], 'source_line': row['source_line'],
                             'reason': 'test_overlap', 'matching_tests': ','.join(matches)})
        else:
            real_kept.append(row)
    for row in synth_rows:
        text = row['description']
        if row['source_item_id'] in overlap_ids or any(
            key(text) in exact or normalize(text) in cleaned for exact, cleaned in test_keys.values()
        ):
            raise ValueError(f'Синтез пересекается с тестом: {row["synth_task_id"]}')

    all_five = [row['item_id'] for row in real_kept if len(set(labels_for(row))) == len(LABELS)]
    dropped = set(random.Random(SEED).sample(all_five, max(0, len(all_five) - ALL_FIVE_CAP)))
    balanced = []
    for row in real_kept:
        if row['item_id'] in dropped:
            excluded.append({'item_id': row['item_id'], 'source_line': row['source_line'],
                             'reason': 'all_five_cap', 'matching_tests': ''})
        else:
            balanced.append({**row, 'origin': 'real', 'source_item_id': row['item_id']})
    rows = balanced + synth_rows
    if len({row['item_id'] for row in rows}) != len(rows):
        raise ValueError('Повторяющиеся item_id')
    description_keys = {duplicate_key(row['description']) for row in rows}
    if '' in description_keys or len(description_keys) != len(rows):
        raise ValueError('Пустое или повторяющееся описание')
    for row in rows:
        labels = labels_for(row)
        if len(set(labels)) != len(labels) or set(labels) - LABELS.keys():
            raise ValueError(f'Недопустимые метки: {row["item_id"]}')

    schema = pa.schema([
        real.schema.field('item_id'), pa.field('title', pa.string()), real.schema.field('description'),
        real.schema.field('found_microcategories'), real.schema.field('source_line'),
        pa.field('origin', pa.string(), nullable=False), pa.field('source_item_id', pa.int64(), nullable=False),
        pa.field('synth_task_id', pa.string()), pa.field('synth_checked', pa.bool_()),
    ])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'.{DESTINATION.name}-', dir=destination.parent) as temporary:
        stage = Path(temporary) / DESTINATION.name
        stage.mkdir()
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), stage / DATASET_FILENAME, compression='zstd')
        write_csv(stage / 'excluded_real.csv', ['item_id', 'source_line', 'reason', 'matching_tests'], excluded)
        manifest = {
            'version': DESTINATION.name,
            'sources': {
                'real': real_manifest['dataset'],
                'synthetic': artifact(synth_path, len(synth_source)),
                'synth_review': artifact(review_path, len(reviews)),
                'raw_real': real_manifest['raw_source'],
                **test_inputs,
            },
            'dataset': artifact(destination / DATASET_FILENAME, len(rows), stage / DATASET_FILENAME),
            'excluded_real': artifact(destination / 'excluded_real.csv', len(excluded), stage / 'excluded_real.csv'),
            'records_by_origin': {'real': len(balanced), 'synthetic': len(synth_rows)},
            'labels_by_origin': {'real': label_counts(balanced), 'synthetic': label_counts(synth_rows),
                                 'total': label_counts(rows)},
            'transformations': {
                'test_overlap_rows_removed': len(overlap_ids),
                'all_five': {'before_cap': len(all_five), 'after_cap': len(all_five) - len(dropped),
                             'limit': ALL_FIVE_CAP, 'seed': SEED},
                'synthetic_filter': 'origin=synthetic, совпадение с результатами слепой проверки',
            },
            'lineage': {
                'source_item_id': 'item_id реального исходного объявления',
                'source_line': 'номер строки этого объявления в исходном JSONL, начиная с единицы',
                'synth_task_id': 'идентификатор задания синтеза',
            },
            'build_sha256': sha256(Path(__file__)),
            'pyarrow_version': pa.__version__,
        }
        (stage / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
                                             encoding='utf-8')
        stage.rename(destination)
    print(f'{len(rows)} строк: {len(balanced)} реальных, {len(synth_rows)} синтетических.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DESTINATION)
    build(parser.parse_args().out)
