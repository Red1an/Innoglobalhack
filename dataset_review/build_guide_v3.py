"""Build the hackathon train and tests from frozen guide_v2 snapshots and audited edits."""

import argparse
import tempfile
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from build_guide_v2 import (
    REVIEW_DIR, REVIEW_FIELDS, ROOT, TEST_OUT as TEST_PARENT, TRAIN_OUT as TRAIN_PARENT,
    artifact, iso_json, json_file, json_rows, load_review, normalized, read_csv,
    review_rows, row_labels, set_gold, set_list, train_labels, validated, write_audit, write_version,
)

TRAIN_OUT = ROOT / 'data/processed/train_guide_v3'
TEST_OUT = ROOT / 'data/test/test_guide_v3'


def read_source(info):
    path = validated(info)
    table = pq.read_table(path) if path.suffix == '.parquet' else None
    rows = table.to_pylist() if table is not None else json_rows(path)
    if len(rows) != info['records']:
        raise ValueError(f'Unexpected record count: {path}')
    return rows, table.schema if table is not None else None


def with_previous_uncertainty(edits, parent):
    for item in read_csv(validated(parent['ambiguous'])):
        reviews = edits[item['dataset']].setdefault(item['row_id'], [])
        if all(edit['mcId'] != item['mcId'] for edit in reviews):
            reviews.append({field: item[field] for field in REVIEW_FIELDS})


def counts(rows):
    return dict(Counter(cat for row in rows for cat in row_labels(row)))


def build_train(stage, out, parent, edits):
    rows, schema = read_source(parent['dataset'])
    before = counts(rows)
    changes, uncertain = [], []
    review_rows(rows, edits['train'], 'item_id', train_labels, changes, uncertain)
    return {
        'sources': {'train': parent['dataset'], **{
            key: parent['sources'][key] for key in ('raw_train', 'synthetic_review')
        }},
        'dataset': write_version(stage, out, 'train.parquet', rows, schema),
        **write_audit(stage, out, changes, uncertain),
        'records_by_origin': dict(Counter(row['origin'] for row in rows)),
        'labels_before': before,
        'labels_after': counts(rows),
    }


def build_tests(stage, out, parent, edits):
    source = parent['datasets']
    gold, gold_schema = read_source(source['gold'])
    hard, _ = read_source(source['hard'])
    stress, stress_schema = read_source(source['stress'])
    stress_jsonl, _ = read_source(source['stress_jsonl'])
    if any({k: v for k, v in row.items() if k in other} != other
           for row, other in zip(stress, stress_jsonl, strict=True)):
        raise ValueError('Stress JSONL and parquet disagree')
    changes, uncertain = [], []
    review_rows(gold, edits['gold'], 'row_id', set_gold, changes, uncertain)
    review_rows(hard, edits['hard'], 'row_id', set_list, changes, uncertain)
    review_rows(stress, edits['stress'], 'row_id', set_list, changes, uncertain)

    seen = {}
    for row in gold + hard + stress:
        source_cat = row.get('mcId', 101)
        key = (source_cat, normalized(row['description']))
        labels = sorted(row_labels(row))
        if seen.setdefault(key, labels) != labels:
            raise ValueError(f'Conflicting evaluation duplicates: {row["row_id"]}')

    for row in stress:
        set_gold(row, set(row['labels']))
        if row['row_ids'] is None:
            del row['row_ids']

    datasets = {
        'gold': write_version(stage, out, 'test_gold.parquet', gold, gold_schema),
        'hard': write_version(stage, out, 'hard.jsonl', hard),
        'stress': write_version(stage, out, 'stress_suite.parquet', stress, stress_schema),
        'stress_jsonl': write_version(stage, out, 'stress_suite.jsonl', stress),
    }
    slices = {}
    for name in parent['slices']:
        rows = [row for row in stress if row['slice'] == name]
        slices[name] = {
            'parquet': write_version(stage, out, f'{name}.parquet', rows, stress_schema),
            'jsonl': write_version(stage, out, f'{name}.jsonl', rows),
        }
    return {
        'sources': {**source, 'public': parent['sources']['public']},
        'datasets': datasets,
        'slices': slices,
        **write_audit(stage, out, changes, uncertain),
        'changes_by_dataset': dict(Counter(edit['dataset'] for edit in changes)),
    }


PARTS = {
    'train': (TRAIN_PARENT, TRAIN_OUT, build_train, {'train'}),
    'test': (TEST_PARENT, TEST_OUT, build_tests, {'gold', 'hard', 'stress'}),
}


def build(part, out=None):
    parent_dir, default_out, builder, datasets = PARTS[part]
    out = (out or default_out).resolve()
    if out.exists():
        raise FileExistsError(f'Choose a new output directory: {out}')
    parent = json_file(parent_dir / 'manifest.json')
    review_path = REVIEW_DIR / f'guide_v3_{part}_review.csv'
    edits = load_review(review_path, datasets)
    with_previous_uncertainty(edits, parent)

    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        stage = Path(tmp) / 'version'
        stage.mkdir()
        data = builder(stage, out, parent, edits)
        manifest = {
            'version': out.name, 'rules': artifact(REVIEW_DIR / 'guide_v3_rules.md', None),
            'review': artifact(review_path, len(read_csv(review_path))),
            'builder': artifact(Path(__file__), None), 'pyarrow_version': pa.__version__, **data,
        }
        (stage / 'manifest.json').write_text(iso_json(manifest) + '\n', encoding='utf-8')
        stage.rename(out)
    print(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('part', choices=PARTS)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    build(args.part, args.out)
