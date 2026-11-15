"""Build reproducible guide_v2 datasets from versioned snapshots and reviewed decisions."""

import argparse
import hashlib
import json
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from build_guide_v1 import artifact, raw_rows, train_labels, write_version
from build_train_dedup_v1 import duplicate_key, replace_visual_spaces, without_stock
from build_train_relabel_v1 import LABELS, labels_for, read_csv, validated, write_csv
from build_train_text_v1 import ROOT
from train_text_cleaning import clean_text

REVIEW_DIR = Path(__file__).resolve().parent
TRAIN_PARENT = ROOT / 'data/processed/train_guide_v1'
TRAIN_OUT = ROOT / 'data/processed/train_guide_v2'
TEST_PARENT = ROOT / 'data/test/test_guide_v1'
TEST_OUT = ROOT / 'data/test/test_guide_v2'
STRESS_SOURCE = ROOT / 'data/test/stress_suite_v1'
SYNTH_REVIEW = ROOT / 'data/processed/train_synth_v1/synth_review.jsonl'
CATEGORIES = (101, *LABELS)
REVIEW_FIELDS = ['dataset', 'row_id', 'mcId', 'old', 'new', 'quote', 'reason']
CHANGE_FIELDS = REVIEW_FIELDS + ['old_labels', 'new_labels', 'source_item_id', 'source_line',
                                 'synth_task_id', 'public_row_ids', 'source_mcId']


def json_file(path):
    return json.loads(path.read_text(encoding='utf-8'))


def json_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def iso_json(value):
    return json.dumps(value, ensure_ascii=False)


def normalized(text):
    return duplicate_key(without_stock(replace_visual_spaces(clean_text(text))))


def load_review(path, datasets):
    edits = defaultdict(lambda: defaultdict(list))
    seen = set()
    for review in read_csv(path):
        dataset = review['dataset']
        row_id = review['row_id']
        cat = int(review['mcId'])
        key = (dataset, row_id, cat)
        if (dataset not in datasets or cat not in CATEGORIES or key in seen
                or review['old'] not in {'0', '1'} or review['new'] not in {'', '0', '1'}
                or review['new'] == review['old'] or not review['quote'] or not review['reason']):
            raise ValueError(f'Invalid review: {review}')
        seen.add(key)
        edits[dataset][row_id].append(review)
    return edits


def row_labels(row):
    if 'found_microcategories' in row:
        return labels_for(row)
    if 'labels' in row:
        return row['labels']
    return [cat for cat in CATEGORIES if row.get(f'y_{cat}')]


def change_metadata(row):
    return {
        'source_item_id': row.get('source_item_id'),
        'source_line': row.get('source_line'),
        'synth_task_id': row.get('synth_task_id'),
        'public_row_ids': iso_json(row['row_ids']) if 'row_ids' in row else None,
        'source_mcId': row.get('mcId', 101),
    }


def apply_review(row, reviews, changes, uncertain):
    old_labels = set(row_labels(row))
    labels = old_labels.copy()
    source = row.get('mcId', 101)
    if len(old_labels) != len(row_labels(row)) or old_labels - set(CATEGORIES) or source in old_labels:
        raise ValueError(f'Invalid source labels: {change_metadata(row)}')
    for review in reviews:
        cat = int(review['mcId'])
        if (cat == source or int(cat in old_labels) != int(review['old'])
                or review['quote'] not in row['description']):
            raise ValueError(f'Review disagrees with source: {review}')
        if review['new']:
            labels.symmetric_difference_update({cat})
    for review in reviews:
        record = {**review, **change_metadata(row), 'old_labels': iso_json(sorted(old_labels)),
                  'new_labels': iso_json(sorted(labels))}
        (changes if review['new'] else uncertain).append(record)
    return labels


def review_rows(rows, reviews, row_key, setter, changes, uncertain):
    pending = reviews.copy()
    seen = set()
    for row in rows:
        row_id = str(row[row_key])
        if row_id in seen:
            raise ValueError(f'Duplicate row ID: {row_id}')
        seen.add(row_id)
        labels = apply_review(row, pending.pop(row_id, []), changes, uncertain)
        setter(row, labels)
    if pending:
        raise ValueError(f'Orphan reviews: {list(pending)}')


def set_gold(row, labels):
    row.update({f'y_{cat}': int(cat in labels) for cat in CATEGORIES})


def set_list(row, labels):
    row['labels'] = sorted(labels)


def write_audit(stage, out, changes, uncertain):
    files = {}
    for name, rows in [('label_changes', changes), ('ambiguous', uncertain)]:
        path = stage / f'{name}.csv'
        write_csv(path, CHANGE_FIELDS, rows)
        files[name] = artifact(out / path.name, len(rows), path)
    return files


def build_tests(stage, out, review_path):
    previous = json_file(TEST_PARENT / 'manifest.json')
    gold_path = validated(previous['datasets']['gold'])
    hard_path = validated(previous['datasets']['hard'])
    public = pq.read_table(validated(previous['sources']['public']))
    stress_path = STRESS_SOURCE / 'stress_suite_v1.jsonl'
    stress_parquet = STRESS_SOURCE / 'stress_suite_v1.parquet'
    stress_manifest = json_file(STRESS_SOURCE / 'manifest.json')
    gold_table = pq.read_table(gold_path)
    gold, hard, stress = gold_table.to_pylist(), json_rows(hard_path), json_rows(stress_path)
    stress_info = artifact(stress_path, len(stress))
    stress_digest = hashlib.sha256(json.dumps(stress, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if (stress_digest != stress_manifest['sha256_jsonl']
            or stress != pq.read_table(stress_parquet).to_pylist()
            or Counter(row['slice'] for row in stress) != stress_manifest['slices']):
        raise ValueError('Fresh stress-suite snapshots disagree')
    if any(row['row_id'] not in row['row_ids'] or any(i < 0 or i >= len(public) for i in row['row_ids'])
           for row in gold):
        raise ValueError('Gold row IDs disagree with the public source')

    edits = load_review(review_path, {'gold', 'hard', 'stress'})
    changes, uncertain = [], []
    review_rows(gold, edits['gold'], 'row_id', set_gold, changes, uncertain)
    review_rows(hard, edits['hard'], 'row_id', set_list, changes, uncertain)

    gold_by_id = {str(row['row_id']): row for row in gold}
    stress_edits = edits['stress'].copy()
    inherited = defaultdict(list)
    for edit in changes + uncertain:
        if edit['dataset'] == 'gold':
            inherited[edit['row_id']].append(edit)

    for row in stress:
        if row['origin_split'] != 'test_gold':
            continue
        match = re.search(r'\brow_id=(\d+)', row['note'])
        if not match:
            raise ValueError(f'Missing reference to gold: {row["row_id"]}')
        gold_id = match[1]
        reference = gold_by_id[gold_id]
        if (row['description'].split() != reference['description'].split()
                or row['row_id'] in stress_edits):
            raise ValueError(f'Unexpected stress copy of gold: {row["row_id"]}')
        old_labels = set(row['labels'])
        differences = old_labels ^ set(row_labels(reference))
        row['row_ids'] = reference['row_ids']
        for source_edit in inherited[gold_id]:
            cat = int(source_edit['mcId'])
            if cat in differences or not source_edit['new']:
                stress_edits.setdefault(row['row_id'], []).append({
                    **{field: source_edit[field] for field in REVIEW_FIELDS},
                    'dataset': 'stress', 'row_id': row['row_id'], 'old': str(int(cat in old_labels)),
                    'reason': f'Копия gold row_id={gold_id}. {source_edit["reason"]}',
                })
        if differences - {int(edit['mcId']) for edit in stress_edits.get(row['row_id'], []) if edit['new']}:
            raise ValueError(f'Unexplained gold and stress disagreement: {row["row_id"]}')

    review_rows(stress, stress_edits, 'row_id', set_list, changes, uncertain)
    for row in stress:
        row.pop('labels_strict')
        row.pop('labels_seo')
        set_gold(row, set(row['labels']))

    stress_table = pa.Table.from_pylist(stress)
    gold_schema = gold_table.schema.remove_metadata().append(pa.field('y_101', pa.int64()))
    datasets = {
        'gold': write_version(stage, out, 'test_gold.parquet', gold, gold_schema),
        'hard': write_version(stage, out, 'hard.jsonl', hard),
        'stress': write_version(stage, out, 'stress_suite.parquet', stress, stress_table.schema),
        'stress_jsonl': write_version(stage, out, 'stress_suite.jsonl', stress),
    }
    slices = {}
    for name in stress_manifest['slices']:
        rows = [row for row in stress if row['slice'] == name]
        slices[name] = {
            'parquet': write_version(stage, out, f'{name}.parquet', rows, stress_table.schema),
            'jsonl': write_version(stage, out, f'{name}.jsonl', rows),
        }
    return {
        'sources': {'gold': previous['datasets']['gold'], 'hard': previous['datasets']['hard'],
                    'stress': stress_info, 'stress_parquet': artifact(stress_parquet, len(stress)),
                    'public': previous['sources']['public']},
        'datasets': datasets,
        'slices': slices,
        **write_audit(stage, out, changes, uncertain),
        'changes_by_dataset': dict(Counter(edit['dataset'] for edit in changes)),
    }


def build_train(stage, out, review_path):
    parent = json_file(TRAIN_PARENT / 'manifest.json')
    table = pq.read_table(validated(parent['dataset']))
    rows = table.to_pylist()
    if len(rows) != parent['dataset']['records']:
        raise ValueError('Train row count disagrees with the previous manifest')

    raw_info = parent['sources']['raw_train']
    raw = raw_rows(raw_info, rows)
    all_synth_reviews = json_rows(SYNTH_REVIEW)
    accepted = {review['task_id']: review for review in all_synth_reviews if review['accepted']}
    for row in rows:
        if row['source_item_id'] != raw[row['source_line']]['item_id']:
            raise ValueError(f'Invalid raw source for {row["item_id"]}')
        if row['origin'] == 'real' and row['item_id'] == row['source_item_id'] and row['synth_task_id'] is None:
            continue
        review = accepted.pop(row['synth_task_id'], None)
        if (row['origin'] != 'synthetic' or review is None or not row['synth_checked']
                or row['source_item_id'] != review['source_item_id'] or row['description'] != review['text']):
            raise ValueError(f'Invalid synthetic source for {row["item_id"]}')
    if accepted:
        raise ValueError('Accepted synthetic examples are missing from train')

    edits = load_review(review_path, {'train'})
    changes, uncertain = [], []
    before = {row['item_id']: sorted(row_labels(row)) for row in rows}
    review_rows(rows, edits['train'], 'item_id', train_labels, changes, uncertain)
    after = {row['item_id']: row_labels(row) for row in rows}
    train_keys = {normalized(row['description']) for row in rows}
    if '' in train_keys or len(train_keys) != len(rows):
        raise ValueError('Train has normalized duplicates or an empty description')

    return {
        'sources': {'train': parent['dataset'], 'raw_train': raw_info,
                    'synthetic_review': artifact(SYNTH_REVIEW, len(all_synth_reviews))},
        'dataset': write_version(stage, out, 'train.parquet', rows, table.schema),
        **write_audit(stage, out, changes, uncertain),
        'records_by_origin': dict(Counter(row['origin'] for row in rows)),
        'labels_before': dict(Counter(cat for labels in before.values() for cat in labels)),
        'labels_after': dict(Counter(cat for labels in after.values() for cat in labels)),
        'normalized_train_duplicates': len(rows) - len(train_keys),
    }


def build(part, out, review_path):
    out = out.resolve()
    if out.exists():
        raise FileExistsError(f'Choose a new output directory: {out}')
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        stage = Path(tmp) / 'version'
        stage.mkdir()
        data = BUILDERS[part](stage, out, review_path)
        manifest = {'version': out.name, 'rules': artifact(REVIEW_DIR / 'guide_v2_rules.md', None),
                    'review': artifact(review_path, len(read_csv(review_path))),
                    'builder': artifact(Path(__file__), None), 'pyarrow_version': pa.__version__, **data}
        (stage / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
                                            encoding='utf-8')
        stage.rename(out)
    print(out)


BUILDERS = {'test': build_tests, 'train': build_train}
DESTINATIONS = {'test': TEST_OUT, 'train': TRAIN_OUT}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('part', choices=BUILDERS)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--review', type=Path)
    args = parser.parse_args()
    build(args.part, args.out or DESTINATIONS[args.part],
          args.review or REVIEW_DIR / f'guide_v2_{args.part}_review.csv')
