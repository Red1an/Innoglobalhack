"""Собирает версии train и внутренних тестов с проверенными правками по справочнику."""

import argparse
import json
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from build_train_relabel_v1 import LABELS, labels_for, read_csv, validated, write_csv
from build_train_text_v1 import ROOT, sha256

sys.path.insert(0, str(ROOT / 'research/experiments'))
from export_hard_test import load_hard_test
sys.path.pop(0)

PARENT_TRAIN = ROOT / 'data/processed/train_relabel_v2_synth_v1'
PARENT_TEST = ROOT / 'data/test'
TRAIN_OUT = ROOT / 'data/processed/train_guide_v1'
TEST_OUT = PARENT_TEST / 'test_guide_v1'
GUIDE = ROOT / 'innoglobalhack_avito/microcategories-guide.md'
REVIEW = Path(__file__).with_name('guide_v1_reviewed.csv')
HARD_SOURCE = ROOT / 'data/hard/hard_v0.xlsx'


def artifact(path, records, staged_path=None):
    path = path.resolve()
    return {'path': str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
            'sha256': sha256(staged_path or path), 'records': records}


def load_reviews():
    groups = defaultdict(dict)
    seen = set()
    for edit in read_csv(REVIEW):
        dataset, row_id = edit['dataset'], edit['row_id']
        category, old, new = (int(edit[field]) for field in ('mcId', 'old', 'new'))
        key = (dataset, row_id, category)
        if (dataset not in {'train', 'gold', 'stress', 'hard'} or key in seen or category not in LABELS
                or {old, new} != {0, 1} or not edit['quote'] or not edit['reason']):
            raise ValueError(f'Некорректный вердикт: {edit}')
        seen.add(key)
        groups[dataset].setdefault(row_id, []).append({**edit, 'mcId': category, 'old': old, 'new': new})
    return groups


def apply_review(rows, row_key, reviews, get_labels, set_labels, context):
    changes = []
    pending = reviews.copy()
    for row in rows:
        labels = set(get_labels(row))
        edits = pending.pop(str(row[row_key]), [])
        for edit in edits:
            category = edit['mcId']
            if int(category in labels) != edit['old'] or edit['quote'] not in row['description']:
                raise ValueError(f'Метка или цитата разошлась с источником: {edit}')
            labels.symmetric_difference_update({category})
            changes.append({**edit, **context(row, category)})
        if edits:
            set_labels(row, labels)
    if pending:
        raise ValueError(f'Вердикты без исходной строки: {list(pending)}')
    return changes


def raw_rows(info, rows):
    path = validated(info)
    identities = {row['source_line']: row['source_item_id'] for row in rows}
    matched = {}
    with path.open(encoding='utf-8') as stream:
        for line, value in enumerate(stream, 1):
            if line in identities:
                raw = json.loads(value)
                if raw['item_id'] != identities[line]:
                    raise ValueError(f'ID источника отличается в строке {line}')
                matched[line] = raw
    if line != info['records'] or matched.keys() != identities.keys():
        raise ValueError('Расхождение числа исходных строк')
    return matched


def raw_comparison(rows, raw):
    real = [row for row in rows if row['origin'] == 'real']
    added, removed = Counter(), Counter()
    equal_labels = equal_text = 0
    for row in real:
        source = raw[row['source_line']]
        source_labels, current = set(labels_for(source)), set(labels_for(row))
        added.update(current - source_labels)
        removed.update(source_labels - current)
        equal_labels += source_labels == current
        equal_text += row['description'] == source['description']
    return {'rows': len(real), 'same_label_sets': equal_labels, 'same_description': equal_text,
            'added_vs_raw': {str(c): added[c] for c in LABELS},
            'removed_vs_raw': {str(c): removed[c] for c in LABELS}}


def train_labels(row, labels):
    entries = row['found_microcategories']
    row['found_microcategories'] = [e for e in entries if e['mcId'] in labels]
    row['found_microcategories'].extend(
        {'mcId': cat, 'label': LABELS[cat]} for cat in sorted(labels - {e['mcId'] for e in entries})
    )


def binary_labels(row):
    return [cat for cat in LABELS if row[f'y_{cat}']]


def set_binary_labels(row, labels):
    row.update({f'y_{cat}': int(cat in labels) for cat in LABELS})


def write_version(stage, destination, filename, rows, schema=None):
    path = stage / filename
    if path.suffix == '.parquet':
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, compression='zstd')
    else:
        with path.open('w', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    return artifact(destination / filename, len(rows), path)


def build_train(stage, destination, reviews, common):
    parent = json.loads((PARENT_TRAIN / 'manifest.json').read_text(encoding='utf-8'))
    source = pq.read_table(validated(parent['dataset']))
    rows = source.to_pylist()
    if len(rows) != parent['dataset']['records']:
        raise ValueError('Число строк отличается от манифеста train')
    raw_info = parent['sources']['raw_real']
    raw = raw_rows(raw_info, rows)
    before = raw_comparison(rows, raw)

    def lineage(row, category):
        return {
            'origin': row['origin'], 'source_item_id': row['source_item_id'], 'source_line': row['source_line'],
            'synth_task_id': row['synth_task_id'],
            'raw_source_label': int(category in labels_for(raw[row['source_line']])),
        }

    changes = apply_review(rows, 'item_id', reviews, labels_for, train_labels, lineage)
    fields = ['row_id', 'origin', 'source_item_id', 'source_line', 'synth_task_id', 'mcId', 'old', 'new',
              'raw_source_label', 'quote', 'reason']
    write_csv(stage / 'label_changes.csv', fields,
              [{k: change[k] for k in fields} for change in changes])
    counts = Counter((row['origin'], cat) for row in rows for cat in labels_for(row))
    return {
        **common,
        'sources': {'train': parent['dataset'], 'raw_train': raw_info},
        'dataset': write_version(stage, destination, 'train.parquet', rows, source.schema),
        'label_changes': artifact(destination / 'label_changes.csv', len(changes), stage / 'label_changes.csv'),
        'records_by_origin': dict(Counter(row['origin'] for row in rows)),
        'labels_by_origin': {origin: {str(cat): counts[origin, cat] for cat in LABELS}
                             for origin in ('real', 'synthetic')},
        'real_all_five': sum(row['origin'] == 'real' and len(labels_for(row)) == len(LABELS) for row in rows),
        'raw_comparison': {'parent_train': before, 'guide_train': raw_comparison(rows, raw)},
        'transformations': ['Применены проверенные изменения меток по справочнику',
                            'Состав исходного объединённого train сохранён для сравнения',
                            'synth_checked описывает исходную независимую проверку синтетики'],
    }


def set_list_labels(row, labels):
    row['labels'] = sorted(labels)


def set_stress_labels(row, labels):
    added, removed = labels - set(row['labels']), set(row['labels']) - labels
    for field in ('labels_strict', 'labels_seo'):
        row[field] = sorted((set(row[field]) | added) - removed)
    set_list_labels(row, labels)
    set_binary_labels(row, labels)


def build_tests(stage, destination, reviews, common):
    gold_path = PARENT_TEST / 'test_gold.parquet'
    public_test = ROOT / 'data/raw/public-test-dataset.parquet'
    stress_path = PARENT_TEST / 'stress_suite_v1/stress_suite_v1.parquet'
    gold = pq.read_table(gold_path)
    stress = pq.read_table(stress_path)
    original_test = pq.read_table(public_test, columns=['categories']).column('categories').to_pylist()
    gold_rows, stress_rows = gold.to_pylist(), stress.to_pylist()
    hard_rows = load_hard_test(HARD_SOURCE)
    label_names = {name: cat for cat, name in LABELS.items()}

    def gold_context(row, category):
        public_ids = row['row_ids']
        source_bits = [int(category in {label_names[name] for name in original_test[i]}) for i in public_ids]
        return {'public_labels': json.dumps(dict(zip(public_ids, source_bits)))}

    gold_changes = apply_review(gold_rows, 'row_id', reviews['gold'], binary_labels,
                                set_binary_labels, gold_context)
    gold_by_id = defaultdict(list)
    for change in gold_changes:
        gold_by_id[change['row_id']].append(change)
    stress_reviews = reviews['stress'].copy()
    for row in stress_rows:
        if row['origin_split'] != 'test_gold':
            continue
        result = re.search(r'\brow_id=(\d+)\b', row['note'])
        if not result:
            raise ValueError(f'Нет исходного row_id в stress: {row["row_id"]}')
        inherited = gold_by_id[result[1]]
        if inherited:
            if row['row_id'] in stress_reviews:
                raise ValueError('Пересечение ручных и унаследованных правок stress')
            stress_reviews[row['row_id']] = [
                {**entry, 'dataset': 'stress', 'row_id': row['row_id'],
                 'reason': f'Перенос из gold, row_id={result[1]}. {entry["reason"]}'} for entry in inherited
            ]

    no_context = lambda row, category: {}
    stress_changes = apply_review(stress_rows, 'row_id', stress_reviews, lambda row: row['labels'],
                                  set_stress_labels, no_context)
    for row in stress_rows:
        if row['row_id'] in reviews['stress']:
            row['note'] = 'prep_only: малярная подготовка под обои или покраску подтверждает 106'
    hard_changes = apply_review(hard_rows, 'row_id', reviews['hard'], lambda row: row['labels'],
                                 set_list_labels, no_context)
    changes = gold_changes + stress_changes + hard_changes
    write_csv(stage / 'label_changes.csv',
              ['dataset', 'row_id', 'mcId', 'old', 'new', 'quote', 'reason', 'public_labels'], changes)
    return {
        **common,
        'sources': {'gold': artifact(gold_path, len(gold)), 'stress': artifact(stress_path, len(stress)),
                    'hard': artifact(HARD_SOURCE, len(hard_rows)), 'public': artifact(public_test, len(original_test))},
        'datasets': {'gold': write_version(stage, destination, 'test_gold.parquet', gold_rows, gold.schema),
                     'stress': write_version(stage, destination, 'stress_suite.parquet', stress_rows, stress.schema),
                     'stress_jsonl': write_version(stage, destination, 'stress_suite.jsonl', stress_rows),
                     'hard': write_version(stage, destination, 'hard.jsonl', hard_rows)},
        'label_changes': artifact(destination / 'label_changes.csv', len(changes), stage / 'label_changes.csv'),
        'changes_by_dataset': dict(Counter(change['dataset'] for change in changes)),
        'transformations': ['Применены проверенные изменения меток по справочнику',
                            'Метки копий gold в stress синхронизированы с новым gold',
                            'hard экспортирован в формат тестового стенда JSONL'],
    }


def build(train_out, test_out):
    destinations = (train_out.resolve(), test_out.resolve())
    if destinations[0] == destinations[1] or any(path.exists() for path in destinations):
        raise FileExistsError('Для train и тестов выберите два новых каталога')
    reviews = load_reviews()
    common = {'guide': artifact(GUIDE, None), 'review': artifact(REVIEW, sum(map(len, (
        edits for group in reviews.values() for edits in group.values())))),
        'builder_sha256': sha256(Path(__file__)), 'pyarrow_version': pa.__version__}
    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destinations[0].parent) as train_temp, \
            tempfile.TemporaryDirectory(dir=destinations[1].parent) as test_temp:
        built = []
        for temporary, destination, builder in zip((train_temp, test_temp), destinations, (build_train, build_tests)):
            stage = Path(temporary) / 'version'
            stage.mkdir()
            selected = reviews['train'] if builder is build_train else reviews
            manifest = builder(stage, destination, selected, common)
            (stage / 'manifest.json').write_text(json.dumps({'version': destination.name, **manifest},
                                                           ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            built.append((stage, destination))
        for stage, destination in built:
            stage.rename(destination)
            print(destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-out', type=Path, default=TRAIN_OUT)
    parser.add_argument('--test-out', type=Path, default=TEST_OUT)
    args = parser.parse_args()
    build(args.train_out, args.test_out)
