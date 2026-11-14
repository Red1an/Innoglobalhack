"""Собирает train_relabel_v2 поверх train_relabel_v1 и исходных текстов."""

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from build_train_dedup_v1 import duplicate_key, replace_visual_spaces, without_stock
from build_train_relabel_v1 import LABELS, REVIEW, labels_for, read_csv, validated, write_csv
from build_train_text_v1 import ROOT, TEXT_FIELDS, sha256
from train_relabel_rules import relabel
from train_text_cleaning import STEPS, STEPS_V1, normalize_emoji

SOURCE = ROOT / 'data/processed/train_relabel_v1/manifest.json'
DESTINATION = ROOT / 'data/processed/train_relabel_v2'


def raw_texts(path, selected, records):
    result = {}
    count = 0
    with path.open(encoding='utf-8') as stream:
        for count, line in enumerate(stream, 1):
            if count in selected:
                result[count] = json.loads(line)
    if count != records or len(result) != len(selected):
        raise ValueError('Количество исходных строк расходится с манифестом')
    return result


def cleaned_field(raw, field, steps):
    changes = []
    value = raw
    for name, transform, _ in steps:
        updated = transform(value)
        if name == 'emoji' and steps is STEPS:
            if updated != normalize_emoji(value, separate_keycaps=False):
                changes.append('keycap_spacing')
        elif name in {'joined_list', 'joined_numbers', 'decorative_separators'} and updated != value:
            changes.append(name)
        value = updated
    value = replace_visual_spaces(value)
    return (without_stock(value) if field == 'description' else value), changes


def build(destination):
    if destination.exists():
        raise FileExistsError(f'Версия уже существует: {destination}')
    parent = json.loads(SOURCE.read_text(encoding='utf-8'))
    source = pq.read_table(validated(parent['dataset']))
    if len(source) != parent['dataset']['records']:
        raise ValueError('Количество записей родительской версии расходится с манифестом')
    rows = source.to_pylist()
    original = raw_texts(validated(parent['raw_source']), {row['source_line'] for row in rows},
                         parent['raw_source']['records'])
    duplicates = read_csv(validated(parent['source_removed_duplicates']))
    if len(duplicates) != parent['source_removed_duplicates']['records']:
        raise ValueError('Количество удалённых дублей расходится с манифестом')
    copies = Counter(int(row['kept_source_line']) for row in duplicates)
    reviewed = {(int(row['source_line']), int(row['mcId'])) for row in read_csv(REVIEW)}

    label_changes, text_changes = [], []
    before, after = Counter(), Counter()
    keys = set()
    for row in rows:
        line = row['source_line']
        item_id = row['item_id']
        raw = original[line]
        if raw['item_id'] != item_id:
            raise ValueError(f'Исходный item_id расходится со строкой {line}')
        for field in TEXT_FIELDS:
            previous, _ = cleaned_field(raw[field], field, STEPS_V1)
            if previous != row[field]:
                raise ValueError(f'Старая очистка не воспроизводит {field} в строке {line}')
            cleaned, changes = cleaned_field(raw[field], field, STEPS)
            if not cleaned:
                raise ValueError(f'Пустое поле {field} после очистки в строке {line}')
            if cleaned != previous:
                text_changes.append({'source_line': line, 'item_id': item_id, 'field': field,
                                     'transformations': ','.join(changes)})
                row[field] = cleaned
        description = row['description']
        key = duplicate_key(description)
        if key in keys:
            raise ValueError(f'Дубликат описания после очистки в строке {line}')
        keys.add(key)

        existing = set(labels_for(row))
        labels = existing.copy()
        before.update(existing)
        for category, (new, quote, reason) in relabel(description, existing).items():
            if (line, category) in reviewed:
                continue
            if not quote or quote not in description:
                raise ValueError(f'Цитата не найдена в строке {line}, метка {category}')
            if new:
                labels.add(category)
            else:
                labels.remove(category)
            label_changes.append({
                'source_line': line, 'item_id': item_id, 'mcId': category, 'old': int(category in existing),
                'new': new, 'quote': quote, 'reason': reason, 'removed_duplicates': copies[line],
            })
        after.update(labels)
        if labels != existing:
            row['found_microcategories'] = [
                entry for entry in row['found_microcategories'] if entry['mcId'] in labels
            ] + [{'mcId': category, 'label': LABELS[category]} for category in sorted(labels - existing)]

    label_lines = {change['source_line'] for change in label_changes}
    text_lines = {change['source_line'] for change in text_changes}
    changed_copies = [row for row in duplicates if int(row['kept_source_line']) in label_lines | text_lines]
    counts = Counter((change['mcId'], change['new']) for change in label_changes)
    with tempfile.TemporaryDirectory(prefix=f'.{DESTINATION.name}-', dir=destination.parent) as temporary:
        stage = Path(temporary) / DESTINATION.name
        stage.mkdir()

        def artifact(name, records):
            path = destination.resolve() / name
            if path.is_relative_to(ROOT):
                path = path.relative_to(ROOT)
            return {'path': path.as_posix(), 'sha256': sha256(stage / name), 'records': records}

        pq.write_table(pa.Table.from_pylist(rows, schema=source.schema), stage / 'train.parquet', compression='zstd')
        write_csv(stage / 'label_changes.csv',
                  ['source_line', 'item_id', 'mcId', 'old', 'new', 'quote', 'reason', 'removed_duplicates'],
                  label_changes)
        write_csv(stage / 'text_changes.csv', ['source_line', 'item_id', 'field', 'transformations'], text_changes)
        write_csv(stage / 'affected_duplicates.csv',
                  ['source_line', 'item_id', 'kept_source_line', 'kept_item_id', 'reason'], changed_copies)
        files = (
            Path(__file__).name, REVIEW.name, 'build_train_text_v1.py', 'build_train_dedup_v1.py',
            'build_train_relabel_v1.py', 'train_text_cleaning.py', 'emoji_ranges.txt', 'train_relabel_rules.py',
        )
        manifest = {
            'version': DESTINATION.name,
            'source': parent['dataset'],
            'raw_source': parent['raw_source'],
            'source_removed_duplicates': parent['source_removed_duplicates'],
            'dataset': artifact('train.parquet', len(rows)),
            'label_changes': artifact('label_changes.csv', len(label_changes)),
            'text_changes': artifact('text_changes.csv', len(text_changes)),
            'affected_duplicates': artifact('affected_duplicates.csv', len(changed_copies)),
            'lineage': {'item_id': 'исходный item_id', 'source_line': 'номер строки исходного JSONL с единицы'},
            'transformed_text_rows': len(text_lines),
            'relabeled_rows': len(label_lines),
            'transformations': [
                {'id': 'clean_from_raw',
                 'description': 'Разделение цифр от keycap-эмодзи и склеенных пунктов перечней, исправление пробелов '
                                'в числовых выражениях и нормализация декоративных разделителей.',
                 'by_rule': dict(Counter(rule for change in text_changes
                                         for rule in change['transformations'].split(',')))},
                {'id': 'relabel_102_106',
                 'description': 'Новые правила подтверждения услуг по основному тексту поверх меток train_relabel_v1. '
                                'Поисковые блоки сохранены в данных, учтены результаты индивидуальной проверки v1.',
                 'by_category': {category: {'before': before[category], 'removed': counts[category, 0],
                                            'added': counts[category, 1], 'after': after[category]}
                                 for category in LABELS}},
            ],
            'build_files_sha256': {
                f'dataset_review/{name}': sha256(Path(__file__).with_name(name)) for name in files
            },
            'pyarrow_version': pa.__version__,
        }
        (stage / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
                                             encoding='utf-8')
        stage.rename(destination)
    print(f'{len(rows)} строк, исправлены тексты {len(text_lines)} строк, '
          f'изменены {len(label_changes)} меток в {len(label_lines)} строках.')
    print(json.dumps(manifest['transformations'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DESTINATION)
    build(parser.parse_args().out)
