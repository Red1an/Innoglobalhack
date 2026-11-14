"""Собирает train_relabel_v1 по согласованным правилам 102–106."""

import argparse
import csv
import json
import tempfile
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from build_train_text_v1 import ROOT, sha256
from train_relabel_rules import relabel

SOURCE = ROOT / 'data/processed/train_dedup_v1'
DESTINATION = ROOT / 'data/processed/train_relabel_v1'
REVIEW = Path(__file__).with_name('train_relabel_v1_reviewed.csv')
LABELS = {
    102: 'Штукатурные работы',
    103: 'Плиточные работы',
    104: 'Сантехника',
    105: 'Электрика',
    106: 'Поклейка обоев и малярные работы',
}


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, columns, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def validated(info):
    path = ROOT / info['path']
    if sha256(path) != info['sha256']:
        raise ValueError(f'SHA-256 расходится с манифестом: {path}')
    return path


def labels_for(row):
    return [entry['mcId'] for entry in row['found_microcategories']]


def build(destination):
    if destination.exists():
        raise FileExistsError(f'Версия уже существует: {destination}')
    parent = json.loads((SOURCE / 'manifest.json').read_text(encoding='utf-8'))
    validated(parent['raw_source'])
    source = pq.read_table(validated(parent['dataset']))
    duplicates = read_csv(validated(parent['removed_duplicates']))
    if len(source) != parent['dataset']['records'] or len(duplicates) != parent['removed_duplicates']['records']:
        raise ValueError('Количество записей расходится с манифестом')
    table = source.to_pylist()
    originals = {row['source_line']: row for row in table}

    reviewed = {}
    for row in read_csv(REVIEW):
        pair = (int(row['source_line']), int(row['mcId']))
        record = originals[pair[0]]
        if pair in reviewed or pair[1] not in LABELS or int(row['item_id']) != record['item_id']:
            raise ValueError(f'Неверная или повторная запись ручной проверки: {pair}')
        if row['decision'] not in ('0', '1', '') or not row['quote'] or row['quote'] not in record['description']:
            raise ValueError(f'Решение или цитата ручной проверки расходится с описанием: {pair}')
        reviewed[pair] = row

    changes = []
    changed_lines = set()
    before, after = Counter(), Counter()
    copies = Counter(int(row['kept_source_line']) for row in duplicates)
    for row in table:
        line = row['source_line']
        description = row['description']
        existing = labels_for(row)
        labels = set(existing)
        before.update(existing)
        proposals = relabel(description, labels)
        for category in LABELS:
            audit = reviewed.get((line, category))
            if audit:
                if audit['decision'] == '':
                    continue
                new = int(audit['decision'])
                quote, reason = audit['quote'], audit['reason']
            elif category in proposals:
                new, quote, reason = proposals[category]
            else:
                continue
            old = int(category in labels)
            if old == new:
                continue
            if not quote or quote not in description:
                raise ValueError(f'Цитата не найдена: строка {line}, категория {category}')
            if new:
                labels.add(category)
            else:
                labels.remove(category)
            changes.append({
                'source_line': line,
                'item_id': row['item_id'],
                'mcId': category,
                'old': old,
                'new': new,
                'method': 'reviewed' if audit else 'rule',
                'quote': quote,
                'reason': reason,
                'removed_duplicates': copies[line],
            })
            changed_lines.add(line)
        if line in changed_lines:
            existing_entries = [entry for entry in row['found_microcategories'] if entry['mcId'] in labels]
            row['found_microcategories'] = existing_entries + [
                {'mcId': category, 'label': LABELS[category]} for category in sorted(labels - set(existing))
            ]
        after.update(labels)

    affected_copies = [row for row in duplicates if int(row['kept_source_line']) in changed_lines]
    counts = Counter((change['mcId'], change['new']) for change in changes)
    with tempfile.TemporaryDirectory(prefix=f'.{DESTINATION.name}-', dir=destination.parent) as temporary:
        staging = Path(temporary) / DESTINATION.name
        staging.mkdir()

        def output_info(name, records):
            return {
                'path': (DESTINATION / name).relative_to(ROOT).as_posix(),
                'sha256': sha256(staging / name),
                'records': records,
            }

        pq.write_table(pa.Table.from_pylist(table, schema=source.schema), staging / 'train.parquet', compression='zstd')
        write_csv(staging / 'label_changes.csv',
                  ['source_line', 'item_id', 'mcId', 'old', 'new', 'method', 'quote', 'reason', 'removed_duplicates'],
                  changes)
        write_csv(staging / 'affected_duplicates.csv',
                  ['source_line', 'item_id', 'kept_source_line', 'kept_item_id', 'reason'], affected_copies)
        files = (Path(__file__), REVIEW, Path(__file__).with_name('train_relabel_rules.py'))
        manifest = {
            'version': DESTINATION.name,
            'source': parent['dataset'],
            'raw_source': parent['raw_source'],
            'source_removed_duplicates': parent['removed_duplicates'],
            'dataset': output_info('train.parquet', len(table)),
            'label_changes': output_info('label_changes.csv', len(changes)),
            'affected_duplicates': output_info('affected_duplicates.csv', len(affected_copies)),
            'changed_rows': len(changed_lines),
            'transformations': [{
                'id': 'relabel_102_106',
                'description': (
                    'Применены согласованные правила подтверждения услуг по полному описанию. '
                    'Явные услуги в составе комплекса учитываются. Широкие SEO-перечни и материалы сами по себе '
                    'не добавляют метку. Тексты не менялись. Индивидуальные решения имеют приоритет. '
                    'При недостатке доказательств сохраняется текущая метка.'
                ),
                'reviewed_pairs': len(reviewed),
                'reviewed_uncertain': sum(not row['decision'] for row in reviewed.values()),
                'by_category': {
                    category: {'before': before[category], 'removed': counts[category, 0],
                               'added': counts[category, 1], 'after': after[category]}
                    for category in LABELS
                },
            }],
            'build_files_sha256': {p.resolve().relative_to(ROOT).as_posix(): sha256(p) for p in files},
            'pyarrow_version': pa.__version__,
        }
        (staging / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
                                              encoding='utf-8')
        staging.rename(destination)
    print(f'{len(table)} строк, изменено {len(changed_lines)} строк и {len(changes)} меток.')
    print(json.dumps(manifest['transformations'][0]['by_category'], ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DESTINATION)
    build(parser.parse_args().out)
