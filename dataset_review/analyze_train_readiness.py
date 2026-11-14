"""Read-only analysis of current train; writes audit artifacts, never datasets.

Run from the repository: .venv/bin/python dataset_review/analyze_train_readiness.py
"""
from collections import Counter, defaultdict
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo
import ast
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pyarrow.parquet as pq

from build_train_dedup_v1 import without_stock
from train_text_cleaning import clean_text

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/train_readiness_2026-11-14'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def labels(row):
    if 'found_microcategories' in row:
        return tuple(sorted(x['mcId'] for x in row['found_microcategories']))
    if 'labels' in row:
        return tuple(row['labels'])
    return tuple(m for m in range(102, 107) if row[f'y_{m}'])


@lru_cache(maxsize=30000)
def key(text):
    return ' '.join(clean_text(text).casefold().split())


def group_stats(rows, transform):
    groups = defaultdict(list)
    for row in rows:
        groups[transform(row['description'])].append(row)
    return groups, {
        'groups': len(groups), 'excess_rows': len(rows) - len(groups),
        'rows_in_duplicate_groups': sum(len(g) for g in groups.values() if len(g) > 1),
        'conflicting_groups': sum(len({labels(r) for r in g}) > 1 for g in groups.values()),
        'largest_groups': sorted((len(g) for g in groups.values()), reverse=True)[:10],
    }


def workbook_status(path):
    ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    result = {}
    with zipfile.ZipFile(path) as archive:
        shared = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            shared = [''.join(n.itertext()) for n in ET.fromstring(archive.read('xl/sharedStrings.xml'))]
        for name in archive.namelist():
            if not (name.startswith('xl/worksheets/sheet') and name.endswith('.xml')):
                continue
            rows = []
            for node in ET.fromstring(archive.read(name)).findall('.//s:sheetData/s:row', ns):
                row = {}
                for cell in node:
                    value = cell.find('s:v', ns)
                    value = value.text if value is not None else None
                    if cell.get('t') == 'inlineStr':
                        inline = cell.find('s:is', ns)
                        value = ''.join(inline.itertext()) if inline is not None else None
                    elif cell.get('t') == 's':
                        value = shared[int(value)]
                    row[re.sub(r'\d', '', cell.get('r'))] = value
                rows.append(row)
            header = rows[0]
            result[name] = {'rows': len(rows)-1, 'filled': {
                title: sum(r.get(column) not in (None, '') for r in rows[1:])
                for column, title in header.items() if title in ('decision', 'comment')
            }}
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw_path = ROOT / 'data/raw/public-train-annotated.jsonl'
    raw = [json.loads(line) for line in raw_path.read_text().splitlines()]
    path = ROOT / 'data/processed/train_text_v1/train.parquet'
    train = pq.read_table(path).to_pylist()
    gold_path = ROOT / 'data/test/test_gold.parquet'
    gold = pq.read_table(gold_path).to_pylist()
    old = pq.read_table(ROOT / 'data/processed/train_clean.parquet').to_pylist()
    manifest = json.loads(path.with_name('manifest.json').read_text())
    assert sha(raw_path) == manifest['source']['sha256']
    assert sha(path) == manifest['dataset']['sha256']
    assert len(raw) == len(train) == 10000
    assert all(r['item_id'] == c['item_id'] and c['source_line'] == i+1 and
               labels(r) == labels(c) and all(clean_text(r[f]) == c[f] == clean_text(c[f])
               for f in ('title', 'description')) for i, (r, c) in enumerate(zip(raw, train)))
    stats = {'created_msk': datetime.now(ZoneInfo('Europe/Moscow')).isoformat(),
             'hashes': {str(p.relative_to(ROOT)): sha(p) for p in (raw_path, path, gold_path)},
             'integrity_verified_rows': len(train),
             'old_train_description_identical_to_raw': sum(o['description'] == r['description'] for o,r in zip(old,raw)),
             'old_train_description_different_from_v1': sum(o['description'] != r['description'] for o,r in zip(old,train)),
             'duplicates': {}, 'distributions': {}, 'candidates': {}}
    groupings = {}
    for name, rows, transform in [('raw_exact',raw,lambda t:t), ('v1_exact',train,lambda t:t),
                                  ('v1_key',train,key), ('v1_no_stock_key',train,lambda t:key(without_stock(t)))]:
        groupings[name], stats['duplicates'][name] = group_stats(rows, transform)
    for name, rows in [('train',train), ('train_unique', [g[0] for g in groupings['v1_no_stock_key'].values()]), ('gold',gold)]:
        counts=Counter(len(labels(r)) for r in rows)
        stats['distributions'][name] = {'rows':len(rows),'label_cardinality':dict(sorted(counts.items())),
            'categories':{m:sum(m in labels(r) for r in rows) for m in range(102,107)},
            'chars_quantiles':dict(zip(['min','median','p90','p95','p99','max'],
                np.quantile([len(r['description']) for r in rows], [0,.5,.9,.95,.99,1]).tolist()))}
    gold_keys = {key(without_stock(r['description'])):r for r in gold}
    overlap=[]
    for text, rows in groupings['v1_no_stock_key'].items():
        if text in gold_keys:
            g=gold_keys[text]
            overlap.append({'train_source_lines':[r['source_line'] for r in rows], 'gold_row_id':g['row_id'],
                            'train_labels':labels(rows[0]), 'gold_labels':labels(g)})
    stats['overlap']={'groups':len(overlap), 'train_rows':sum(len(r['train_source_lines']) for r in overlap),
                      'different_label_groups':sum(r['train_labels'] != r['gold_labels'] for r in overlap), 'details':overlap}
    # Reuse the exact keyword inventory of the existing audit without executing its notebook.
    notebook=json.loads((ROOT/'dataset_review/02_train_full_audit.ipynb').read_text())
    tree=ast.parse('\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code'))
    rules=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
               and any(isinstance(t,ast.Name) and t.id=='KEYWORDS' for t in n.targets))
    compiled={m:re.compile(p) for m,p in rules.items()}
    evidence=defaultdict(list)
    for r in train:
        text=r['description'].lower().replace('ё','е'); title=r['title'].lower().replace('ё','е')
        hits={m:list(p.finditer(text)) for m,p in compiled.items()}
        def add(name, mc=None):
            evidence[name].append({'source_line':r['source_line'],'item_id':r['item_id'],'mcId':mc,'labels':labels(r)})
        for m, hh in hits.items():
            if hh and m not in labels(r): add('keyword_without_label',m)
            if not hh and m in labels(r): add('label_without_keyword',m)
            if not hh and compiled[m].search(title): add('title_only',m)
        if 102 in labels(r) and not re.search(r'штукатур|оштукатур|маяк', text): add('102_without_plaster_or_beacon',102)
        if re.search(r'шпакл|шпатл',text) and not re.search(r'обои|обоев|обоя|поклей|оклей|маляр|покрас|окрас|стеклохолст',text): add('putty_without_finish_cues',106)
        preparation = re.compile(r'\b(?:под|к)\s+(?:покраск\w*|окраск\w*|обои|обоям|поклейк\w*(?:\s+обоев)?|оклейк\w*(?:\s+обоев)?)(?:\s+(?:или|и)\s+(?:покраск\w*|окраск\w*|обои|обоям))?')
        if 106 in labels(r) and re.search(r'шпакл|шпатл|подготов',text):
            remaining = preparation.sub(' ', text)
            if not re.search(r'\b(?:обои|обоев|обоя\w*|поклей\w*|оклей\w*|маляр\w*|покрас\w*|окрас\w*|стеклохолст\w*)', remaining):
                add('106_preparation_candidate',106)
        if re.search(r'[\u3164\u2800\u115f\u1160]',text): add('visual_blank_characters')
        if re.search(r'номер (?:объекта|оборудования)|код товара|артикул|объявление №|инд:',text): add('technical_code_candidate')
        if re.search(r'для поиска|ключевые слова|поисковые запросы|нас (?:также )?ищут',text): add('explicit_seo')
        if without_stock(r['description']) != r['description']: add('stock_suffix')
        if len(text)<40: add('short_under_40')
    for name, events in evidence.items():
        lines={e['source_line'] for e in events}
        stats['candidates'][name]={'pairs':len(events),'rows':len(lines),
            'unique_descriptions':len({key(train[n-1]['description']) for n in lines}),
            'positive_pairs':sum(e['mcId'] in e['labels'] for e in events if e['mcId'] is not None)}
    stats['truncation_chars_description_only']=[]
    for budget in (512,1024,2048,4096):
        pairs=[]
        for r in train:
            text=r['description'].lower().replace('ё','е')
            for m in labels(r):
                matches=list(compiled[m].finditer(text))
                if matches and all(x.end()>budget for x in matches):pairs.append((r['source_line'],m))
        stats['truncation_chars_description_only'].append({'chars':budget,'truncated_rows':sum(len(r['description'])>budget for r in train),
             'affected_rows':len({p[0] for p in pairs}),'labelled_pairs_without_complete_hit':len(pairs)})
    stats['workbooks']={str(p.relative_to(ROOT)):workbook_status(p) for p in (ROOT/'review').glob('*.xlsx')}
    stats['test_versions']={}
    for p in sorted((ROOT/'data/test').glob('*.parquet')):
        rr=pq.read_table(p).to_pylist()
        stats['test_versions'][p.name]={'rows':len(rr),'positive_labels':sum(len(labels(r)) for r in rr),'empty':sum(not labels(r) for r in rr)}
    for name, value in [('summary.json',stats),('candidate_evidence.json',evidence)]:
        (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(stats,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
