"""Read the hard-test XLSX without changing it; export validated API JSONL."""
import argparse
import json
from pathlib import Path
import posixpath
import xml.etree.ElementTree as ET
import zipfile


NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
REL = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'


def load_hard_test(path):
    with zipfile.ZipFile(path) as archive:
        book = ET.fromstring(archive.read('xl/workbook.xml'))
        sheets = book.findall('s:sheets/s:sheet', NS)
        sheet = next(s for s in sheets if s.attrib['name'] == 'разметка')
        relations = ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))
        target = next(r.attrib['Target'] for r in relations
                      if r.attrib['Id'] == sheet.attrib[f'{{{REL}}}id'])
        target = target.lstrip('/') if target.startswith('/') else posixpath.normpath('xl/' + target)
        shared = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            shared = [''.join(s.itertext()) for s in
                      ET.fromstring(archive.read('xl/sharedStrings.xml'))]
        rows = []
        for row in ET.fromstring(archive.read(target)).findall('s:sheetData/s:row', NS):
            values = {}
            for cell in row.findall('s:c', NS):
                if cell.find('s:f', NS) is not None:
                    raise ValueError('Formula cells are not supported in the evaluation sheet')
                column = ''.join(c for c in cell.attrib['r'] if c.isalpha())
                kind = cell.attrib.get('t')
                value = cell.findtext('s:v', default='', namespaces=NS)
                if kind == 'inlineStr':
                    value = ''.join(t.text or '' for t in cell.findall('.//s:t', NS))
                elif kind == 's':
                    value = shared[int(value)]
                values[column] = value
            rows.append(values)
    if not rows or [rows[0].get(c) for c in 'ABCDEFG'] != ['id', 'description', '102', '103', '104', '105', '106']:
        raise ValueError('Unexpected hard-test headers')
    result, seen = [], set()
    for row in rows[1:]:
        if not row.get('A') and not row.get('B'):
            continue
        if not row.get('A') or row['A'] in seen or not row.get('B', '').strip():
            raise ValueError('Missing/duplicate ID or empty description')
        if any(row.get(c) not in ('0', '1') for c in 'CDEFG'):
            raise ValueError('Labels must be explicit 0/1 values')
        seen.add(row['A'])
        result.append({'row_id': row['A'], 'description': row['B'], 'mcId': 101,
                       'labels': [mc for c, mc in zip('CDEFG', range(102, 107)) if row[c] == '1']})
    if not result:
        raise ValueError('Empty hard test')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = load_hard_test(args.input)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    print(f'Exported {len(rows)} hard-test rows')


if __name__ == '__main__':
    main()
