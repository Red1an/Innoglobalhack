"""Build a standalone HTML audit from the adjacent Markdown and findings JSON.

Standard library only. This renders existing audit results; it does not recompute
dataset statistics. Original datasets are never modified.
"""
from pathlib import Path
import html
import json
import re

ROOT = Path(__file__).resolve().parent
findings = json.loads((ROOT / 'audit_findings.json').read_text())
markdown = (ROOT / 'avito_data_audit.md').read_text()

def inline(text):
    text = html.escape(text)
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    return re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)

def render(text):
    lines = text.splitlines()
    parts, navigation = [], []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.startswith('|'):
            table = []
            while i < len(lines) and lines[i].startswith('|'):
                cells = [c.strip() for c in lines[i].strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?', c) for c in cells):
                    table.append(cells)
                i += 1
            parts.append('<div class="table-wrap"><table><thead><tr>' + ''.join('<th>' + inline(c) + '</th>' for c in table[0]) + '</tr></thead><tbody>')
            parts.extend('<tr>' + ''.join('<td>' + inline(c) + '</td>' for c in row) + '</tr>' for row in table[1:])
            parts.append('</tbody></table></div>')
            continue
        match = re.match(r'^(#{1,3}) (.*)', line)
        if match:
            level, title = len(match[1]), match[2]
            anchor = 'section-' + str(len(navigation)) if level == 2 else ''
            if level == 2:
                navigation.append((anchor, title))
            parts.append(f'<h{level} id="{anchor}">{inline(title)}</h{level}>')
            i += 1
            continue
        if re.match(r'^(?:- |\d+\. )', line):
            ordered = bool(re.match(r'^\d+\. ', line))
            pattern = r'^\d+\. ' if ordered else r'^- '
            tag = 'ol' if ordered else 'ul'
            parts.append('<' + tag + '>')
            while i < len(lines) and re.match(pattern, lines[i]):
                parts.append('<li>' + inline(re.sub(pattern, '', lines[i])) + '</li>')
                i += 1
            parts.append('</' + tag + '>')
            continue
        paragraph = []
        while i < len(lines) and lines[i].strip() and not re.match(r'^(?:#|\||- |\d+\. )', lines[i]):
            paragraph.append(lines[i])
            i += 1
        parts.append('<p>' + inline(' '.join(paragraph)) + '</p>')
    return '\n'.join(parts), navigation

body, navigation = render(markdown)
kinds = {'error': 'Ошибка по справочнику', 'probable': 'Вероятная ошибка', 'policy': 'Нужно правило', 'noise': 'Шум в тексте', 'valid': 'Корректный сложный пример'}
cards = []
for case in findings['reviewed_cases']:
    label = ', '.join(map(str, case['labels'])) or 'пустой набор'
    location = f"{case['source']} · строка {case['row']}"
    if case['item_id']:
        location += f" · item_id {case['item_id']}"
    cards.append(f'''<article class="case" data-kind="{case['kind']}">
      <div class="case-meta">{html.escape(location)} <span class="badge {case['kind']}">{kinds[case['kind']]}</span></div>
      <h3>{html.escape(case['finding'])}</h3>
      <p>{html.escape(case['comment'])}</p>
      <p class="muted">Исходные метки: <strong>{label}</strong> · Одинаковых описаний в этом файле: {case['description_duplicates']}</p>
      <details><summary>Показать исходный текст: {html.escape(case['title'])}</summary><pre>{html.escape(case['description'])}</pre></details>
    </article>''')

css = '''
:root{color-scheme:light;--ink:#172534;--muted:#536476;--accent:#c84c27;--line:#dce3e9;--paper:#fff;--back:#eef2f5}
*{box-sizing:border-box}body{margin:0;background:var(--back);color:var(--ink);font:16px/1.68 system-ui,-apple-system,"Segoe UI",sans-serif}
.top{background:#142737;color:#fff;padding:44px max(24px,calc((100vw - 1120px)/2)) 34px}.eyebrow{text-transform:uppercase;letter-spacing:.14em;font-size:12px;color:#ffb28f}.top h1{font-size:clamp(28px,5vw,46px);line-height:1.15;margin:14px 0}.top p{max-width:810px;color:#d3dfe8}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:28px}.metric{border-top:3px solid #ff986e;padding:12px 0}.metric strong{display:block;font-size:30px;line-height:1.3}.metric span{font-size:13px;color:#d3dfe8}
main{max-width:1120px;margin:24px auto;padding:30px 44px;background:var(--paper);border:1px solid var(--line);border-radius:12px}h1{font-size:27px}h2{font-size:25px;line-height:1.3;margin:40px 0 18px;padding-top:14px;border-top:1px solid var(--line);scroll-margin-top:20px}h3{font-size:19px;line-height:1.4}p,li{overflow-wrap:anywhere}li{margin:9px 0}code{background:#edf2f5;padding:2px 5px;border-radius:4px;font-size:.9em;overflow-wrap:anywhere}a{color:#ab3c1e}.table-wrap{overflow-x:auto;margin:20px 0}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:12px 14px;text-align:left;vertical-align:top;border-bottom:1px solid var(--line)}th{background:#eaf0f4}tr:nth-child(even){background:#f8fafb}
.nav{display:flex;gap:10px;flex-wrap:wrap}.nav a{border:1px solid var(--line);border-radius:7px;text-decoration:none;padding:6px 12px;font-size:13px}.actions{display:flex;gap:10px;flex-wrap:wrap;margin:22px 0}button,select,input{font:inherit;border:1px solid #b9c7d1;border-radius:7px;padding:10px 14px;background:white;color:var(--ink)}button{cursor:pointer}button.primary{background:#142737;color:white;border-color:#142737}button:hover{filter:brightness(.94)}input{flex:1;min-width:200px}button:focus-visible,a:focus-visible,summary:focus-visible,input:focus,select:focus{outline:3px solid #f6aa73;outline-offset:3px}
.case{padding:23px;border:1px solid var(--line);border-radius:10px;margin:16px 0}.case h3{margin:12px 0}.case-meta{font-size:12px;color:var(--muted);display:flex;gap:12px;align-items:center;justify-content:space-between;flex-wrap:wrap}.badge{padding:4px 9px;border-radius:4px;font-weight:650;background:#e8eef4}.error{background:#ffe6df;color:#9a351d}.valid{background:#e4f2e9;color:#20563a}.probable,.policy{background:#fff0d2;color:#73500d}.muted{color:var(--muted);font-size:14px}summary{cursor:pointer;color:#9c351b}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.7 system-ui,sans-serif;padding:16px;background:#f5f7f9;border-radius:6px}footer{max-width:1120px;margin:20px auto 40px;padding:0 20px;color:var(--muted);font-size:13px}.case[hidden]{display:none}
@media(max-width:720px){main{margin:12px;padding:20px;border-radius:8px}.metrics{grid-template-columns:repeat(2,1fr)}.top{padding:30px 24px}h2{font-size:22px}.case{padding:16px}th,td{padding:9px}.actions>*{width:100%}.case-meta{display:block}.badge{display:inline-block;margin-top:6px}}
@media print{body{background:white}main{border:0;padding:0;max-width:none}.top{padding:10px;color:black;background:white}.top p,.metric span{color:#333}.eyebrow{color:#333}.actions,.nav,#case-tools,footer{display:none}h2{break-after:avoid}.case,table{break-inside:avoid}.metrics{grid-template-columns:repeat(4,1fr)}details:not([open]){display:none}}
'''
js = '''
const input = document.getElementById('case-search');
const filter = document.getElementById('kind-filter');
const cards = Array.from(document.querySelectorAll('.case'));
function applyFilter(){
  const query=input.value.toLocaleLowerCase('ru');let n=0;
  for(const card of cards){const match=(!filter.value||card.dataset.kind===filter.value)&&card.textContent.toLocaleLowerCase('ru').includes(query);card.hidden=!match;if(match)n++;}
  document.getElementById('case-count').textContent='Показано примеров: '+n+' из '+cards.length;
}
input.addEventListener('input',applyFilter);filter.addEventListener('change',applyFilter);applyFilter();
function download(text,name,type){const url=URL.createObjectURL(new Blob([text],{type}));const a=document.createElement('a');a.href=url;a.download=name;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);}
document.getElementById('download-json').addEventListener('click',()=>download(document.getElementById('audit-data').textContent,'audit_findings.json','application/json'));
document.getElementById('download-md').addEventListener('click',()=>download(JSON.parse(document.getElementById('report-markdown').textContent),'avito_data_audit.md','text/markdown;charset=utf-8'));
document.getElementById('print').addEventListener('click',()=>window.print());
'''
nav = ''.join(f'<a href="#{anchor}">{html.escape(title)}</a>' for anchor,title in navigation)
options = ''.join(f'<option value="{key}">{value}</option>' for key,value in kinds.items())
def script_json(value):
    return json.dumps(value,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')

document = f'''<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Аудит данных Avito — Инноглобалхак</title><style>{css}</style></head><body>
<header class="top"><div class="eyebrow">Инноглобалхак · Avito · 14 ноября 2026</div><h1>Сначала проверить эталон</h1><p>Полный технический аудит двух файлов и разбор ошибок разметки. Данные пригодны для прототипа; доверять локальной метрике без проверки повторов и эталона нельзя.</p><div class="metrics"><div class="metric"><strong>10 208</strong><span>записей проверено программно</span></div><div class="metric"><strong>4 866</strong><span>разных описаний в train</span></div><div class="metric"><strong>13 групп</strong><span>противоречивых дублей в test</span></div><div class="metric"><strong>22 / 208</strong><span>описания test встречаются в train</span></div></div></header>
<main><nav class="nav" aria-label="Разделы отчёта">{nav}<a href="#examples">Примеры с исходными текстами</a></nav><div class="actions"><button class="primary" id="download-json">Скачать все результаты JSON</button><button id="download-md">Скачать отчёт Markdown</button><button id="print">Печать / PDF</button></div>{body}
<h2 id="examples">23 разобранных примера с исходными текстами</h2><p>«Ошибка» — вывод по тексту и предоставленному справочнику; «Нужно правило» — случай, который нельзя безопасно исправить без единой политики разметки. Для правок test предполагается исходная категория 101. Никакие исходные записи не изменены.</p><div id="case-tools"><div class="actions"><input id="case-search" aria-label="Поиск по разобранным примерам" placeholder="Слово, item_id или номер строки"><select id="kind-filter" aria-label="Тип замечания"><option value="">Все типы замечаний</option>{options}</select></div><p id="case-count" class="muted" aria-live="polite"></p></div>{''.join(cards)}</main><footer>Технические проверки покрывают оба файла полностью. Смысловая проверка — выборочная, по подозрительным случаям; это не полная экспертная переразметка. Отчёт работает без подключения к интернету.</footer>
<script id="audit-data" type="application/json">{script_json(findings)}</script><script id="report-markdown" type="application/json">{script_json(markdown)}</script><script>{js}</script></body></html>'''
(ROOT/'avito_data_audit.html').write_text(document)
print(f'HTML written: {len(document.encode())} bytes; {len(cards)} cases; {len(navigation)} sections')
