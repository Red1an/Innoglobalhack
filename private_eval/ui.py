"""Простой веб-интерфейс к run.py: выбор датасетов и моделей, прогон каждой модели по каждому датасету, выгрузка ответов.

Запуск из codebase/: .venv/bin/python private_eval/ui.py, затем http://localhost:8765
Датасеты на входе: CSV организаторов testcase,description (UTF-8, BOM допускается).
Эталон для подсчёта F1: <имя>.gold.csv рядом, в формате ответа testcase,microcats,descriptions.
"""
import json
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from private_eval import fmt  # noqa: E402
from private_eval.llm import API_ALLOWED  # noqa: E402

INPUT, RUNS, EXAMPLES = HERE / "input", HERE / "runs", HERE / "examples"
PROMPTS = ROOT / "research" / "experiments" / "prompts"
PORT = 8765


def model_name(tag):
    """hf.co/NightForger/avibe-GGUF:Q8_0 → avibe Q8_0, t-lite-it-2.1:q8_0 → t-lite-it-2.1 Q8_0."""
    name = (tag or "").split("/")[-1].replace("-GGUF", "")
    base, _, quant = name.partition(":")
    return f"{base} {quant.upper()}" if quant else base


def run_title(method, tag):
    if method == "charlr":
        return "Char-LR"
    return f"Char-LR + {model_name(tag)}" if method == "cascade" else model_name(tag)


def presets():
    from private_eval.llm import OLLAMA_MODELS
    items = [{"id": "charlr", "label": "Char-LR", "args": ["--method", "charlr"]}]
    for name in API_ALLOWED:
        items.append({"id": f"llm-api-{name}", "label": name, "where": "платный API",
                      "args": ["--method", "llm", "--backend", "api", "--model", name]})
        items.append({"id": f"cascade-api-{name}", "label": f"Char-LR + {name}", "where": "платный API",
                      "args": ["--method", "cascade", "--backend", "api", "--model", name]})
    for m, tag in OLLAMA_MODELS.items():
        items.append({"id": f"llm-local-{m}", "label": model_name(tag), "where": "Ollama",
                      "args": ["--method", "llm", "--backend", "local", "--model", m]})
        items.append({"id": f"cascade-local-{m}", "label": f"Char-LR + {model_name(tag)}", "where": "Ollama",
                      "args": ["--method", "cascade", "--backend", "local", "--model", m]})
    return items


def datasets():
    files = sorted(p for p in INPUT.glob("*.csv") if not p.name.endswith(".gold.csv"))
    files += sorted(p for p in EXAMPLES.glob("*.csv") if not p.name.endswith(".gold.csv"))
    out = []
    for p in files:
        try:
            rows = len(fmt.read_input(p))
            error = None
        except (ValueError, OSError) as exc:
            rows, error = 0, str(exc)
        out.append({"path": str(p.relative_to(HERE)), "name": p.stem, "rows": rows, "error": error,
                    "deletable": p.parent == INPUT, "modified": p.stat().st_mtime,
                    "gold": p.with_name(p.stem + ".gold.csv").exists()})
    return out


def runs():
    out = []
    for meta_path in sorted(RUNS.glob("*/*/meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        score_path = meta_path.with_name("score.json")
        gold = Path(meta["input"]).with_name(Path(meta["input"]).stem + ".gold.csv")
        submission = meta_path.with_name("submission.csv")
        if gold.exists() and submission.exists() and (
                not score_path.exists() or gold.stat().st_mtime > score_path.stat().st_mtime):
            from private_eval.run import score as score_run
            result = score_run(submission, gold)
            score_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        score = json.loads(score_path.read_text(encoding="utf-8")) if score_path.exists() and gold.exists() else {}
        per = score.get("per_category_f1") or {}
        macro = score.get("macro_f1", sum(per.values()) / len(per) if per else None)
        run_dir = meta_path.parent
        seconds = meta["classify_seconds"] + meta.get("gen_seconds", 0)
        out.append({"dataset": run_dir.parent.name, "title": run_title(meta["method"], meta.get("model")) + (" · предобработка" if meta.get("prep") else ""),
                    "prompt": meta.get("prompt"), "f1": score.get("micro_f1"), "macro": macro,
                    "seconds": round(seconds, 1), "per_row": round(seconds / max(meta["rows"], 1), 2),
                    "llm_calls": meta["llm_calls"], "errors": meta["llm_errors"], "created": meta["created"],
                    "download": f"/download/{run_dir.parent.name}/{run_dir.name}/submission.csv"})
    return sorted(out, key=lambda r: r["created"], reverse=True)


PROMPT_HISTORY = HERE / "prompt_history"


def prompt_path(name):
    path = PROMPTS / f"{name}.txt"
    if not re.fullmatch(r"[\w.-]+", name or "") or not path.exists():
        raise ValueError(f"Нет промпта {name}")
    return path


def prompt_versions(name):
    folder = PROMPT_HISTORY / name
    return [{"id": p.stem, "time": p.stem} for p in sorted(folder.glob("*.txt"), key=lambda f: f.stat().st_mtime_ns, reverse=True)] if folder.exists() else []


def prompt_info(name):
    return {"name": name, "text": prompt_path(name).read_text(encoding="utf-8"), "versions": prompt_versions(name)}


def write_prompt(name, text):
    """Saves the current text into prompt_history before overwriting, so every state can be restored."""
    path = prompt_path(name)
    current = path.read_text(encoding="utf-8")
    if text == current:
        return prompt_info(name)
    folder = PROMPT_HISTORY / name
    folder.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H-%M-%S")
    n = 1
    while (folder / f"{stamp}{'' if n == 1 else f' ({n})'}.txt").exists():
        n += 1
    (folder / f"{stamp}{'' if n == 1 else f' ({n})'}.txt").write_text(current, encoding="utf-8")
    path.write_text(text, encoding="utf-8")
    return prompt_info(name)


def prompt_version(name, version):
    path = PROMPT_HISTORY / name / f"{version}.txt"
    if not re.fullmatch(r"[\w .()-]+", version or "") or not path.exists():
        raise ValueError("Нет такой версии")
    return path.read_text(encoding="utf-8")


DEFAULT_RATE = {"charlr": 0.01, "llm-local": 3.0, "cascade-local": 1.5, "llm-api": 1.0, "cascade-api": 0.6}


def history_rate(preset_id):
    """Median seconds per ad over earlier runs of this preset that really called the model."""
    rates = []
    for meta_path in RUNS.glob(f"*/{preset_id}*/meta.json"):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["rows"] and (meta["llm_calls"] == 0 or meta.get("llm_cached", 0) == 0):
            rates.append(meta["classify_seconds"] / meta["rows"])
    if rates:
        rates.sort()
        return rates[len(rates) // 2], True
    kind = next((k for k in DEFAULT_RATE if preset_id.startswith(k)), "llm-local")
    return DEFAULT_RATE[kind], False


class Job:
    def __init__(self):
        self.lock = threading.Lock()
        self.state = {"running": False, "log": "", "done": 0, "total": 0, "current": None}
        self.proc = None
        self.stop = False

    def log(self, text):
        with self.lock:
            self.state["log"] = (self.state["log"] + text)[-60000:]

    def start(self, data_paths, preset_ids, gen, prompt, limit, no_cache=False, prep=False):
        by_id = {p["id"]: p for p in presets()}
        tasks = [(d, by_id[m]) for d in data_paths for m in preset_ids if m in by_id]
        plan = []
        for d, preset in tasks:
            try:
                rows = len(fmt.read_input(HERE / d))
            except (ValueError, OSError):
                rows = 0
            rate, known = history_rate(preset["id"])
            plan.append({"rows": min(rows, limit) if limit else rows, "rate": rate, "known": known})
        with self.lock:
            if self.state["running"]:
                return "Прогон уже идёт"
            self.state = {"running": True, "log": "", "done": 0, "total": len(tasks), "current": None,
                          "started": time.time(), "plan": plan, "task_started": time.time(),
                          "labeled": any((HERE / d).with_name(Path(d).stem + ".gold.csv").exists() for d in data_paths)}
            self.stop = False
        threading.Thread(target=self._run, args=(tasks, gen, prompt, limit, no_cache, prep), daemon=True).start()
        return None

    def _run(self, tasks, gen, prompt, limit, no_cache, prep):
        for dataset, preset in tasks:
            if self.stop:
                self.log("\nОстановлено пользователем\n")
                break
            path = (HERE / dataset).resolve()
            cmd = [sys.executable, "-u", str(HERE / "run.py"), "predict", str(path), *preset["args"], "--gen", gen]
            if prompt and "llm" in preset["args"]:
                cmd += ["--prompt", prompt]
            if limit:
                cmd += ["--limit", str(limit)]
            if no_cache and preset["id"] != "charlr":
                cmd += ["--no-cache"]
            if prep:
                cmd += ["--prep"]
            with self.lock:
                self.state["current"] = f"{path.stem} × {preset['label']}"
                self.state["task_started"] = time.time()
            self.log(f"\n=== {path.stem} × {preset['label']}\n")
            start = time.perf_counter()
            self.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in self.proc.stdout:
                self.log(line)
            code = self.proc.wait()
            self.log(f"--- код выхода {code}, {time.perf_counter() - start:.0f} с\n")
            with self.lock:
                self.state["done"] += 1
        with self.lock:
            self.state["running"], self.state["current"] = False, None

    def cancel(self):
        self.stop = True
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()


JOB = Job()
SAFE = re.compile(r"[^\w.-]+")


def save_upload(filename, content):
    name = SAFE.sub("_", Path(filename).name)
    if not name.lower().endswith(".csv"):
        raise ValueError("Нужен CSV: testcase,description или эталон <имя>.gold.csv")
    INPUT.mkdir(parents=True, exist_ok=True)
    target = INPUT / name
    tmp = target.with_suffix(".upload")
    tmp.write_text(content, encoding="utf-8")
    try:
        if name.endswith(".gold.csv"):
            rows = len(fmt.read_answer(tmp))
        else:
            rows = len(fmt.read_input(tmp))
            labeled = fmt.embedded_labels(tmp)
            if labeled is not None:
                return attach_labels(name, tmp, labeled)
    except (ValueError, json.JSONDecodeError) as exc:
        tmp.unlink(missing_ok=True)
        raise ValueError(str(exc).replace(str(tmp), name)) from None
    tmp.replace(target)
    return {"saved": str(target.relative_to(HERE)), "rows": rows, "labeled": False}


def attach_labels(name, tmp, labeled):
    """A CSV with labels inside becomes its own dataset: <имя>.csv plus <имя>.gold.csv; the original goes to input/sources/."""
    base = Path(name).stem
    dataset = INPUT / f"{base}.csv"
    fmt.write_input(dataset, labeled)
    fmt.write_answer(INPUT / f"{base}.gold.csv", [{"testcase": r["testcase"], "labels": r["labels"],
                                                   "descriptions": ["-"] * len(r["labels"])} for r in labeled])
    (INPUT / "sources").mkdir(exist_ok=True)
    tmp.replace(INPUT / "sources" / name)
    return {"saved": str(dataset.relative_to(HERE)), "rows": len(labeled), "labeled": True, "dataset": base}


def delete_datasets(paths):
    """Deletes uploaded datasets and their gold files from input/; examples and runs are kept."""
    with JOB.lock:
        if JOB.state["running"]:
            raise ValueError("Идёт прогон, удалять датасеты сейчас нельзя")
    deleted = []
    for rel in paths:
        target = (HERE / rel).resolve()
        if target.parent != INPUT.resolve() or target.suffix != ".csv" or not target.exists():
            raise ValueError(f"{rel}: удалять можно только загруженные датасеты из input/")
        target.with_name(target.stem + ".gold.csv").unlink(missing_ok=True)
        target.unlink()
        deleted.append(target.stem)
    return {"deleted": deleted}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
        data = body if isinstance(body, bytes) else (
            body.encode() if isinstance(body, str) else json.dumps(body, ensure_ascii=False).encode())
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path == "/":
            return self._send(200, page(), "text/html; charset=utf-8")
        if path == "/api/state":
            prompts = sorted(p.stem for p in PROMPTS.glob("*.txt"))
            with JOB.lock:
                job = dict(JOB.state)
            return self._send(200, {"datasets": datasets(), "presets": presets(), "runs": runs(),
                                    "prompts": prompts, "job": job})
        if path == "/api/prompt":
            from urllib.parse import parse_qs
            q = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
            try:
                if q.get("version"):
                    return self._send(200, {"text": prompt_version(q["name"], q["version"])})
                return self._send(200, prompt_info(q.get("name")))
            except ValueError as exc:
                return self._send(400, {"error": str(exc)})
        if path.startswith("/download/"):
            target = (RUNS / path[len("/download/"):]).resolve()
            if RUNS.resolve() not in target.parents or not target.is_file():
                return self._send(404, {"error": "нет файла"})
            ds, run = target.parent.parent.name, target.parent.name
            return self._send(200, target.read_bytes(), "text/csv; charset=utf-8",
                              {"Content-Disposition": f'attachment; filename="{ds}__{run}.csv"'})
        self._send(404, {"error": "нет такого адреса"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        path = urlparse(self.path).path
        try:
            if path == "/api/upload":
                return self._send(200, save_upload(body["filename"], body["content"]))
            if path == "/api/run":
                if not body.get("datasets") or not body.get("models"):
                    raise ValueError("Выберите хотя бы один датасет и одну модель")
                error = JOB.start(body["datasets"], body["models"], body.get("gen", "extract"),
                                  body.get("prompt"), body.get("limit") or None, bool(body.get("no_cache")),
                                  bool(body.get("prep")))
                if error:
                    raise ValueError(error)
                return self._send(200, {"ok": True})
            if path == "/api/prompt":
                return self._send(200, write_prompt(body["name"], body["text"]))
            if path == "/api/prompt/restore":
                return self._send(200, write_prompt(body["name"], prompt_version(body["name"], body["version"])))
            if path == "/api/delete":
                return self._send(200, delete_datasets(body.get("datasets") or []))
            if path == "/api/cancel":
                JOB.cancel()
                return self._send(200, {"ok": True})
        except (ValueError, KeyError) as exc:
            return self._send(400, {"error": str(exc)})
        self._send(404, {"error": "нет такого адреса"})

    def log_message(self, *args):
        pass


PAGE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Прогон датасетов</title>
<style>
:root{--bg:#fff;--fg:#1b1b1b;--muted:#7a7a7a;--line:#e3e3e3;--soft:rgba(127,127,127,.07);--accent:#0a66c2;--bad:#b42318;--ok:#067647;--best:rgba(6,118,71,.07)}
@media (prefers-color-scheme:dark){:root{--bg:#161616;--fg:#ececec;--muted:#8f8f8f;--line:#2e2e2e;--accent:#5aa7ff;--bad:#ff7a6e;--ok:#5fd39a;--best:rgba(95,211,154,.08)}}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,system-ui,sans-serif;margin:0 auto;padding:20px 16px;max-width:1100px}
h1{font-size:19px;margin:0 0 14px}h2{font-size:14px;margin:0 0 8px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}@media(max-width:760px){.grid{grid-template-columns:1fr}}
.box{border:1px solid var(--line);border-radius:8px;padding:12px}
.list{border:1px solid var(--line);border-radius:6px;max-height:220px;overflow:auto}
.list label{display:flex;gap:8px;align-items:baseline;padding:6px 8px;border-bottom:1px solid var(--line);cursor:pointer}
.list label:last-child{border-bottom:0}.list label:hover{background:var(--soft)}
.list .main{flex:1;min-width:0}.list .side{color:var(--muted);font-size:12px;white-space:nowrap}
.empty{padding:8px;color:var(--muted)}
select,input,button,textarea{font:inherit}select,input[type=number]{background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:3px 6px}
button,.btn{background:var(--accent);color:#fff;border:1px solid var(--accent);border-radius:6px;padding:5px 12px;cursor:pointer;font-size:13px}
.ghost{background:transparent;color:var(--accent)}.danger{background:transparent;color:var(--bad);border-color:var(--bad)}
button:disabled{opacity:.45;cursor:default}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:10px}
.muted{color:var(--muted)}.bad{color:var(--bad)}.ok{color:var(--ok)}.msg{font-size:12px;margin-top:8px}
input[type=file]{display:none}
details.panel{margin-top:14px;border:1px solid var(--line);border-radius:8px}
details.panel>summary{cursor:pointer;padding:9px 12px;font-weight:600;font-size:14px}
details.panel .inner{padding:0 12px 12px}
textarea{width:100%;height:340px;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:8px;font:12px/1.5 ui-monospace,Menlo,monospace;resize:vertical}
pre{background:var(--soft);border-radius:6px;padding:10px;max-height:260px;overflow:auto;font-size:12px;white-space:pre-wrap;margin:0}
.results-head{display:flex;justify-content:space-between;align-items:center;margin:22px 0 8px}
.group{border:1px solid var(--line);border-radius:8px;margin-bottom:10px}
.group>summary{list-style:none;cursor:pointer;display:flex;gap:10px;align-items:baseline;padding:10px 12px;background:var(--soft);border-radius:8px}
.group[open]>summary{border-bottom:1px solid var(--line);border-radius:8px 8px 0 0}
.group>summary::-webkit-details-marker{display:none}
.chev{display:inline-block;width:10px;color:var(--muted);transition:transform .15s}.group[open] .chev{transform:rotate(90deg)}
.group h3{margin:0;font-size:14px}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:8px 12px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}
th{font-weight:500;color:var(--muted);font-size:12px}tbody tr:last-child td{border-bottom:0}td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
td.name{white-space:normal;width:40%}tr.best td{background:var(--best)}
a.dl{color:var(--accent);text-decoration:none}a.dl:hover{text-decoration:underline}
</style></head><body>
<h1>Прогон датасетов</h1>
<div class="grid">
 <div class="box"><h2>Датасеты</h2>
  <div class="list" id="ds"></div>
  <div class="row"><label class="btn ghost" for="file">Загрузить CSV</label><input type="file" id="file" accept=".csv" multiple>
   <button class="danger" id="del">Удалить выбранные</button></div>
  <div id="upmsg" class="msg"></div></div>
 <div class="box"><h2>Модели</h2>
  <div class="list" id="models"></div>
  <div class="row"><label>Промпт LLM <select id="prompt"></select></label>
   <label>Первые <input type="number" id="limit" min="1" style="width:70px" placeholder="все"> объявлений</label></div></div>
</div>
<details class="panel" id="editor"><summary>Редактировать промпт</summary><div class="inner">
 <div class="row" style="margin-top:0"><label>Версия <select id="version"></select></label>
  <button class="ghost" id="restore" disabled>Вернуть эту версию</button>
  <span id="pmsg" class="msg muted" style="margin:0"></span></div>
 <textarea id="ptext" spellcheck="false" style="margin-top:10px"></textarea>
 <div class="row"><button id="psave">Сохранить</button><span class="muted msg" style="margin:0">Перед сохранением текущий текст уходит в историю, его можно вернуть.</span></div>
</div></details>
<div class="row" style="margin-top:14px"><button id="run">Запустить</button><button class="ghost" id="cancel" disabled>Остановить</button><label title="Модель вызывается заново: время честное, запросы к API оплачиваются повторно"><input type="checkbox" id="nocache"> Не брать ответы из кэша</label><label title="Перед моделью: удалить инъекции для LLM, эмодзи, невидимые символы и латиницу внутри русских слов (private_eval/preprocess.py)"><input type="checkbox" id="prep"> Предобработка</label><span id="status" class="muted"></span></div>
<details class="panel" id="logbox" hidden><summary>Журнал прогона</summary><div class="inner"><pre id="log"></pre></div></details>
<div class="results-head"><h2 style="margin:0">Результаты</h2><span><button class="ghost" id="openall">Развернуть все</button> <button class="ghost" id="closeall">Свернуть все</button></span></div>
<div id="runs"></div>
<script>
const $=id=>document.getElementById(id);let first=true,lastRuns='',DS=[];
const store={get(k,d){try{return JSON.parse(localStorage.getItem(k))??d}catch{return d}},set(k,v){try{localStorage.setItem(k,JSON.stringify(v))}catch{}}};
const closed=new Set(store.get('closed',[]));
function plural(n,a,b,c){const m=n%10,h=n%100;return `${n} ${m==1&&h!=11?a:m>=2&&m<=4&&(h<12||h>14)?b:c}`}
const ads=n=>plural(n,'объявление','объявления','объявлений');
function esc(s){return String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
const checked=id=>[...document.querySelectorAll(`#${id} input:checked`)].map(i=>i.value);
const picked={ds:[],models:[]};
for(const id of ['ds','models'])document.getElementById(id).addEventListener('change',e=>{const v=e.target.value;picked[id]=picked[id].filter(x=>x!==v);if(e.target.checked)picked[id].push(v)});
const inOrder=id=>{const on=new Set(checked(id));return [...picked[id].filter(v=>on.has(v)),...checked(id).filter(v=>!picked[id].includes(v))]};
function fillList(id,items,val,main,side){
 const keep=new Set(checked(id));
 const html=items.map(it=>`<label><input type="checkbox" value="${esc(val(it))}"${keep.has(val(it))?' checked':''}><span class="main">${main(it)}</span><span class="side">${side(it)}</span></label>`).join('')||'<div class="empty">Пусто</div>';
 const el=$(id);if(el.dataset.html!==html){el.innerHTML=html;el.dataset.html=html}}
const date=t=>new Date(t*1000).toLocaleString('ru-RU',{day:'2-digit',month:'2-digit',year:'numeric',hour:'2-digit',minute:'2-digit'});
async function refresh(){
 const s=await (await fetch('/api/state')).json();DS=s.datasets;
 fillList('ds',s.datasets,d=>d.path,d=>`${esc(d.name)} <span class="muted">· ${d.error?'<span class="bad">файл не читается</span>':ads(d.rows)}${d.gold?' · есть метки':''}</span>`,d=>date(d.modified));
 fillList('models',s.presets,p=>p.id,p=>esc(p.label),p=>esc(p.where||''));
 if(first){const sel=$('prompt');sel.innerHTML=s.prompts.map(p=>`<option${p==='classify_v3_seo_on'?' selected':''}>${esc(p)}</option>`).join('');first=false;loadPrompt()}
 const j=s.job;$('run').disabled=j.running;$('cancel').disabled=!j.running;
 $('status').textContent=j.running?`${j.done+1} из ${j.total}: ${j.current||''}`:(j.total?`Готово: ${j.done} из ${j.total}`:'');
 if(j.log){$('logbox').hidden=false;if(j.running)$('logbox').open=true;const p=$('log'),bottom=p.scrollTop+p.clientHeight>=p.scrollHeight-20;p.textContent=j.log;if(bottom)p.scrollTop=p.scrollHeight}
 const key=JSON.stringify([s.runs,s.datasets]);if(key!==lastRuns){lastRuns=key;renderRuns(s.runs,s.datasets)}
}
const pct=x=>x==null?'—':(100*x).toFixed(1);
function renderRuns(runs,ds){
 if(!runs.length){$('runs').innerHTML='<div class="muted">Прогонов пока нет.</div>';return}
 const groups={};for(const r of runs)(groups[r.dataset]??=[]).push(r);
 const info=Object.fromEntries(ds.map(d=>[d.name,d]));
 $('runs').innerHTML=Object.entries(groups).sort().map(([name,rs])=>{
  rs.sort((a,b)=>(b.f1??-1)-(a.f1??-1)||b.created.localeCompare(a.created));
  const best=rs[0].f1;
  const count={};for(const r of rs)count[r.title]=(count[r.title]||0)+1;
  const rows=rs.map(r=>`<tr class="${best!=null&&r.f1===best?'best':''}"><td class="name">${esc(r.title)}${count[r.title]>1&&r.prompt?` <span class="muted">· ${esc(r.prompt)}</span>`:''}${r.errors?`<div class="bad" style="font-size:12px">${r.errors===r.llm_calls?'LLM не ответила ни разу, все ответы от Char-LR':`${r.errors} из ${r.llm_calls} вызовов LLM с ошибкой, там ответы от Char-LR`}</div>`:''}</td>
    <td class="n">${pct(r.f1)}</td><td class="n">${pct(r.macro)}</td><td class="n">${r.seconds} с</td><td class="n">${r.per_row} с</td><td class="n">${r.llm_calls}</td>
    <td class="n"><a class="dl" href="${esc(r.download)}">CSV</a></td></tr>`).join('');
  const d=info[name];
  return `<details class="group" data-name="${esc(name)}"${closed.has(name)?'':' open'}><summary><span class="chev">▶</span><h3>${esc(name)}</h3><span class="muted">${d?ads(d.rows):'файл удалён'}</span></summary>
   <div class="scroll"><table><thead><tr><th>Модель</th><th class="n">micro-F1, %</th><th class="n">macro-F1, %</th><th class="n">Время</th><th class="n">На объявление</th><th class="n">Вызовов LLM</th><th></th></tr></thead><tbody>${rows}</tbody></table></div></details>`}).join('');
 for(const el of document.querySelectorAll('details.group'))el.addEventListener('toggle',()=>{el.open?closed.delete(el.dataset.name):closed.add(el.dataset.name);store.set('closed',[...closed])});
}
function setAll(open){for(const el of document.querySelectorAll('details.group'))el.open=open}
$('openall').onclick=()=>setAll(true);$('closeall').onclick=()=>setAll(false);
let current='';
async function loadPrompt(info){
 const name=$('prompt').value;
 if(!info){const r=await fetch('/api/prompt?name='+encodeURIComponent(name));info=await r.json()}
 current=info.text;$('ptext').value=info.text;$('restore').disabled=true;
 $('version').innerHTML='<option value="">текущая</option>'+info.versions.map(v=>`<option value="${esc(v.id)}">${esc(v.time.replace(/(\\d+)-(\\d+)-(\\d+) (\\d+)-(\\d+)-(\\d+)/,'$3.$2.$1 $4:$5:$6'))}</option>`).join('');
 $('editor').querySelector('summary').textContent=`Редактировать промпт ${name}`;
}
$('prompt').onchange=()=>{$('pmsg').textContent='';loadPrompt()};
$('version').onchange=async()=>{const v=$('version').value;
 if(!v){$('ptext').value=current;$('restore').disabled=true;$('pmsg').textContent='';return}
 const r=await fetch(`/api/prompt?name=${encodeURIComponent($('prompt').value)}&version=${encodeURIComponent(v)}`);
 $('ptext').value=(await r.json()).text;$('restore').disabled=false;$('pmsg').textContent='Показана старая версия, она ещё не применена.'};
$('restore').onclick=async()=>{const r=await fetch('/api/prompt/restore',{method:'POST',body:JSON.stringify({name:$('prompt').value,version:$('version').value})});
 const d=await r.json();if(!r.ok){$('pmsg').innerHTML=`<span class="bad">${esc(d.error)}</span>`;return}loadPrompt(d);$('pmsg').textContent='Версия возвращена.'};
$('psave').onclick=async()=>{const r=await fetch('/api/prompt',{method:'POST',body:JSON.stringify({name:$('prompt').value,text:$('ptext').value})});
 const d=await r.json();if(!r.ok){$('pmsg').innerHTML=`<span class="bad">${esc(d.error)}</span>`;return}loadPrompt(d);$('pmsg').textContent='Сохранено.'};
$('file').onchange=async()=>{const files=[...$('file').files];const msgs=[];
 for(const f of files){const r=await fetch('/api/upload',{method:'POST',body:JSON.stringify({filename:f.name,content:await f.text()})});const d=await r.json();
  msgs.push(r.ok?`<span class="ok">${esc(f.name)}: ${ads(d.rows)}${d.labeled?' · есть метки':''}</span>`:`<span class="bad">${esc(f.name)}: ${esc(d.error)}</span>`)}
 $('upmsg').innerHTML=msgs.join('<br>');$('file').value='';refresh()};
$('del').onclick=async()=>{const sel=checked('ds');const items=DS.filter(d=>sel.includes(d.path));
 if(!items.length){alert('Отметьте датасеты в списке');return}
 const fixed=items.filter(d=>!d.deletable);if(fixed.length){alert(`Примеры организаторов удалить нельзя: ${fixed.map(d=>d.name).join(', ')}`);return}
 if(!confirm(`Удалить ${items.map(d=>d.name).join(', ')} вместе с метками? Результаты прогонов останутся.`))return;
 const r=await fetch('/api/delete',{method:'POST',body:JSON.stringify({datasets:items.map(d=>d.path)})});const d=await r.json();
 $('upmsg').innerHTML=r.ok?`<span class="muted">Удалено: ${esc(d.deleted.join(', '))}</span>`:`<span class="bad">${esc(d.error)}</span>`;refresh()};
$('run').onclick=async()=>{const body={datasets:inOrder('ds'),models:inOrder('models'),prompt:$('prompt').value,limit:parseInt($('limit').value)||null,no_cache:$('nocache').checked,prep:$('prep').checked};
 const r=await fetch('/api/run',{method:'POST',body:JSON.stringify(body)});if(!r.ok)alert((await r.json()).error);refresh()};
$('cancel').onclick=async()=>{await fetch('/api/cancel',{method:'POST'});refresh()};
refresh();setInterval(refresh,2000);
</script></body></html>
"""


SHOW = HERE / "show.html"


def page():
    """The base page plus the optional presentation overlay from show.html."""
    extra = SHOW.read_text(encoding="utf-8") if SHOW.exists() else ""
    return PAGE.replace("</body></html>", extra + "</body></html>")


def main():
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"UI: http://localhost:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
